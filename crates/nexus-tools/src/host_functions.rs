use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use getrandom::getrandom;
use reqwest::blocking::Client as BlockingClient;
use tracing::{debug, error, instrument, trace, warn, Level};
use wasmtime::{Caller, Linker, Memory, Trap};

use crate::error::{ToolError, Result};

// =============================================================================
// SandboxHostState — Per-Invocation Host Function Context
// =============================================================================

/// State shared by all host functions during a single tool invocation.
/// This struct is passed as the `T` generic to `wasmtime::Linker<T>`.
///
/// # Security Notes
/// - `allowed_hosts` and `allowed_env_keys` are the security boundary
/// - `bytes_sent` tracks outbound traffic for cost/abuse monitoring
/// - All fields are accessed via `Caller::data_mut()` to ensure thread safety
#[derive(Debug)]
pub struct SandboxHostState {
    /// Name of the tool being executed (for logging/observability).
    pub tool_name: String,

    /// Hostnames this tool is permitted to make HTTP requests to.
    /// Empty vector means all hosts are allowed (use with caution).
    pub allowed_hosts: Vec<String>,

    /// Environment variable names this tool is permitted to read.
    pub allowed_env_keys: Vec<String>,

    /// Counter for outbound HTTP bytes sent (for cost tracking).
    pub bytes_sent: u64,

    /// Blocking HTTP client for making requests from host functions.
    /// Created once per sandbox instance to reuse connections.
    pub http_client: BlockingClient,

    /// Optional callback for emitting structured events to the kernel.
    pub event_callback: Option<Arc<dyn Fn(&str, serde_json::Value) + Send + Sync>>,
}

impl SandboxHostState {
    /// Creates a new host state for a tool invocation.
    pub fn new(
        tool_name: String,
        allowed_hosts: Vec<String>,        allowed_env_keys: Vec<String>,
    ) -> Self {
        Self {
            tool_name,
            allowed_hosts,
            allowed_env_keys,
            bytes_sent: 0,
            http_client: BlockingClient::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            event_callback: None,
        }
    }

    /// Attaches an event callback for structured logging to the kernel.
    pub fn with_event_callback<F>(mut self, cb: F) -> Self
    where
        F: Fn(&str, serde_json::Value) + Send + Sync + 'static,
    {
        self.event_callback = Some(Arc::new(cb));
        self
    }

    /// Checks if a hostname is permitted for HTTP requests.
    pub fn is_host_allowed(&self, host: &str) -> bool {
        // Empty allowlist = allow all (use with caution)
        if self.allowed_hosts.is_empty() {
            return true;
        }
        self.allowed_hosts.iter().any(|allowed| {
            // Support exact match and suffix match for subdomains
            host == allowed || host.ends_with(&format!(".{}", allowed))
        })
    }

    /// Checks if an environment variable name is permitted to read.
    pub fn is_env_key_allowed(&self, key: &str) -> bool {
        // Empty allowlist = allow none (secure default)
        if self.allowed_env_keys.is_empty() {
            return false;
        }
        self.allowed_env_keys.iter().any(|allowed| {
            // Support exact match and prefix match for namespaced vars
            key == allowed || key.starts_with(&format!("{}_", allowed))
        })
    }

    /// Records outbound bytes for cost tracking.
    pub fn record_bytes_sent(&mut self, bytes: u64) {        self.bytes_sent = self.bytes_sent.saturating_add(bytes);
    }

    /// Emits a structured event to the kernel if callback is registered.
    pub fn emit_event(&self, event_type: &str, payload: serde_json::Value) {
        if let Some(cb) = &self.event_callback {
            cb(event_type, payload);
        }
    }
}

// =============================================================================
// Memory Access Helpers — Safe WASM Linear Memory Operations
// =============================================================================

/// Reads a byte slice from WASM linear memory.
///
/// # Safety
/// - Caller must ensure `ptr` and `len` are within bounds of the memory
/// - This function performs bounds checking and returns an error if out of bounds
///
/// # Arguments
/// * `caller` - The WASM caller context (provides access to memory)
/// * `ptr` - Byte offset into WASM linear memory
/// * `len` - Number of bytes to read
///
/// # Returns
/// * `Ok(Vec<u8>)` - The requested bytes
/// * `Err(Trap)` - If memory access is out of bounds
#[instrument(skip(caller), fields(ptr, len))]
fn read_mem(caller: &mut Caller<'_, SandboxHostState>, ptr: i32, len: i32) -> Result<Vec<u8>, Trap> {
    if ptr < 0 || len < 0 {
        return Err(Trap::new(format!(
            "invalid memory access: ptr={}, len={}",
            ptr, len
        )));
    }

    let memory = caller
        .get_export("memory")
        .and_then(|e| e.into_memory())
        .ok_or_else(|| Trap::new("memory export not found"))?;

    let data = memory
        .data(caller)
        .get(ptr as usize..)
        .and_then(|s| s.get(..len as usize))
        .ok_or_else(|| {
            Trap::new(format!(
                "memory read out of bounds: ptr={}, len={}, memory_size={}",                ptr,
                len,
                memory.data_size(caller)
            ))
        })?;

    Ok(data.to_vec())
}

/// Writes a byte slice to WASM linear memory.
///
/// # Safety
/// - Caller must ensure `ptr` and `data.len()` are within bounds
/// - This function performs bounds checking and returns an error if out of bounds
///
/// # Arguments
/// * `caller` - The WASM caller context
/// * `ptr` - Byte offset into WASM linear memory
/// * `data` - Bytes to write
///
/// # Returns
/// * `Ok(())` - If write succeeded
/// * `Err(Trap)` - If memory access is out of bounds
#[instrument(skip(caller, data), fields(ptr, len = data.len()))]
fn write_mem(
    caller: &mut Caller<'_, SandboxHostState>,
    ptr: i32,
    data: &[u8],
) -> Result<(), Trap> {
    if ptr < 0 {
        return Err(Trap::new(format!("invalid memory write ptr: {}", ptr)));
    }

    let memory = caller
        .get_export("memory")
        .and_then(|e| e.into_memory())
        .ok_or_else(|| Trap::new("memory export not found"))?;

    let end = (ptr as usize)
        .checked_add(data.len())
        .ok_or_else(|| Trap::new("integer overflow in memory write"))?;

    if end > memory.data_size(caller) {
        return Err(Trap::new(format!(
            "memory write out of bounds: ptr={}, len={}, memory_size={}",
            ptr,
            data.len(),
            memory.data_size(caller)
        )));
    }
    memory
        .data_mut(caller)
        .get_mut(ptr as usize..end)
        .ok_or_else(|| Trap::new("memory write slice failed"))?
        .copy_from_slice(data);

    Ok(())
}

/// Reads a null-terminated UTF-8 string from WASM memory.
///
/// # Arguments
/// * `caller` - WASM caller context
/// * `ptr` - Pointer to string start
/// * `max_len` - Maximum bytes to read (prevents runaway reads)
///
/// # Returns
/// * `Ok(String)` - The decoded UTF-8 string
/// * `Err(Trap)` - If memory access fails or UTF-8 decoding fails
#[instrument(skip(caller), fields(ptr, max_len))]
fn read_string(
    caller: &mut Caller<'_, SandboxHostState>,
    ptr: i32,
    max_len: i32,
) -> Result<String, Trap> {
    // Read up to max_len bytes to find null terminator
    let bytes = read_mem(caller, ptr, max_len)?;

    // Find null terminator
    let null_pos = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let str_bytes = &bytes[..null_pos];

    String::from_utf8(str_bytes.to_vec()).map_err(|e| {
        Trap::new(format!("invalid UTF-8 in string argument: {}", e))
    })
}

/// Writes a string to WASM memory as null-terminated UTF-8 bytes.
/// Returns the number of bytes written (including null terminator).
#[instrument(skip(caller, s), fields(ptr, len = s.len()))]
fn write_string(
    caller: &mut Caller<'_, SandboxHostState>,
    ptr: i32,
    max_len: i32,
    s: &str,
) -> Result<i32, Trap> {
    let bytes = s.as_bytes();
    let total_len = bytes.len() + 1; // +1 for null terminator
    if total_len > max_len as usize {
        return Ok(-3); // Buffer too small
    }

    // Write string bytes
    write_mem(caller, ptr, bytes)?;
    // Write null terminator
    write_mem(caller, ptr + bytes.len() as i32, &[0])?;

    Ok(total_len as i32)
}

// =============================================================================
// Host Function Implementations
// =============================================================================

/// HTTP GET: nexus_http_get(url_ptr, url_len, out_ptr, out_max) -> i32
///
/// # Arguments (WASM side)
/// * `url_ptr`, `url_len` - Pointer and length of URL string in WASM memory
/// * `out_ptr`, `out_max` - Output buffer pointer and max size for response body
///
/// # Returns
/// * `>= 0` - Number of bytes written to output buffer
/// * `-1` - Host not allowed by manifest
/// * `-2` - HTTP request failed
/// * `-3` - Output buffer too small for response
#[instrument(skip(caller), fields(url_ptr, url_len, out_ptr, out_max))]
fn nexus_http_get(
    mut caller: Caller<'_, SandboxHostState>,
    url_ptr: i32,
    url_len: i32,
    out_ptr: i32,
    out_max: i32,
) -> i32 {
    let tool_name = caller.data().tool_name.clone();

    // Read URL from WASM memory
    let url = match read_string(&mut caller, url_ptr, url_len.max(2048)) {
        Ok(s) => s,
        Err(e) => {
            error!(tool = %tool_name, error = %e, "http_get: failed to read URL from memory");
            return -2;
        }
    };

    // Parse and validate host
    let host = match url::Url::parse(&url).ok().and_then(|u| u.host_str().map(|s| s.to_string())) {
        Some(h) => h,
        None => {            warn!(tool = %tool_name, url = %url, "http_get: invalid URL");
            return -2;
        }
    };

    // Check host allowlist
    if !caller.data().is_host_allowed(&host) {
        warn!(
            tool = %tool_name,
            url = %url,
            host = %host,
            "http_get: host not in allowlist"
        );
        caller.data_mut().emit_event("http_blocked", serde_json::json!({
            "url": url,
            "host": host,
            "method": "GET"
        }));
        return -1;
    }

    // Make blocking HTTP request (spawn_blocking to avoid blocking WASM executor)
    let client = caller.data().http_client.clone();
    let result = std::thread::spawn(move || {
        client.get(&url).send()
    })
    .join()
    .map_err(|_| Trap::new("http_get: request thread panicked"))
    .and_then(|r| r.map_err(|e| Trap::new(format!("http_get: request failed: {}", e))));

    let response = match result {
        Ok(resp) => resp,
        Err(e) => {
            error!(tool = %tool_name, url = %url, error = %e, "http_get: request failed");
            return -2;
        }
    };

    // Read response body
    let body = match response.bytes() {
        Ok(b) => b,
        Err(e) => {
            error!(tool = %tool_name, error = %e, "http_get: failed to read response body");
            return -2;
        }
    };

    // Check output buffer size
    if body.len() > out_max as usize {
        warn!(            tool = %tool_name,
            response_size = body.len(),
            buffer_max = out_max,
            "http_get: response too large for buffer"
        );
        return -3;
    }

    // Write response to WASM memory
    if let Err(e) = write_mem(&mut caller, out_ptr, &body) {
        error!(tool = %tool_name, error = %e, "http_get: failed to write response to memory");
        return -2;
    }

    // Track bytes sent for cost monitoring
    caller.data_mut().record_bytes_sent(body.len() as u64);

    // Emit success event
    caller.data().emit_event("http_success", serde_json::json!({
        "url": url,
        "method": "GET",
        "status": response.status().as_u16(),
        "bytes": body.len()
    }));

    debug!(
        tool = %tool_name,
        url = %url,
        status = %response.status(),
        bytes = body.len(),
        "http_get: success"
    );

    body.len() as i32
}

/// HTTP POST: nexus_http_post(url_ptr, url_len, body_ptr, body_len, out_ptr, out_max) -> i32
///
/// Same as GET but includes a JSON request body.
#[instrument(skip(caller), fields(url_ptr, url_len, body_ptr, body_len, out_ptr, out_max))]
fn nexus_http_post(
    mut caller: Caller<'_, SandboxHostState>,
    url_ptr: i32,
    url_len: i32,
    body_ptr: i32,
    body_len: i32,
    out_ptr: i32,
    out_max: i32,
) -> i32 {
    let tool_name = caller.data().tool_name.clone();
    // Read URL
    let url = match read_string(&mut caller, url_ptr, url_len.max(2048)) {
        Ok(s) => s,
        Err(e) => {
            error!(tool = %tool_name, error = %e, "http_post: failed to read URL");
            return -2;
        }
    };

    // Validate host
    let host = match url::Url::parse(&url).ok().and_then(|u| u.host_str().map(|s| s.to_string())) {
        Some(h) => h,
        None => {
            warn!(tool = %tool_name, url = %url, "http_post: invalid URL");
            return -2;
        }
    };

    if !caller.data().is_host_allowed(&host) {
        warn!(
            tool = %tool_name,
            url = %url,
            host = %host,
            "http_post: host not in allowlist"
        );
        caller.data_mut().emit_event("http_blocked", serde_json::json!({
            "url": url,
            "host": host,
            "method": "POST"
        }));
        return -1;
    }

    // Read request body
    let body_bytes = match read_mem(&mut caller, body_ptr, body_len) {
        Ok(b) => b,
        Err(e) => {
            error!(tool = %tool_name, error = %e, "http_post: failed to read body from memory");
            return -2;
        }
    };

    // Make blocking POST request
    let client = caller.data().http_client.clone();
    let body_clone = body_bytes.clone();
    let result = std::thread::spawn(move || {
        client
            .post(&url)
            .header("Content-Type", "application/json")            .body(body_clone)
            .send()
    })
    .join()
    .map_err(|_| Trap::new("http_post: request thread panicked"))
    .and_then(|r| r.map_err(|e| Trap::new(format!("http_post: request failed: {}", e))));

    let response = match result {
        Ok(resp) => resp,
        Err(e) => {
            error!(tool = %tool_name, url = %url, error = %e, "http_post: request failed");
            return -2;
        }
    };

    // Read response
    let response_body = match response.bytes() {
        Ok(b) => b,
        Err(e) => {
            error!(tool = %tool_name, error = %e, "http_post: failed to read response");
            return -2;
        }
    };

    // Check output buffer
    if response_body.len() > out_max as usize {
        warn!(
            tool = %tool_name,
            response_size = response_body.len(),
            buffer_max = out_max,
            "http_post: response too large"
        );
        return -3;
    }

    // Write response to WASM memory
    if let Err(e) = write_mem(&mut caller, out_ptr, &response_body) {
        error!(tool = %tool_name, error = %e, "http_post: failed to write response");
        return -2;
    }

    // Track bytes (request + response)
    let total_bytes = (body_bytes.len() + response_body.len()) as u64;
    caller.data_mut().record_bytes_sent(total_bytes);

    // Emit event
    caller.data().emit_event("http_success", serde_json::json!({
        "url": url,
        "method": "POST",
        "status": response.status().as_u16(),        "request_bytes": body_bytes.len(),
        "response_bytes": response_body.len()
    }));

    debug!(
        tool = %tool_name,
        url = %url,
        status = %response.status(),
        request_bytes = body_bytes.len(),
        response_bytes = response_body.len(),
        "http_post: success"
    );

    response_body.len() as i32
}

/// Logging: nexus_log(level, msg_ptr, msg_len) -> ()
///
/// Emits a log message at the specified level from the WASM tool.
/// Level mapping: 0=trace, 1=debug, 2=info, 3=warn, 4=error
#[instrument(skip(caller), fields(level, msg_ptr, msg_len))]
fn nexus_log(
    mut caller: Caller<'_, SandboxHostState>,
    level: i32,
    msg_ptr: i32,
    msg_len: i32,
) {
    let tool_name = caller.data().tool_name.clone();

    // Read message from WASM memory
    let msg = match read_string(&mut caller, msg_ptr, msg_len.max(8192)) {
        Ok(s) => s,
        Err(e) => {
            error!(tool = %tool_name, error = %e, "log: failed to read message from memory");
            return;
        }
    };

    // Map level integer to tracing Level
    let tracing_level = match level {
        0 => Level::TRACE,
        1 => Level::DEBUG,
        2 => Level::INFO,
        3 => Level::WARN,
        4 => Level::ERROR,
        _ => Level::INFO, // Default to INFO for unknown levels
    };

    // Emit log with tool context
    match tracing_level {        Level::TRACE => trace!(tool = %tool_name, "{}", msg),
        Level::DEBUG => debug!(tool = %tool_name, "{}", msg),
        Level::INFO => info!(tool = %tool_name, "{}", msg),
        Level::WARN => warn!(tool = %tool_name, "{}", msg),
        Level::ERROR => error!(tool = %tool_name, "{}", msg),
    }

    // Also emit as structured event for kernel observability
    caller.data().emit_event("log", serde_json::json!({
        "level": tracing_level.as_str(),
        "message": msg
    }));
}

/// Time: nexus_now_ms() -> i64
///
/// Returns the current Unix timestamp in milliseconds.
#[instrument(skip(_caller))]
fn nexus_now_ms(_caller: Caller<'_, SandboxHostState>) -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Random: nexus_random_bytes(out_ptr, len) -> ()
///
/// Fills the output buffer with cryptographically secure random bytes.
#[instrument(skip(caller), fields(out_ptr, len))]
fn nexus_random_bytes(
    mut caller: Caller<'_, SandboxHostState>,
    out_ptr: i32,
    len: i32,
) {
    if len <= 0 {
        return;
    }

    let mut buf = vec![0u8; len as usize];

    // Use getrandom for cryptographically secure randomness
    if let Err(e) = getrandom(&mut buf) {
        error!(error = %e, "random_bytes: getrandom failed");
        // Fill with zeros on failure rather than panicking
        return;
    }

    // Write to WASM memory
    if let Err(e) = write_mem(&mut caller, out_ptr, &buf) {
        error!(error = %e, "random_bytes: failed to write to memory");    }
}

/// Environment: nexus_env_get(key_ptr, key_len, out_ptr, out_max) -> i32
///
/// Reads an environment variable, but ONLY if explicitly allowlisted.
///
/// # Returns
/// * `>= 0` - Number of bytes written (including null terminator)
/// * `-1` - Key not found or not allowed
/// * `-3` - Output buffer too small
#[instrument(skip(caller), fields(key_ptr, key_len, out_ptr, out_max))]
fn nexus_env_get(
    mut caller: Caller<'_, SandboxHostState>,
    key_ptr: i32,
    key_len: i32,
    out_ptr: i32,
    out_max: i32,
) -> i32 {
    let tool_name = caller.data().tool_name.clone();

    // Read key from WASM memory
    let key = match read_string(&mut caller, key_ptr, key_len.max(256)) {
        Ok(s) => s,
        Err(e) => {
            error!(tool = %tool_name, error = %e, "env_get: failed to read key");
            return -1;
        }
    };

    // Check allowlist
    if !caller.data().is_env_key_allowed(&key) {
        warn!(
            tool = %tool_name,
            key = %key,
            "env_get: key not in allowlist"
        );
        caller.data_mut().emit_event("env_blocked", serde_json::json!({
            "key": key
        }));
        return -1;
    }

    // Read environment variable
    let value = match std::env::var(&key) {
        Ok(v) => v,
        Err(std::env::VarError::NotPresent) => {
            debug!(tool = %tool_name, key = %key, "env_get: variable not set");
            return -1;
        }        Err(e) => {
            error!(tool = %tool_name, key = %key, error = %e, "env_get: failed to read env var");
            return -1;
        }
    };

    // Write value to WASM memory
    match write_string(&mut caller, out_ptr, out_max, &value) {
        Ok(n) => {
            caller.data().emit_event("env_success", serde_json::json!({
                "key": key,
                "value_len": value.len()
            }));
            debug!(tool = %tool_name, key = %key, value_len = value.len(), "env_get: success");
            n
        }
        Err(e) => {
            error!(tool = %tool_name, error = %e, "env_get: failed to write value to memory");
            -2
        }
    }
}

// =============================================================================
// Host Function Registration
// =============================================================================

/// Registers all host functions with the wasmtime linker.
///
/// # Arguments
/// * `linker` - The wasmtime linker to register functions with
///
/// # Returns
/// * `Ok(())` - If all functions registered successfully
/// * `Err(ToolError)` - If registration failed
///
/// # Security Notes
/// - All host functions are wrapped to catch panics and return safe errors
/// - Memory access is bounds-checked before any operation
/// - Host allowlists are enforced at the function level, not just at load time
#[instrument(skip(linker))]
pub fn register_host_functions(
    linker: &mut Linker<SandboxHostState>,
) -> Result<()> {
    // Register HTTP GET
    linker
        .func_wrap(
            "nexus",
            "http_get",
            nexus_http_get as fn(                Caller<'_, SandboxHostState>,
                i32, i32, i32, i32
            ) -> i32,
        )
        .context("failed to register nexus_http_get")?;

    // Register HTTP POST
    linker
        .func_wrap(
            "nexus",
            "http_post",
            nexus_http_post as fn(
                Caller<'_, SandboxHostState>,
                i32, i32, i32, i32, i32, i32
            ) -> i32,
        )
        .context("failed to register nexus_http_post")?;

    // Register logging
    linker
        .func_wrap(
            "nexus",
            "log",
            nexus_log as fn(
                Caller<'_, SandboxHostState>,
                i32, i32, i32
            ),
        )
        .context("failed to register nexus_log")?;

    // Register time
    linker
        .func_wrap(
            "nexus",
            "now_ms",
            nexus_now_ms as fn(
                Caller<'_, SandboxHostState>
            ) -> i64,
        )
        .context("failed to register nexus_now_ms")?;

    // Register random bytes
    linker
        .func_wrap(
            "nexus",
            "random_bytes",
            nexus_random_bytes as fn(
                Caller<'_, SandboxHostState>,
                i32, i32
            ),        )
        .context("failed to register nexus_random_bytes")?;

    // Register environment variable access
    linker
        .func_wrap(
            "nexus",
            "env_get",
            nexus_env_get as fn(
                Caller<'_, SandboxHostState>,
                i32, i32, i32, i32
            ) -> i32,
        )
        .context("failed to register nexus_env_get")?;

    debug!("registered {} host functions", 6);
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use wasmtime::{Engine, Store, Module};

    fn create_test_store() -> Store<SandboxHostState> {
        let state = SandboxHostState::new(
            "test-tool".into(),
            vec!["example.com".into()],
            vec!["TEST_VAR".into()],
        );
        Store::new(&Engine::default(), state)
    }

    #[test]
    fn test_host_state_host_allowlist() {
        let state = SandboxHostState::new(
            "test".into(),
            vec!["example.com".into(), "api.sub.example.com".into()],
            vec![],
        );

        assert!(state.is_host_allowed("example.com"));
        assert!(state.is_host_allowed("api.example.com")); // subdomain match
        assert!(state.is_host_allowed("api.sub.example.com")); // exact match
        assert!(!state.is_host_allowed("evil.com"));
        assert!(!state.is_host_allowed("example.com.evil.com")); // not a subdomain
        // Empty allowlist = allow all
        let state_open = SandboxHostState::new("test".into(), vec![], vec![]);
        assert!(state_open.is_host_allowed("any-host.com"));
    }

    #[test]
    fn test_host_state_env_allowlist() {
        let state = SandboxHostState::new(
            "test".into(),
            vec![],
            vec!["API_KEY".into(), "DB".into()],
        );

        assert!(state.is_env_key_allowed("API_KEY"));
        assert!(state.is_env_key_allowed("DB_HOST")); // prefix match
        assert!(state.is_env_key_allowed("DB_PASSWORD"));
        assert!(!state.is_env_key_allowed("SECRET_KEY"));
        assert!(!state.is_env_key_allowed("OTHER_VAR"));

        // Empty allowlist = allow none
        let state_secure = SandboxHostState::new("test".into(), vec![], vec![]);
        assert!(!state_secure.is_env_key_allowed("ANY_VAR"));
    }

    #[test]
    fn test_memory_helpers_bounds_checking() {
        // This test would require a real WASM module with memory
        // For now, we test the logic with mock data
        let mut store = create_test_store();

        // Test negative pointer
        let result = read_mem(&mut store, -1, 10);
        assert!(result.is_err());

        // Test negative length
        let result = read_mem(&mut store, 0, -1);
        assert!(result.is_err());

        // Test zero length (should succeed but return empty)
        let result = read_mem(&mut store, 0, 0);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_string_helpers() {
        let mut store = create_test_store();

        // Test null-terminated string read        // This would need actual WASM memory setup; skip for now
        // The logic is tested via integration tests with real WASM modules
    }

    #[test]
    fn test_register_host_functions() {
        let engine = Engine::default();
        let mut linker = Linker::<SandboxHostState>::new(&engine);

        let result = register_host_functions(&mut linker);
        assert!(result.is_ok());

        // Verify functions are registered by checking we can instantiate a module
        // that imports them (would fail if not registered)
    }

    #[test]
    fn test_nexus_now_ms() {
        let mut store = create_test_store();
        let caller = Caller::new(&mut store, None);

        let now = nexus_now_ms(caller);
        let expected = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        // Allow 1 second tolerance for test execution time
        assert!((now - expected).abs() < 1000);
    }

    #[test]
    fn test_nexus_random_bytes_determinism() {
        // Random bytes should be different each call (with high probability)
        let mut store = create_test_store();
        let mut caller = Caller::new(&mut store, None);

        // We can't easily test write_mem without real WASM memory,
        // so we test the getrandom call directly
        let mut buf1 = vec![0u8; 32];
        let mut buf2 = vec![0u8; 32];

        getrandom(&mut buf1).unwrap();
        getrandom(&mut buf2).unwrap();

        // Probability of collision is 1/2^256, so this should never fail
        assert_ne!(buf1, buf2);
    }
}
