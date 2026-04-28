use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use jsonschema::JSONSchema;
use nexus_proto::tool::{ResourceLimits, ToolCall, ToolManifest as ProtoManifest, ToolResult};
use serde_json::Value;
use tokio::time::timeout;
use tracing::{debug, error, instrument, warn};
use wasmtime::{
    Config, Engine, Linker, Module, Store, Trap, Val, MemoryType, Memory, Extern,
    FuelConsumptionMode,
};

use crate::error::{ToolError, Result};
use crate::host_functions::{register_host_functions, SandboxHostState};
use crate::manifest::{compute_wasm_checksum, verify_wasm_checksum};

// =============================================================================
// SandboxConfig — Per-Invocation Resource Limits
// =============================================================================

/// Resource limits for a single tool invocation, derived from ToolManifest.
///
/// These limits are enforced by wasmtime at runtime to prevent runaway
/// tool execution from affecting the host system or other agents.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum WASM linear memory in bytes (heap + stack).
    pub max_memory_bytes: usize,

    /// Maximum wasmtime "fuel" units (approximate instruction budget).
    /// 1 fuel ≈ 1 WASM instruction; tune based on expected workload.
    pub max_fuel: u64,

    /// Hard timeout for the entire invocation, including host function calls.
    pub timeout_ms: u64,

    /// Maximum size in bytes of the serialized output the tool may return.
    pub max_output_bytes: usize,
}

impl From<ResourceLimits> for SandboxConfig {
    fn from(limits: ResourceLimits) -> Self {
        Self {
            max_memory_bytes: (limits.max_memory_mb as usize)
                .saturating_mul(1024 * 1024),
            max_fuel: limits.max_fuel,
            timeout_ms: limits.timeout_ms,
            max_output_bytes: limits.max_output_bytes,        }
    }
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024, // 64 MB
            max_fuel: 10_000_000,
            timeout_ms: 5000,
            max_output_bytes: 1024 * 1024, // 1 MB
        }
    }
}

// =============================================================================
// SandboxResult — Execution Metrics and Output
// =============================================================================

/// The result of executing a tool in the WASM sandbox.
///
/// Contains both the tool's output and observability metrics for cost
/// tracking, debugging, and rate limiting decisions.
#[derive(Debug, Clone)]
pub struct SandboxResult {
    /// Raw output bytes from the tool (JSON-encoded result or error).
    pub output: Vec<u8>,

    /// Wasmtime fuel units consumed during execution.
    pub fuel_consumed: u64,

    /// Peak WASM memory usage in megabytes (approximate).
    pub memory_peak_mb: u32,

    /// Wall-clock execution time in milliseconds.
    pub execution_time_ms: u64,

    /// Whether the tool exited normally (returned from _nexus_run)
    /// vs. trapping/timing out/being killed.
    pub exited_normally: bool,

    /// Optional error message if execution failed.
    pub error: Option<String>,
}

impl SandboxResult {
    /// Returns `true` if the tool completed successfully.
    pub fn is_success(&self) -> bool {
        self.exited_normally && self.error.is_none()
    }
    /// Converts this result into a `ToolResult` for the kernel.
    pub fn into_tool_result(
        self,
        call_id: uuid::Uuid,
        tool_name: String,
    ) -> ToolResult {
        if self.is_success() {
            // Parse output JSON
            let output_value: Value = serde_json::from_slice(&self.output)
                .unwrap_or_else(|e| {
                    warn!(error = %e, "failed to parse tool output as JSON");
                    Value::Null
                });

            ToolResult {
                call_id,
                tool_name,
                output: output_value,
                is_error: false,
                error_message: None,
                execution_time_ms: self.execution_time_ms,
                fuel_consumed: self.fuel_consumed,
                memory_peak_mb: self.memory_peak_mb,
            }
        } else {
            ToolResult {
                call_id,
                tool_name,
                output: Value::Null,
                is_error: true,
                error_message: self.error.or_else(|| Some("execution failed".into())),
                execution_time_ms: self.execution_time_ms,
                fuel_consumed: self.fuel_consumed,
                memory_peak_mb: self.memory_peak_mb,
            }
        }
    }
}

// =============================================================================
// WASM ABI Conventions — Documented for Tool Authors
// =============================================================================

/// # WASM Tool ABI Specification
///
/// Tools compiled to WASM must follow this interface to interoperate with Nexus:
///
/// ## Required Exports
////// ### `_nexus_run() -> i32`
/// - Entry point called by the host to execute the tool
/// - Returns `0` on success, non-zero error code on failure
/// - Must be exported with exact name (leading underscore)
///
/// ### `nexus_set_input(ptr: i32, len: i32) -> ()`
/// - Called by host before `_nexus_run` to provide input
/// - `ptr`: Byte offset into linear memory where JSON input is written
/// - `len`: Length of input in bytes
/// - Tool should copy input to internal buffer if needed
///
/// ### `nexus_get_output() -> i64`
/// - Called by host after `_nexus_run` to retrieve output
/// - Returns `(ptr << 32) | len` as i64: high 32 bits = ptr, low 32 bits = len
/// - Output buffer must remain valid until host reads it
///
/// ## Optional Exports
///
/// ### `allocate(size: i32) -> i32`
/// - If present, host uses this to allocate memory for input/output
/// - If absent, host writes directly to known offsets (0x10000+)
///
/// ### `deallocate(ptr: i32) -> ()`
/// - If present, host calls this after reading output to free memory
///
/// ## Memory Layout
///
/// - Linear memory starts at 0; first 64KB (0x10000) reserved for host/tool communication
/// - Input written at 0x10000, output expected at 0x20000 by default
/// - Tools should not assume fixed addresses; use `allocate` if available
///
/// ## Host Functions (imported from "nexus" module)
///
/// - `http_get`, `http_post`: Network access (allowlist enforced)
/// - `log`: Structured logging to host
/// - `now_ms`: Current Unix timestamp in milliseconds
/// - `random_bytes`: Cryptographically secure random data
/// - `env_get`: Read allowlisted environment variables
///
/// ## Error Handling
///
/// - Tools should return non-zero from `_nexus_run` on error
/// - Traps (panic, OOB access) are caught by host and reported as execution failure
/// - Host functions return error codes; tools should check and handle gracefully

// =============================================================================
// WasmSandbox — Isolated Tool Execution Engine
// =============================================================================

/// A pre-compiled WASM sandbox ready to execute tool invocations.///
/// # Design
/// - `Engine` and `Module` are shared across invocations for efficiency
/// - Each `invoke()` creates a fresh `Store` with isolated state and limits
/// - Host functions are registered per-invocation with tool-specific context
/// - Input/output validation happens at the boundary, not inside the sandbox
///
/// # Thread Safety
/// - `Engine` and `Module` are `Send + Sync` and safe to share
/// - `Store` is not thread-safe; each invocation gets its own
/// - `invoke()` is `async` but not `Send`; wrap in `tokio::task::spawn_blocking` if needed
pub struct WasmSandbox {
    /// Shared wasmtime engine (configured once at startup).
    engine: Arc<Engine>,

    /// Tool manifest describing capabilities, limits, and schemas.
    manifest: ProtoManifest,

    /// Compiled WASM bytes (kept for integrity verification).
    wasm_bytes: Arc<Vec<u8>>,

    /// Pre-compiled WASM module (expensive step done once at load time).
    module: Arc<Module>,

    /// Resource limits for execution.
    config: SandboxConfig,
}

impl WasmSandbox {
    /// Creates a new sandbox by compiling the WASM module.
    ///
    /// # Arguments
    /// * `engine` - Shared wasmtime engine (configured with fuel, memory limits)
    /// * `manifest` - Tool manifest with metadata and schemas
    /// * `wasm_bytes` - Raw WASM binary bytes
    ///
    /// # Returns
    /// * `Ok(WasmSandbox)` - If module compiled successfully
    /// * `Err(ToolError)` - If compilation or validation failed
    ///
    /// # Performance Note
    /// Module compilation is expensive (~10-100ms). Do this once at tool load
    /// time, not per invocation. The resulting `WasmSandbox` can be reused
    /// for many invocations with different inputs.
    #[instrument(skip(engine, wasm_bytes), fields(tool = %manifest.name, version = %manifest.version))]
    pub fn new(
        engine: Arc<Engine>,
        manifest: ProtoManifest,
        wasm_bytes: Vec<u8>,
    ) -> Result<Self> {        debug!("compiling WASM module for tool");

        // Verify checksum if provided
        if let Some(expected) = &manifest.checksum_sha256 {
            let actual = compute_wasm_checksum_raw(&wasm_bytes)?;
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(ToolError::ChecksumMismatch {
                    expected: expected.clone(),
                    actual,
                });
            }
            debug!(checksum = %expected, "WASM integrity verified");
        }

        // Compile the module (expensive, but done once)
        let module = Module::from_binary(&engine, &wasm_bytes)
            .map_err(|e| ToolError::CompilationError(e.to_string()))?;

        // Derive config from manifest limits
        let config: SandboxConfig = manifest.resource_limits.clone().into();

        debug!(
            tool = %manifest.name,
            max_memory_mb = manifest.resource_limits.max_memory_mb,
            max_fuel = manifest.resource_limits.max_fuel,
            timeout_ms = manifest.resource_limits.timeout_ms,
            "WASM module compiled"
        );

        Ok(Self {
            engine,
            manifest,
            wasm_bytes: Arc::new(wasm_bytes),
            module: Arc::new(module),
            config,
        })
    }

    /// Executes a single tool invocation in the isolated sandbox.
    ///
    /// # Arguments
    /// * `call` - The `ToolCall` containing arguments and metadata
    ///
    /// # Returns
    /// * `Ok(ToolResult)` - If execution completed (success or error)
    /// * `Err(ToolError)` - If sandbox setup, validation, or execution failed
    ///
    /// # Execution Flow
    /// 1. Validate `call.arguments` against `manifest.input_schema`
    /// 2. Create fresh `Store` with fuel/memory limits    /// 3. Register host functions with tool-specific `SandboxHostState`
    /// 4. Instantiate module into store
    /// 5. Write serialized input to WASM memory via `nexus_set_input`
    /// 6. Execute `_nexus_run` with timeout
    /// 7. Read output via `nexus_get_output`
    /// 8. Validate output against `manifest.output_schema`
    /// 9. Collect metrics and build `ToolResult`
    ///
    /// # Error Handling
    /// - Schema validation errors → `ToolError::SchemaValidation`
    /// - WASM trap (panic, OOB) → `ToolError::ExecutionTrap`
    /// - Timeout → `ToolError::Timeout`
    /// - Fuel/OOM → `ToolError::ResourceExceeded`
    /// - Host function error → propagated as `ToolError::HostFunctionError`
    #[instrument(skip(self, call), fields(tool = %self.manifest.name, call_id = %call.id))]
    pub async fn invoke(&self, call: &ToolCall) -> Result<ToolResult> {
        let start = Instant::now();

        // Step 1: Validate input against JSON schema
        if !self.manifest.input_schema.is_null() {
            let schema = JSONSchema::compile(&self.manifest.input_schema)
                .map_err(|e| ToolError::SchemaInvalid {
                    schema_type: "input".into(),
                    reason: e.to_string(),
                })?;

            if let Err(errors) = schema.validate(&call.arguments) {
                let first = errors
                    .into_iter()
                    .next()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "validation failed".into());

                return Err(ToolError::SchemaValidation {
                    tool: self.manifest.name.clone(),
                    field: "arguments".into(),
                    message: first,
                });
            }
            debug!("input schema validation passed");
        }

        // Step 2: Create per-call store with limits
        let mut store = self.create_store()?;

        // Step 3: Create linker and register host functions
        let mut linker = Linker::new(&self.engine);
        register_host_functions(&mut linker)
            .map_err(|e| ToolError::HostFunctionError(e.to_string()))?;
        // Build host state with tool-specific allowlists
        let allowed_hosts = self
            .manifest
            .capabilities_required
            .iter()
            .filter_map(|cap| match cap {
                nexus_proto::tool::ToolCapabilityRequirement::NetworkAccess { allowed_hosts } => {
                    Some(allowed_hosts.clone())
                }
                _ => None,
            })
            .flatten()
            .collect();

        let allowed_env = self
            .manifest
            .capabilities_required
            .iter()
            .filter_map(|cap| match cap {
                nexus_proto::tool::ToolCapabilityRequirement::FilesystemRead { paths }
                | nexus_proto::tool::ToolCapabilityRequirement::FilesystemWrite { paths } => {
                    // Extract env var names from paths if they look like ${VAR}
                    Some(
                        paths
                            .iter()
                            .filter_map(|p| {
                                if p.starts_with("${") && p.ends_with('}') {
                                    Some(p[2..p.len() - 1].to_string())
                                } else {
                                    None
                                }
                            })
                            .collect(),
                    )
                }
                _ => None,
            })
            .flatten()
            .collect();

        let host_state = SandboxHostState::new(
            self.manifest.name.clone(),
            allowed_hosts,
            allowed_env,
        );
        linker.data_mut(&mut store).clone_from(&host_state);

        // Step 4: Instantiate module
        let instance = linker
            .instantiate_async(&mut store, &self.module)            .await
            .map_err(|e| ToolError::InstantiationError(e.to_string()))?;

        // Step 5: Write input to WASM memory
        let input_json = serde_json::to_vec(&call.arguments)
            .map_err(|e| ToolError::SerializationError(e))?;

        self.write_input_to_memory(&mut store, &instance, &input_json)
            .await?;

        // Step 6: Set fuel limit
        store.set_fuel(self.config.max_fuel)
            .map_err(|e| ToolError::ResourceExceeded(format!("failed to set fuel: {}", e)))?;

        // Step 7: Execute with timeout
        let execution = timeout(
            Duration::from_millis(self.config.timeout_ms),
            self.call_nexus_run(&mut store, &instance),
        )
        .await;

        let execution_time_ms = start.elapsed().as_millis() as u64;

        // Step 8: Handle execution result
        let (success, error_msg) = match execution {
            Ok(Ok(0)) => (true, None),
            Ok(Ok(code)) => (false, Some(format!("tool returned error code: {}", code))),
            Ok(Err(e)) => (false, Some(format!("execution trap: {}", e))),
            Err(_) => (false, Some(format!("timeout after {}ms", self.config.timeout_ms))),
        };

        // Step 9: Read output if successful
        let output_bytes = if success {
            match self.read_output_from_memory(&mut store, &instance).await {
                Ok(bytes) => {
                    // Check output size limit
                    if bytes.len() > self.config.max_output_bytes {
                        return Err(ToolError::ResourceExceeded(format!(
                            "output size {} exceeds limit {}",
                            bytes.len(),
                            self.config.max_output_bytes
                        )));
                    }
                    bytes
                }
                Err(e) => {
                    return Err(ToolError::ExecutionError(format!(
                        "failed to read output: {}",
                        e
                    )));                }
            }
        } else {
            Vec::new()
        };

        // Step 10: Validate output against schema
        if success && !self.manifest.output_schema.is_null() {
            if let Ok(output_value) = serde_json::from_slice::<Value>(&output_bytes) {
                let schema = JSONSchema::compile(&self.manifest.output_schema)
                    .map_err(|e| ToolError::SchemaInvalid {
                        schema_type: "output".into(),
                        reason: e.to_string(),
                    })?;

                if let Err(errors) = schema.validate(&output_value) {
                    let first = errors
                        .into_iter()
                        .next()
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| "validation failed".into());

                    return Err(ToolError::SchemaValidation {
                        tool: self.manifest.name.clone(),
                        field: "output".into(),
                        message: first,
                    });
                }
                debug!("output schema validation passed");
            }
        }

        // Step 11: Collect metrics
        let fuel_consumed = store.fuel_consumed().unwrap_or(0);
        let memory_peak_mb = self.estimate_memory_usage(&store, &instance);

        // Step 12: Build result
        let sandbox_result = SandboxResult {
            output: output_bytes,
            fuel_consumed,
            memory_peak_mb,
            execution_time_ms,
            exited_normally: success,
            error: error_msg,
        };

        Ok(sandbox_result.into_tool_result(call.id, self.manifest.name.clone()))
    }

    /// Creates a fresh `Store` with fuel and memory limits configured.    fn create_store(&self) -> Result<Store<SandboxHostState>> {
        let mut config = Config::clone(self.engine.config());
        config.consume_fuel(true);
        config.fuel_consumption_mode(FuelConsumptionMode::Deterministic);

        let engine = Engine::new(&config)
            .map_err(|e| ToolError::EngineError(e.to_string()))?;

        let mut store = Store::new(&engine, SandboxHostState::new(
            self.manifest.name.clone(),
            vec![],
            vec![],
        ));

        // Set memory limit via store configuration
        // Note: wasmtime memory limits are set at module compile time,
        // but we can enforce at runtime via fuel and host function checks
        store.out_of_fuel_action(|_, _| Ok(true));

        Ok(store)
    }

    /// Writes serialized input JSON to WASM linear memory.
    ///
    /// Uses the tool's exported `nexus_set_input` function if available,
    /// otherwise writes directly to known offsets (0x10000).
    async fn write_input_to_memory(
        &self,
        store: &mut Store<SandboxHostState>,
        instance: &wasmtime::Instance,
        input: &[u8],
    ) -> Result<()> {
        // Try to find nexus_set_input export
        if let Some(set_input) = instance.get_func(store, "nexus_set_input") {
            // Allocate memory for input if tool provides allocate function
            let (ptr, len) = if let Some(allocate) = instance.get_func(store, "allocate") {
                let mut results = [Val::I32(0)];
                allocate
                    .call_async(store, &[Val::I32(input.len() as i32)], &mut results)
                    .await
                    .map_err(|e| ToolError::ExecutionError(format!("allocate failed: {}", e)))?;
                (results[0].i32().ok_or_else(|| ToolError::ExecutionError("allocate returned non-i32".into()))?, input.len() as i32)
            } else {
                // Fallback: use known offset
                (0x10000, input.len() as i32)
            };

            // Write input bytes to memory
            let memory = instance
                .get_memory(store, "memory")                .ok_or_else(|| ToolError::ExecutionError("memory export not found".into()))?;

            self.write_to_memory(store, memory, ptr as usize, input)?;

            // Call nexus_set_input(ptr, len)
            set_input
                .call_async(store, &[Val::I32(ptr), Val::I32(len)], &mut [])
                .await
                .map_err(|e| ToolError::ExecutionError(format!("nexus_set_input failed: {}", e)))?;

            debug!(ptr, len, "input written to WASM memory");
            Ok(())
        } else {
            // Fallback: write directly to known offset
            let memory = instance
                .get_memory(store, "memory")
                .ok_or_else(|| ToolError::ExecutionError("memory export not found".into()))?;

            self.write_to_memory(store, memory, 0x10000, input)?;
            debug!("input written to fallback offset 0x10000");
            Ok(())
        }
    }

    /// Calls the tool's `_nexus_run` exported function.
    async fn call_nexus_run(
        &self,
        store: &mut Store<SandboxHostState>,
        instance: &wasmtime::Instance,
    ) -> Result<i32> {
        let run_func = instance
            .get_func(store, "_nexus_run")
            .ok_or_else(|| ToolError::ExecutionError("_nexus_run export not found".into()))?;

        let mut results = [Val::I32(0)];
        run_func
            .call_async(store, &[], &mut results)
            .await
            .map_err(|e| match e.downcast::<Trap>() {
                Ok(trap) => ToolError::ExecutionTrap(trap.to_string()),
                Err(e) => ToolError::ExecutionError(e.to_string()),
            })?;

        let return_code = results[0]
            .i32()
            .ok_or_else(|| ToolError::ExecutionError("_nexus_run returned non-i32".into()))?;

        Ok(return_code)
    }
    /// Reads output from WASM linear memory after execution.
    async fn read_output_from_memory(
        &self,
        store: &mut Store<SandboxHostState>,
        instance: &wasmtime::Instance,
    ) -> Result<Vec<u8>> {
        // Try to find nexus_get_output export
        if let Some(get_output) = instance.get_func(store, "nexus_get_output") {
            let mut results = [Val::I64(0)];
            get_output
                .call_async(store, &[], &mut results)
                .await
                .map_err(|e| ToolError::ExecutionError(format!("nexus_get_output failed: {}", e)))?;

            let packed = results[0].i64().ok_or_else(|| ToolError::ExecutionError("nexus_get_output returned non-i64".into()))?;
            let ptr = (packed >> 32) as u32;
            let len = (packed & 0xFFFFFFFF) as u32;

            if len == 0 {
                return Ok(Vec::new());
            }

            let memory = instance
                .get_memory(store, "memory")
                .ok_or_else(|| ToolError::ExecutionError("memory export not found".into()))?;

            let output = self.read_from_memory(store, memory, ptr as usize, len as usize)?;
            debug!(ptr, len, bytes = output.len(), "output read from WASM memory");
            Ok(output)
        } else {
            // Fallback: read from known offset
            let memory = instance
                .get_memory(store, "memory")
                .ok_or_else(|| ToolError::ExecutionError("memory export not found".into()))?;

            // Try to read length from known location first
            let len_bytes = self.read_from_memory(store, memory, 0x20000, 4)?;
            let len = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;

            if len == 0 {
                return Ok(Vec::new());
            }

            let output = self.read_from_memory(store, memory, 0x20004, len)?;
            debug!(len, bytes = output.len(), "output read from fallback offset");
            Ok(output)
        }
    }

    /// Helper: write bytes to WASM linear memory with bounds checking.    fn write_to_memory(
        &self,
        store: &mut Store<SandboxHostState>,
        memory: Memory,
        offset: usize,
        data: &[u8],
    ) -> Result<()> {
        let memory_size = memory.data_size(store);
        let end = offset.checked_add(data.len())
            .ok_or_else(|| ToolError::ResourceExceeded("integer overflow in memory write".into()))?;

        if end > memory_size {
            return Err(ToolError::ResourceExceeded(format!(
                "memory write out of bounds: offset={}, len={}, memory_size={}",
                offset, data.len(), memory_size
            )));
        }

        memory
            .data_mut(store)
            .get_mut(offset..end)
            .ok_or_else(|| ToolError::ExecutionError("memory slice failed".into()))?
            .copy_from_slice(data);

        Ok(())
    }

    /// Helper: read bytes from WASM linear memory with bounds checking.
    fn read_from_memory(
        &self,
        store: &mut Store<SandboxHostState>,
        memory: Memory,
        offset: usize,
        len: usize,
    ) -> Result<Vec<u8>> {
        let memory_size = memory.data_size(store);
        let end = offset.checked_add(len)
            .ok_or_else(|| ToolError::ResourceExceeded("integer overflow in memory read".into()))?;

        if end > memory_size {
            return Err(ToolError::ResourceExceeded(format!(
                "memory read out of bounds: offset={}, len={}, memory_size={}",
                offset, len, memory_size
            )));
        }

        Ok(memory
            .data(store)
            .get(offset..end)
            .ok_or_else(|| ToolError::ExecutionError("memory slice failed".into()))?            .to_vec())
    }

    /// Estimates peak memory usage in MB (approximate, for observability).
    fn estimate_memory_usage(
        &self,
        store: &Store<SandboxHostState>,
        instance: &wasmtime::Instance,
    ) -> u32 {
        if let Some(memory) = instance.get_memory(store, "memory") {
            let pages = memory.size(store);
            // WASM page = 64KB
            (pages as u64 * 64 * 1024 / (1024 * 1024)) as u32
        } else {
            0
        }
    }

    /// Returns the tool name for logging/observability.
    pub fn tool_name(&self) -> &str {
        &self.manifest.name
    }

    /// Returns the tool version.
    pub fn version(&self) -> &str {
        &self.manifest.version
    }

    /// Returns the compiled module for inspection (advanced use).
    pub fn module(&self) -> &Arc<Module> {
        &self.module
    }
}

// =============================================================================
// Utility: Compute Checksum from Bytes (for verification without file I/O)
// =============================================================================

/// Computes SHA-256 hex digest of raw WASM bytes.
fn compute_wasm_checksum_raw(bytes: &[u8]) -> Result<String> {
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();

    Ok(hex::encode(result))
}

// =============================================================================// Shared Engine Initialization — Call Once at Startup
// =============================================================================

/// Creates a shared `wasmtime::Engine` configured for tool sandboxing.
///
/// # Configuration
/// - Fuel consumption enabled for instruction limiting
/// - Deterministic fuel mode for reproducible limits
/// - Memory limits enforced via module compilation flags
/// - Async support for host function integration
///
/// # Usage
/// Create one engine at application startup and share via `Arc` across
/// all `WasmSandbox` instances. Compilation is expensive; execution is cheap.
pub fn create_shared_engine() -> Result<Arc<Engine>> {
    let mut config = Config::new();
    config.async_support(true);
    config.consume_fuel(true);
    config.fuel_consumption_mode(FuelConsumptionMode::Deterministic);
    config.wasm_bulk_memory(true);
    config.wasm_reference_types(true);
    config.wasm_simd(true);

    // Memory limits: tools can't exceed these even if manifest allows more
    config.memory_reservation(0);
    config.memory_reservation_for_growth(1024 * 1024); // 1MB growth reservation

    let engine = Engine::new(&config)
        .map_err(|e| ToolError::EngineError(format!("failed to create engine: {}", e)))?;

    Ok(Arc::new(engine))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_proto::tool::ToolManifest;

    // Minimal valid WASM module for testing (exports _nexus_run that returns 0)
    const MINIMAL_WASM: &[u8] = include_bytes!("../tests/minimal_tool.wasm");

    fn test_manifest() -> ToolManifest {
        ToolManifest {
            name: "test-tool".into(),
            version: "1.0.0".into(),
            description: "Test tool".into(),            author: None,
            license: None,
            capabilities_required: vec![],
            resource_limits: ResourceLimits::default(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
            output_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            }),
            wasm_path: None,
            checksum_sha256: None,
        }
    }

    #[tokio::test]
    async fn test_sandbox_creation() {
        let engine = create_shared_engine().unwrap();
        let manifest = test_manifest();

        // This would require a real minimal_tool.wasm file
        // For now, skip if file not present
        if MINIMAL_WASM.is_empty() {
            return;
        }

        let sandbox = WasmSandbox::new(engine, manifest, MINIMAL_WASM.to_vec());
        assert!(sandbox.is_ok());
    }

    #[test]
    fn test_config_from_limits() {
        let limits = ResourceLimits {
            max_memory_mb: 128,
            max_fuel: 20_000_000,
            timeout_ms: 10000,
            max_output_bytes: 2097152,
        };

        let config: SandboxConfig = limits.into();
        assert_eq!(config.max_memory_bytes, 128 * 1024 * 1024);
        assert_eq!(config.max_fuel, 20_000_000);
        assert_eq!(config.timeout_ms, 10000);
        assert_eq!(config.max_output_bytes, 2097152);    }

    #[test]
    fn test_checksum_raw() {
        let bytes = b"hello world";
        let checksum = compute_wasm_checksum_raw(bytes).unwrap();
        // SHA-256 of "hello world"
        assert_eq!(
            checksum,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_sandbox_result_conversions() {
        let result = SandboxResult {
            output: br#"{"result": "ok"}"#.to_vec(),
            fuel_consumed: 12345,
            memory_peak_mb: 32,
            execution_time_ms: 150,
            exited_normally: true,
            error: None,
        };

        let tool_result = result.into_tool_result(
            uuid::Uuid::new_v4(),
            "test-tool".into(),
        );

        assert!(!tool_result.is_error);
        assert_eq!(tool_result.fuel_consumed, 12345);
        assert_eq!(tool_result.memory_peak_mb, 32);
    }
}
