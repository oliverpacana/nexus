// tools/web-search/src/lib.rs
// WASM tool for web search using DuckDuckGo Instant Answer API
// Compile with: cargo build --target wasm32-unknown-unknown --release

#![no_std]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::c_char;
use core::ptr;

// =============================================================================
// Host Function Declarations
// =============================================================================

extern "C" {
    /// HTTP GET request. Returns number of bytes written to out_buf, or negative error code.
    /// -1: host not allowed, -2: request failed, -3: output buffer too small
    fn nexus_http_get(
        url_ptr: *const c_char,
        url_len: i32,
        out_ptr: *mut c_char,
        out_max: i32,
    ) -> i32;

    /// Log a message at the given level (0=trace, 1=debug, 2=info, 3=warn, 4=error)
    fn nexus_log(level: i32, msg_ptr: *const c_char, msg_len: i32);
}

// =============================================================================
// Static Buffers for Input/Output
// =============================================================================

static mut INPUT_BUFFER: [u8; 65536] = [0u8; 65536];
static mut OUTPUT_BUFFER: [u8; 262144] = [0u8; 262144];
static mut INPUT_LEN: u32 = 0;
static mut OUTPUT_LEN: u32 = 0;

// =============================================================================
// Exported ABI Functions
// =============================================================================

#[no_mangle]
pub extern "C" fn nexus_get_input_ptr() -> *mut u8 {
    unsafe { INPUT_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn nexus_get_input_len() -> u32 {    unsafe { INPUT_LEN }
}

#[no_mangle]
pub unsafe extern "C" fn nexus_set_input_len(len: u32) {
    INPUT_LEN = len;
}

#[no_mangle]
pub extern "C" fn nexus_get_output_ptr() -> *mut u8 {
    unsafe { OUTPUT_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn nexus_get_output_len() -> u32 {
    unsafe { OUTPUT_LEN }
}

#[no_mangle]
pub unsafe extern "C" fn nexus_set_output_len(len: u32) {
    OUTPUT_LEN = len;
}

// =============================================================================
// Minimal JSON Parser (for no_std environment)
// =============================================================================

/// Very basic JSON string extractor. Looks for "key": "value" patterns.
/// Returns the value as a String, or None if not found.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let search_key = format!("\"{}\"", key);
    if let Some(key_pos) = json.find(&search_key) {
        if let Some(colon_pos) = json[key_pos..].find(':') {
            let start = key_pos + colon_pos + 1;
            let rest = &json[start..].trim_start();
            if rest.starts_with('"') {
                if let Some(end_quote) = rest[1..].find('"') {
                    return Some(rest[1..end_quote + 1].to_string());
                }
            }
        }
    }
    None
}

/// Extracts an integer value for a given key from JSON.
fn extract_json_int(json: &str, key: &str) -> Option<i32> {
    let search_key = format!("\"{}\"", key);
    if let Some(key_pos) = json.find(&search_key) {
        if let Some(colon_pos) = json[key_pos..].find(':') {            let start = key_pos + colon_pos + 1;
            let rest = &json[start..].trim_start();
            // Parse digits (including negative)
            let mut end = 0;
            for (i, c) in rest.chars().enumerate() {
                if c.is_ascii_digit() || (i == 0 && c == '-') {
                    end = i + 1;
                } else {
                    break;
                }
            }
            if end > 0 {
                return rest[..end].parse().ok();
            }
        }
    }
    None
}

/// Extracts a JSON array of objects for a given key. Very limited: only handles flat objects with string fields.
fn extract_json_array_objects(json: &str, key: &str) -> Vec<alloc::collections::BTreeMap<String, String>> {
    let mut results = Vec::new();
    let search_key = format!("\"{}\"", key);
    if let Some(key_pos) = json.find(&search_key) {
        if let Some(array_start) = json[key_pos..].find('[') {
            let mut depth = 0;
            let mut obj_start = None;
            let mut chars = json[key_pos + array_start..].char_indices().peekable();
            
            while let Some((idx, ch)) = chars.next() {
                match ch {
                    '[' | '{' => depth += 1,
                    ']' | '}' => {
                        depth -= 1;
                        if depth == 0 && ch == ']' {
                            break;
                        }
                    }
                    '{' if depth == 1 => {
                        obj_start = Some(idx);
                    }
                    '}' if depth == 1 && obj_start.is_some() => {
                        if let Some(start) = obj_start {
                            let obj_str = &json[key_pos + array_start + start..key_pos + array_start + idx + 1];
                            if let Some(obj) = parse_simple_json_object(obj_str) {
                                results.push(obj);
                            }
                        }
                        obj_start = None;
                    }                    _ => {}
                }
            }
        }
    }
    results
}

/// Parses a simple JSON object with string values only.
fn parse_simple_json_object(s: &str) -> Option<alloc::collections::BTreeMap<String, String>> {
    let mut map = alloc::collections::BTreeMap::new();
    let mut chars = s.chars().peekable();
    
    // Skip opening brace
    if chars.next() != Some('{') {
        return None;
    }
    
    loop {
        // Skip whitespace
        while let Some(&ch) = chars.peek() {
            if ch.is_whitespace() {
                chars.next();
            } else {
                break;
            }
        }
        
        // Check for closing brace or comma
        match chars.peek() {
            Some('}') => {
                chars.next();
                break;
            }
            Some(',') => {
                chars.next();
                continue;
            }
            _ => {}
        }
        
        // Parse key
        if chars.next() != Some('"') {
            return None;
        }
        let mut key = String::new();
        while let Some(&ch) = chars.peek() {
            if ch == '"' {
                chars.next();
                break;            }
            key.push(chars.next().unwrap());
        }
        
        // Skip colon and whitespace
        while let Some(&ch) = chars.peek() {
            if ch == ':' || ch.is_whitespace() {
                chars.next();
            } else {
                break;
            }
        }
        
        // Parse value (string only for now)
        if chars.next() != Some('"') {
            return None;
        }
        let mut value = String::new();
        while let Some(&ch) = chars.peek() {
            if ch == '"' {
                chars.next();
                break;
            }
            value.push(chars.next().unwrap());
        }
        
        map.insert(key, value);
    }
    
    Some(map)
}

// =============================================================================
// Main Tool Logic
// =============================================================================

/// Main entry point called by the Nexus runtime.
#[no_mangle]
pub extern "C" fn _nexus_run() -> i32 {
    // 1. Read and parse input
    let input_json = unsafe {
        let ptr = nexus_get_input_ptr();
        let len = nexus_get_input_len() as usize;
        if len == 0 || len > INPUT_BUFFER.len() {
            write_error("Invalid input length");
            return 1;
        }
        core::str::from_utf8(&INPUT_BUFFER[..len]).unwrap_or("")
    };
    let query = match extract_json_string(input_json, "query") {
        Some(q) if !q.is_empty() => q,
        _ => {
            write_error("Missing or empty 'query' field in input");
            return 1;
        }
    };

    let max_results = extract_json_int(input_json, "max_results").unwrap_or(10).max(1).min(30);

    log_info(&format!("Searching for: {} (max {} results)", query, max_results));

    // 2. Construct DuckDuckGo API URL
    let encoded_query = url_encode(&query);
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&no_redirect=1",
        encoded_query
    );

    // 3. Make HTTP GET request
    let mut response_buffer = [0u8; 65536];
    let response_len = unsafe {
        nexus_http_get(
            url.as_ptr() as *const c_char,
            url.len() as i32,
            response_buffer.as_mut_ptr() as *mut c_char,
            response_buffer.len() as i32,
        )
    };

    if response_len < 0 {
        let error_msg = match response_len {
            -1 => "Host not allowed by manifest",
            -2 => "HTTP request failed",
            -3 => "Response buffer too small",
            _ => "Unknown HTTP error",
        };
        write_error(&format!("HTTP error: {}", error_msg));
        return 1;
    }

    let response_json = unsafe {
        core::str::from_utf8(&response_buffer[..response_len as usize]).unwrap_or("")
    };

    log_debug(&format!("Received {} bytes from DuckDuckGo", response_len));

    // 4. Parse DuckDuckGo response and extract results
    let mut results = Vec::new();
    // Try to extract Abstract (main result)
    if let Some(abstract_text) = extract_json_string(response_json, "Abstract") {
        if !abstract_text.is_empty() {
            let mut result = alloc::collections::BTreeMap::new();
            result.insert("title".to_string(), extract_json_string(response_json, "Heading").unwrap_or_else(|| query.clone()));
            result.insert("url".to_string(), extract_json_string(response_json, "AbstractURL").unwrap_or_default());
            result.insert("snippet".to_string(), abstract_text);
            results.push(result);
        }
    }

    // Extract RelatedTopics array
    let topics = extract_json_array_objects(response_json, "RelatedTopics");
    for topic in topics {
        if results.len() >= max_results as usize {
            break;
        }
        
        // DuckDuckGo format: each topic has "Text" and optionally "FirstURL"
        let text = topic.get("Text").cloned().unwrap_or_default();
        let url = topic.get("FirstURL").cloned().unwrap_or_default();
        
        // Parse "Text" field which is usually "Title - Snippet"
        let (title, snippet) = if let Some(dash_pos) = text.find(" - ") {
            (text[..dash_pos].to_string(), text[dash_pos + 3..].to_string())
        } else {
            (text.clone(), String::new())
        };

        if !title.is_empty() {
            let mut result = alloc::collections::BTreeMap::new();
            result.insert("title".to_string(), title);
            result.insert("url".to_string(), url);
            result.insert("snippet".to_string(), snippet);
            results.push(result);
        }
    }

    // 5. Build output JSON
    let mut output = String::from("{\"results\":[");
    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            output.push(',');
        }
        output.push('{');
        output.push_str(&format!("\"title\":\"{}\",", escape_json_string(result.get("title").unwrap_or(&String::new()))));
        output.push_str(&format!("\"url\":\"{}\",", escape_json_string(result.get("url").unwrap_or(&String::new()))));
        output.push_str(&format!("\"snippet\":\"{}\"", escape_json_string(result.get("snippet").unwrap_or(&String::new()))));
        output.push('}');
    }    output.push_str("]}");

    // 6. Write output to buffer
    let output_bytes = output.as_bytes();
    if output_bytes.len() > OUTPUT_BUFFER.len() {
        write_error("Output too large for buffer");
        return 1;
    }
    unsafe {
        ptr::copy_nonoverlapping(output_bytes.as_ptr(), OUTPUT_BUFFER.as_mut_ptr(), output_bytes.len());
        nexus_set_output_len(output_bytes.len() as u32);
    }

    log_info(&format!("Successfully returned {} results", results.len()));
    0 // Success
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Writes an error JSON to the output buffer and sets the length.
fn write_error(msg: &str) {
    let error_json = format!("{{\"error\":\"{}\"}}", escape_json_string(msg));
    let bytes = error_json.as_bytes();
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), OUTPUT_BUFFER.as_mut_ptr(), bytes.len());
        nexus_set_output_len(bytes.len() as u32);
    }
}

/// Logs an info message via the host nexus_log function.
fn log_info(msg: &str) {
    unsafe {
        nexus_log(2, msg.as_ptr() as *const c_char, msg.len() as i32);
    }
}

/// Logs a debug message.
fn log_debug(msg: &str) {
    unsafe {
        nexus_log(1, msg.as_ptr() as *const c_char, msg.len() as i32);
    }
}

/// URL-encodes a string (very basic, handles spaces and common chars).
fn url_encode(s: &str) -> String {
    let mut encoded = String::new();
    for ch in s.chars() {
        match ch {            ' ' => encoded.push_str("%20"),
            '!' => encoded.push_str("%21"),
            '#' => encoded.push_str("%23"),
            '$' => encoded.push_str("%24"),
            '&' => encoded.push_str("%26"),
            '\'' => encoded.push_str("%27"),
            '(' => encoded.push_str("%28"),
            ')' => encoded.push_str("%29"),
            '*' => encoded.push_str("%2A"),
            '+' => encoded.push_str("%2B"),
            ',' => encoded.push_str("%2C"),
            '/' => encoded.push_str("%2F"),
            ':' => encoded.push_str("%3A"),
            ';' => encoded.push_str("%3B"),
            '=' => encoded.push_str("%3D"),
            '?' => encoded.push_str("%3F"),
            '@' => encoded.push_str("%40"),
            '[' => encoded.push_str("%5B"),
            ']' => encoded.push_str("%5D"),
            _ if ch.is_ascii_alphanumeric() => encoded.push(ch),
            _ => encoded.push_str("%"),
        }
    }
    encoded
}

/// Escapes a string for safe inclusion in JSON.
fn escape_json_string(s: &str) -> alloc::string::String {
    let mut escaped = String::new();
    for ch in s.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                // Write as \uXXXX
                let code = ch as u32;
                escaped.push_str(&format!("\\u{:04x}", code));
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}
