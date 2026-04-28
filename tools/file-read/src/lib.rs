// tools/file-read/src/lib.rs
// WASM tool for reading and parsing files in a sandboxed environment
// Compile with: cargo build --target wasm32-unknown-unknown --release

#![no_std]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use alloc::boxed::Box;
use core::fmt;
use core::ffi::c_char;
use core::ptr;
use core::str;

// =============================================================================
// Host Function Declarations
// =============================================================================

extern "C" {
    fn nexus_http_get(
        url_ptr: *const c_char,
        url_len: i32,
        out_ptr: *mut c_char,
        out_max: i32,
    ) -> i32;
    fn nexus_log(level: i32, msg_ptr: *const c_char, msg_len: i32);
}

// =============================================================================
// Static Buffers
// =============================================================================

static mut INPUT_BUFFER: [u8; 65536] = [0u8; 65536];
static mut OUTPUT_BUFFER: [u8; 2097152] = [0u8; 2097152]; // 2MB for output
static mut FILE_BUFFER: [u8; 1048576] = [0u8; 1048576]; // 1MB for file content
static mut INPUT_LEN: u32 = 0;
static mut OUTPUT_LEN: u32 = 0;

// =============================================================================
// ABI Exports
// =============================================================================

#[no_mangle]
pub extern "C" fn nexus_get_input_ptr() -> *mut u8 {
    unsafe { INPUT_BUFFER.as_mut_ptr() }
}

#[no_mangle]pub extern "C" fn nexus_get_input_len() -> u32 {
    unsafe { INPUT_LEN }
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
// Input Parameters
// =============================================================================

struct InputParams {
    path: String,
    format: Option<String>,
    encoding: String,
    max_bytes: usize,
    start_line: Option<usize>,
    end_line: Option<usize>,
    csv_delimiter: char,
    csv_has_header: bool,
    json_path: Option<String>,
    pattern: Option<String>,
    include_metadata: bool,
}

// =============================================================================
// Path Safety Validation
// =============================================================================

fn is_safe_path(path: &str) -> bool {
    // Block paths longer than 4096 chars
    if path.len() > 4096 {
        log_warn("Path too long");        return false;
    }
    
    // Block null bytes
    if path.contains('\0') {
        log_warn("Path contains null byte");
        return false;
    }
    
    // Block directory traversal
    if path.contains("..") {
        log_warn("Path contains directory traversal");
        return false;
    }
    
    // Block home directory expansion
    if path.starts_with('~') {
        log_warn("Path starts with ~");
        return false;
    }
    
    // Block shell metacharacters
    if path.contains(|c| matches!(c, ';' | '&' | '|' | '$' | '`' | '>' | '<' | '!' | '*')) {
        log_warn("Path contains shell metacharacters");
        return false;
    }
    
    // Block sensitive system paths (case-insensitive check)
    let lower_path = path.to_lowercase();
    let sensitive_paths = [
        "/etc/passwd", "/etc/shadow", "/proc/", "/sys/", "/dev/", 
        "/boot/", "/root/", "/private/", "c:\\windows\\system32"
    ];
    for sensitive in sensitive_paths.iter() {
        if lower_path.contains(sensitive) {
            log_warn(&format!("Path accesses sensitive location: {}", sensitive));
            return false;
        }
    }
    
    true
}

// =============================================================================
// File Reading via Host Function
// =============================================================================

fn read_file(path: &str, max_bytes: usize) -> Result<Vec<u8>, &'static str> {
    // Construct file:// URL
    let url = format!("file://{}", path);    let url_cstr = alloc::ffi::CString::new(url.as_str()).map_err(|_| "URL conversion failed")?;
    
    let response_ptr = unsafe { FILE_BUFFER.as_mut_ptr() as *mut c_char };
    let response_max = FILE_BUFFER.len() as i32;
    
    let response_len = unsafe {
        nexus_http_get(
            url_cstr.as_ptr(),
            url.len() as i32,
            response_ptr,
            response_max.min(max_bytes as i32),
        )
    };
    
    if response_len < 0 {
        return match response_len {
            -1 => Err("Host not allowed by manifest"),
            -2 => Err("File read failed"),
            -3 => Err("File too large for buffer"),
            _ => Err("Unknown file read error"),
        };
    }
    
    Ok(unsafe { FILE_BUFFER[..response_len as usize].to_vec() })
}

// =============================================================================
// Format Detection
// =============================================================================

fn detect_format(path: &str, bytes: &[u8]) -> &'static str {
    // Check by extension first
    if let Some(ext) = path.split('.').last() {
        match ext.to_lowercase().as_str() {
            "json" => return "json",
            "toml" => return "toml",
            "csv" => return "csv",
            "tsv" => return "csv", // TSV is CSV with tab delimiter
            "yaml" | "yml" => return "yaml",
            "txt" | "md" | "rs" | "py" | "js" | "html" | "css" | "xml" => return "text",
            _ => {}
        }
    }
    
    // Sniff content if extension unknown
    if let Ok(text) = str::from_utf8(bytes) {
        let trimmed = text.trim_start();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return "json";
        }        if text.contains('=') && !text.contains('{') && !text.contains('[') {
            return "toml";
        }
        // Simple CSV sniff: check first 3 lines for consistent comma count
        let lines: Vec<&str> = text.lines().take(3).collect();
        if lines.len() >= 2 {
            let first_commas = lines[0].matches(',').count();
            if first_commas > 0 && lines[1..].iter().all(|line| line.matches(',').count() == first_commas) {
                return "csv";
            }
        }
    }
    
    // Default to text if printable, otherwise binary
    if bytes.iter().all(|&b| b >= 32 || b == 9 || b == 10 || b == 13) {
        "text"
    } else {
        "binary"
    }
}

// =============================================================================
// Text Processing
// =============================================================================

struct ProcessedText {
    content: String,
    line_count: Option<usize>,
    truncated: bool,
}

fn process_text(bytes: &[u8], params: &InputParams) -> ProcessedText {
    // Decode as UTF-8, replacing invalid sequences
    let text = String::from_utf8_lossy(bytes).into_owned();
    
    // Split into lines (handle different line endings)
    let mut lines: Vec<&str> = text
        .split(|c| c == '\n' || c == '\r')
        .filter(|line| !line.is_empty() || params.end_line.is_none())
        .collect();
    
    let total_lines = lines.len();
    
    // Apply line range filtering (1-indexed, inclusive)
    if let Some(start) = params.start_line {
        if start > 0 && start <= lines.len() {
            lines = lines[start - 1..].to_vec();
        } else {
            lines.clear();
        }    }
    if let Some(end) = params.end_line {
        if end > 0 && end <= lines.len() {
            lines = lines[..end].to_vec();
        } else {
            lines.clear();
        }
    }
    
    // Apply pattern filter
    if let Some(ref pattern) = params.pattern {
        lines.retain(|line| line.contains(pattern));
    }
    
    // Join lines back together
    let content = lines.join("\n");
    
    // Check truncation
    let truncated = content.len() > params.max_bytes;
    let final_content = if truncated {
        content[..params.max_bytes].to_string()
    } else {
        content
    };
    
    ProcessedText {
        content: final_content,
        line_count: if params.start_line.is_none() && params.end_line.is_none() {
            Some(total_lines)
        } else {
            None
        },
        truncated,
    }
}

// =============================================================================
// CSV Parser (RFC 4180 Compliant)
// =============================================================================

fn parse_csv(text: &str, delimiter: char, has_header: bool) -> alloc::collections::BTreeMap<String, alloc::string::String> {
    let mut result = alloc::collections::BTreeMap::new();
    
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut current_row = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();
    
    while let Some(ch) = chars.next() {        if in_quotes {
            if ch == '"' {
                if let Some(&next_ch) = chars.peek() {
                    if next_ch == '"' {
                        // Escaped quote
                        current_field.push('"');
                        chars.next();
                    } else {
                        // End of quoted field
                        in_quotes = false;
                    }
                } else {
                    in_quotes = false;
                }
            } else {
                current_field.push(ch);
            }
        } else {
            match ch {
                '"' => in_quotes = true,
                c if c == delimiter => {
                    current_row.push(current_field.clone());
                    current_field.clear();
                }
                '\n' | '\r' => {
                    if !current_field.is_empty() || !current_row.is_empty() {
                        current_row.push(current_field.clone());
                        current_field.clear();
                        rows.push(current_row.clone());
                        current_row.clear();
                    }
                    // Handle \r\n
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                }
                _ => current_field.push(ch),
            }
        }
    }
    
    // Handle last field/row
    if !current_field.is_empty() || !current_row.is_empty() {
        current_row.push(current_field);
        rows.push(current_row);
    }
    
    // Limit to 10000 rows
    if rows.len() > 10000 {
        rows.truncate(10000);        result.insert("truncated".to_string(), "true".to_string());
    }
    
    // Convert to JSON-like structure
    if has_header && !rows.is_empty() {
        let headers = &rows[0];
        let mut json_array = String::from("[");
        for (i, row) in rows[1..].iter().enumerate() {
            if i > 0 { json_array.push(','); }
            json_array.push('{');
            for (j, header) in headers.iter().enumerate() {
                if j > 0 { json_array.push(','); }
                let value = row.get(j).map(|s| s.as_str()).unwrap_or("");
                json_array.push_str(&format!("\"{}\":\"{}\"", 
                    escape_json_string(header), 
                    escape_json_string(value)));
            }
            json_array.push('}');
        }
        json_array.push(']');
        result.insert("data".to_string(), json_array);
    } else {
        let mut json_array = String::from("[");
        for (i, row) in rows.iter().enumerate() {
            if i > 0 { json_array.push(','); }
            json_array.push('[');
            for (j, field) in row.iter().enumerate() {
                if j > 0 { json_array.push(','); }
                json_array.push('"');
                json_array.push_str(&escape_json_string(field));
                json_array.push('"');
            }
            json_array.push(']');
        }
        json_array.push(']');
        result.insert("data".to_string(), json_array);
    }
    
    result
}

// =============================================================================
// JSON Parser and JSONPath Evaluator
// =============================================================================

#[derive(Clone, Debug)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),    String(String),
    Array(Vec<JsonValue>),
    Object(alloc::collections::BTreeMap<String, JsonValue>),
}

fn parse_json(text: &str) -> Result<JsonValue, String> {
    let mut chars = text.trim().chars().peekable();
    parse_json_value(&mut chars)
}

fn parse_json_value(chars: &mut core::iter::Peekable<core::str::Chars>) -> Result<JsonValue, String> {
    skip_whitespace(chars);
    
    match chars.peek() {
        Some('n') => {
            // null
            consume_literal(chars, "null")?;
            Ok(JsonValue::Null)
        }
        Some('t') => {
            // true
            consume_literal(chars, "true")?;
            Ok(JsonValue::Bool(true))
        }
        Some('f') => {
            // false
            consume_literal(chars, "false")?;
            Ok(JsonValue::Bool(false))
        }
        Some('"') => {
            // string
            chars.next();
            let mut s = String::new();
            loop {
                match chars.next() {
                    Some('"') => break,
                    Some('\\') => {
                        match chars.next() {
                            Some('"') => s.push('"'),
                            Some('\\') => s.push('\\'),
                            Some('/') => s.push('/'),
                            Some('b') => s.push('\x08'),
                            Some('f') => s.push('\x0C'),
                            Some('n') => s.push('\n'),
                            Some('r') => s.push('\r'),
                            Some('t') => s.push('\t'),
                            Some('u') => {
                                // Unicode escape \uXXXX
                                let mut code = String::new();
                                for _ in 0..4 {                                    if let Some(c) = chars.next() {
                                        code.push(c);
                                    }
                                }
                                if let Ok(code_point) = u32::from_str_radix(&code, 16) {
                                    if let Some(c) = char::from_u32(code_point) {
                                        s.push(c);
                                    }
                                }
                            }
                            _ => return Err("Invalid escape sequence".into()),
                        }
                    }
                    Some(c) => s.push(c),
                    None => return Err("Unterminated string".into()),
                }
            }
            Ok(JsonValue::String(s))
        }
        Some('-') | Some('0'..='9') => {
            // number
            let mut num_str = String::new();
            if let Some('-') = chars.peek() {
                num_str.push(chars.next().unwrap());
            }
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    num_str.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            if let Some(&'.') = chars.peek() {
                num_str.push(chars.next().unwrap());
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() {
                        num_str.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
            }
            if let Some(&'e') | Some(&'E') = chars.peek() {
                num_str.push(chars.next().unwrap());
                if let Some(&'+' | &'-') = chars.peek() {
                    num_str.push(chars.next().unwrap());
                }
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() {
                        num_str.push(chars.next().unwrap());                    } else {
                        break;
                    }
                }
            }
            num_str.parse::<f64>()
                .map(JsonValue::Number)
                .map_err(|_| "Invalid number".into())
        }
        Some('[') => {
            // array
            chars.next();
            let mut arr = Vec::new();
            skip_whitespace(chars);
            if chars.peek() != Some(&']') {
                loop {
                    arr.push(parse_json_value(chars)?);
                    skip_whitespace(chars);
                    match chars.peek() {
                        Some(',') => { chars.next(); skip_whitespace(chars); }
                        Some(']') => break,
                        _ => return Err("Expected , or ] in array".into()),
                    }
                }
            }
            chars.next(); // consume ']'
            Ok(JsonValue::Array(arr))
        }
        Some('{') => {
            // object
            chars.next();
            let mut obj = alloc::collections::BTreeMap::new();
            skip_whitespace(chars);
            if chars.peek() != Some(&'}') {
                loop {
                    skip_whitespace(chars);
                    if let Some('"') = chars.peek() {
                        chars.next();
                        let mut key = String::new();
                        loop {
                            match chars.next() {
                                Some('"') => break,
                                Some(c) => key.push(c),
                                None => return Err("Unterminated string key".into()),
                            }
                        }
                        skip_whitespace(chars);
                        if chars.next() != Some(':') {
                            return Err("Expected : after key".into());
                        }                        skip_whitespace(chars);
                        let value = parse_json_value(chars)?;
                        obj.insert(key, value);
                        skip_whitespace(chars);
                        match chars.peek() {
                            Some(',') => { chars.next(); skip_whitespace(chars); }
                            Some('}') => break,
                            _ => return Err("Expected , or } in object".into()),
                        }
                    } else {
                        return Err("Expected string key in object".into());
                    }
                }
            }
            chars.next(); // consume '}'
            Ok(JsonValue::Object(obj))
        }
        _ => Err("Unexpected token in JSON".into()),
    }
}

fn consume_literal(chars: &mut core::iter::Peekable<core::str::Chars>, literal: &str) -> Result<(), String> {
    for expected in literal.chars() {
        if chars.next() != Some(expected) {
            return Err(format!("Expected '{}'", literal));
        }
    }
    Ok(())
}

fn skip_whitespace(chars: &mut core::iter::Peekable<core::str::Chars>) {
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
}

fn apply_json_path(value: &JsonValue, path: &str) -> Result<JsonValue, String> {
    if path == "$" {
        return Ok(value.clone());
    }
    
    let mut current = value.clone();
    let mut parts = path[1..].split('.').peekable(); // skip '$'
    
    while let Some(part) = parts.next() {
        if part.is_empty() { continue; }        
        // Handle array index [N] or [*] or [N:M]
        if part.starts_with('[') && part.ends_with(']') {
            let index_str = &part[1..part.len()-1];
            
            if index_str == "*" {
                // All array elements
                if let JsonValue::Array(arr) = current {
                    current = JsonValue::Array(arr);
                } else {
                    return Err("Expected array for [*]".into());
                }
            } else if index_str.contains(':') {
                // Slice [N:M]
                let mut bounds = index_str.split(':');
                let start = bounds.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                let end = bounds.next().and_then(|s| s.parse::<usize>().ok());
                
                if let JsonValue::Array(arr) = current {
                    let sliced = if let Some(end) = end {
                        arr.get(start..end.min(arr.len())).unwrap_or(&[]).to_vec()
                    } else {
                        arr.get(start..).unwrap_or(&[]).to_vec()
                    };
                    current = JsonValue::Array(sliced);
                } else {
                    return Err("Expected array for slice".into());
                }
            } else {
                // Single index [N]
                if let Ok(idx) = index_str.parse::<isize>() {
                    if let JsonValue::Array(arr) = current {
                        let actual_idx = if idx < 0 {
                            arr.len().saturating_sub(idx.unsigned_abs())
                        } else {
                            idx as usize
                        };
                        if actual_idx < arr.len() {
                            current = arr[actual_idx].clone();
                        } else {
                            return Err("Array index out of bounds".into());
                        }
                    } else {
                        return Err("Expected array for index access".into());
                    }
                } else {
                    return Err("Invalid array index".into());
                }
            }
        } else {            // Object field access .key
            if let JsonValue::Object(obj) = current {
                if let Some(val) = obj.get(part) {
                    current = val.clone();
                } else {
                    return Err(format!("Key '{}' not found", part));
                }
            } else {
                return Err("Expected object for field access".into());
            }
        }
    }
    
    Ok(current)
}

// =============================================================================
// TOML Parser (Minimal)
// =============================================================================

fn parse_toml_basic(text: &str) -> Result<JsonValue, String> {
    let mut root = alloc::collections::BTreeMap::new();
    let mut current_table = &mut root;
    let mut lines = text.lines().peekable();
    
    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        
        // Table header [section] or [[array of tables]]
        if line.starts_with('[') {
            let table_name = if line.starts_with("[[") {
                // Array of tables
                let name = line[2..line.len()-2].trim();
                // Simplified: just use the name as key
                name
            } else {
                // Regular table
                &line[1..line.len()-1]
            };
            
            // Create nested structure
            let mut parts = table_name.split('.');
            let mut current = &mut root;
            while let Some(part) = parts.next() {
                if !current.contains_key(part) {
                    current.insert(part.to_string(), JsonValue::Object(alloc::collections::BTreeMap::new()));
                }                if let JsonValue::Object(obj) = current.get_mut(part).unwrap() {
                    current = obj;
                } else {
                    return Err("Expected object in table path".into());
                }
            }
            current_table = current;
            continue;
        }
        
        // Key = value
        if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim();
            let value_str = line[eq_pos+1..].trim();
            
            let value = parse_toml_value(value_str)?;
            current_table.insert(key.to_string(), value);
        }
    }
    
    Ok(JsonValue::Object(root))
}

fn parse_toml_value(s: &str) -> Result<JsonValue, String> {
    let s = s.trim();
    
    // String (quoted)
    if s.starts_with('"') && s.ends_with('"') {
        return Ok(JsonValue::String(s[1..s.len()-1].to_string()));
    }
    
    // Boolean
    if s == "true" {
        return Ok(JsonValue::Bool(true));
    }
    if s == "false" {
        return Ok(JsonValue::Bool(false));
    }
    
    // Number
    if let Ok(num) = s.parse::<i64>() {
        return Ok(JsonValue::Number(num as f64));
    }
    if let Ok(num) = s.parse::<f64>() {
        return Ok(JsonValue::Number(num));
    }
    
    // Array [1, 2, 3]
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len()-1];        let mut arr = Vec::new();
        for item in inner.split(',') {
            arr.push(parse_toml_value(item.trim())?);
        }
        return Ok(JsonValue::Array(arr));
    }
    
    // Inline table { key = value }
    if s.starts_with('{') && s.ends_with('}') {
        let inner = &s[1..s.len()-1];
        let mut obj = alloc::collections::BTreeMap::new();
        for pair in inner.split(',') {
            if let Some(eq_pos) = pair.find('=') {
                let key = pair[..eq_pos].trim();
                let value = parse_toml_value(pair[eq_pos+1..].trim())?;
                obj.insert(key.to_string(), value);
            }
        }
        return Ok(JsonValue::Object(obj));
    }
    
    // Default to string
    Ok(JsonValue::String(s.to_string()))
}

// =============================================================================
// Base64 Encoder (RFC 4648 Standard)
// =============================================================================

fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    let mut result = String::new();
    let mut i = 0;
    
    while i < bytes.len() {
        let b0 = bytes[i] as u32;
        let b1 = if i + 1 < bytes.len() { bytes[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < bytes.len() { bytes[i + 2] as u32 } else { 0 };
        
        let triplet = (b0 << 16) | (b1 << 8) | b2;
        
        result.push(ALPHABET[((triplet >> 18) & 0x3F) as usize] as char);
        result.push(ALPHABET[((triplet >> 12) & 0x3F) as usize] as char);
        
        if i + 1 < bytes.len() {
            result.push(ALPHABET[((triplet >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }        
        if i + 2 < bytes.len() {
            result.push(ALPHABET[(triplet & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        
        i += 3;
    }
    
    result
}

// =============================================================================
// JSON Output Builder
// =============================================================================

fn build_output_json(
    path: &str,
    format: &str,
    content: &JsonValue,
    line_count: Option<usize>,
    byte_size: usize,
    truncated: bool,
    metadata: Option<&alloc::collections::BTreeMap<String, String>>,
    error: Option<&str>,
) -> String {
    let mut json = String::from("{\"path\":\"");
    json.push_str(&escape_json_string(path));
    json.push_str("\",\"format\":\"");
    json.push_str(format);
    json.push_str("\",\"content\":");
    
    // Serialize content to JSON
    json.push_str(&json_value_to_string(content));
    
    json.push_str(",\"line_count\":");
    if let Some(lc) = line_count {
        json.push_str(&lc.to_string());
    } else {
        json.push_str("null");
    }
    
    json.push_str(",\"byte_size\":");
    json.push_str(&byte_size.to_string());
    
    json.push_str(",\"truncated\":");
    json.push_str(if truncated { "true" } else { "false" });
    
    if let Some(meta) = metadata {        json.push_str(",\"metadata\":{");
        for (i, (k, v)) in meta.iter().enumerate() {
            if i > 0 { json.push(','); }
            json.push('"');
            json.push_str(&escape_json_string(k));
            json.push_str("\":\"");
            json.push_str(&escape_json_string(v));
            json.push('"');
        }
        json.push('}');
    } else {
        json.push_str(",\"metadata\":null");
    }
    
    json.push_str(",\"error\":");
    if let Some(err) = error {
        json.push('"');
        json.push_str(&escape_json_string(err));
        json.push('"');
    } else {
        json.push_str("null");
    }
    
    json.push('}');
    json
}

fn json_value_to_string(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => {
            if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                (*n as i64).to_string()
            } else {
                n.to_string()
            }
        }
        JsonValue::String(s) => format!("\"{}\"", escape_json_string(s)),
        JsonValue::Array(arr) => {
            let mut s = String::from("[");
            for (i, v) in arr.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&json_value_to_string(v));
            }
            s.push(']');
            s
        }
        JsonValue::Object(obj) => {
            let mut s = String::from("{");            for (i, (k, v)) in obj.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push('"');
                s.push_str(&escape_json_string(k));
                s.push_str("\":");
                s.push_str(&json_value_to_string(v));
            }
            s.push('}');
            s
        }
    }
}

fn escape_json_string(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() * 2);
    for ch in s.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                let code = ch as u32;
                escaped.push_str(&format!("\\u{:04x}", code));
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}

// =============================================================================
// Helper Functions
// =============================================================================

fn parse_json_string(json: &str, key: &str) -> Option<String> {
    let search_key = format!("\"{}\"", key);
    if let Some(key_pos) = json.find(&search_key) {
        if let Some(colon_pos) = json[key_pos..].find(':') {
            let start = key_pos + colon_pos + 1;
            let rest = &json[start..].trim_start();
            if rest.starts_with('"') {
                let mut end = 1;
                while end < rest.len() {
                    let ch = rest.as_bytes()[end];
                    if ch == b'"' {
                        return Some(rest[1..end].to_string());
                    } else if ch == b'\\' && end + 1 < rest.len() {
                        end += 2;                    } else {
                        end += 1;
                    }
                }
            }
        }
    }
    None
}

fn parse_json_int(json: &str, key: &str, default: i32) -> i32 {
    let search_key = format!("\"{}\"", key);
    if let Some(key_pos) = json.find(&search_key) {
        if let Some(colon_pos) = json[key_pos..].find(':') {
            let start = key_pos + colon_pos + 1;
            let rest = &json[start..].trim_start();
            let mut end = 0;
            for (i, ch) in rest.chars().enumerate() {
                if ch.is_ascii_digit() || (i == 0 && ch == '-') {
                    end = i + 1;
                } else {
                    break;
                }
            }
            if end > 0 {
                return rest[..end].parse().unwrap_or(default);
            }
        }
    }
    default
}

fn parse_json_bool(json: &str, key: &str, default: bool) -> bool {
    let search_key = format!("\"{}\"", key);
    if let Some(key_pos) = json.find(&search_key) {
        if let Some(colon_pos) = json[key_pos..].find(':') {
            let start = key_pos + colon_pos + 1;
            let rest = &json[start..].trim_start();
            if rest.starts_with("true") {
                return true;
            } else if rest.starts_with("false") {
                return false;
            }
        }
    }
    default
}

fn write_error(path: &str, error_msg: &str) {
    let error_json = format!(        "{{\"path\":\"{}\",\"format\":\"unknown\",\"content\":null,\"line_count\":null,\"byte_size\":0,\"truncated\":false,\"metadata\":null,\"error\":\"{}\"}}",
        escape_json_string(path),
        escape_json_string(error_msg)
    );
    let bytes = error_json.as_bytes();
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), OUTPUT_BUFFER.as_mut_ptr(), bytes.len());
        nexus_set_output_len(bytes.len() as u32);
    }
}

fn log_info(msg: &str) {
    unsafe {
        nexus_log(2, msg.as_ptr() as *const c_char, msg.len() as i32);
    }
}

fn log_warn(msg: &str) {
    unsafe {
        nexus_log(3, msg.as_ptr() as *const c_char, msg.len() as i32);
    }
}

// =============================================================================
// Main Entry Point
// =============================================================================

#[no_mangle]
pub extern "C" fn _nexus_run() -> i32 {
    // 1. Parse input
    let input_json = unsafe {
        let ptr = nexus_get_input_ptr();
        let len = nexus_get_input_len() as usize;
        if len == 0 || len > INPUT_BUFFER.len() {
            write_error("", "Invalid input length");
            return 1;
        }
        str::from_utf8(&INPUT_BUFFER[..len]).unwrap_or("")
    };
    
    let path = match parse_json_string(input_json, "path") {
        Some(p) => p,
        None => {
            write_error("", "Missing 'path' field in input");
            return 1;
        }
    };
    
    // 2. Validate path safety
    if !is_safe_path(&path) {        write_error(&path, "Path validation failed");
        return 1;
    }
    
    let format = parse_json_string(input_json, "format");
    let encoding = parse_json_string(input_json, "encoding").unwrap_or_else(|| "utf8".to_string());
    let max_bytes = parse_json_int(input_json, "max_bytes", 1048576) as usize;
    let start_line = if let Some(sl) = parse_json_int(input_json, "start_line", -1) {
        if sl > 0 { Some(sl as usize) } else { None }
    } else { None };
    let end_line = if let Some(el) = parse_json_int(input_json, "end_line", -1) {
        if el > 0 { Some(el as usize) } else { None }
    } else { None };
    let csv_delimiter = parse_json_string(input_json, "csv_delimiter")
        .unwrap_or_else(|| ",".to_string())
        .chars().next().unwrap_or(',');
    let csv_has_header = parse_json_bool(input_json, "csv_has_header", true);
    let json_path = parse_json_string(input_json, "json_path");
    let pattern = parse_json_string(input_json, "pattern");
    let include_metadata = parse_json_bool(input_json, "include_metadata", false);
    
    // 3. Read file
    let file_bytes = match read_file(&path, max_bytes) {
        Ok(bytes) => bytes,
        Err(e) => {
            write_error(&path, e);
            return 1;
        }
    };
    
    let byte_size = file_bytes.len();
    
    // 4. Detect format if not specified
    let detected_format = format.as_deref().unwrap_or_else(|| detect_format(&path, &file_bytes));
    
    // 5. Process based on format
    let (content, line_count, truncated) = match detected_format {
        "text" => {
            let processed = process_text(&file_bytes, &InputParams {
                path: path.clone(),
                format: format.clone(),
                encoding: encoding.clone(),
                max_bytes,
                start_line,
                end_line,
                csv_delimiter,
                csv_has_header,
                json_path: json_path.clone(),
                pattern: pattern.clone(),
                include_metadata,            });
            (JsonValue::String(processed.content), processed.line_count, processed.truncated)
        }
        "json" => {
            if let Ok(text) = str::from_utf8(&file_bytes) {
                match parse_json(text) {
                    Ok(mut json_val) => {
                        // Apply JSONPath if specified
                        if let Some(ref jp) = json_path {
                            match apply_json_path(&json_val, jp) {
                                Ok(filtered) => json_val = filtered,
                                Err(e) => {
                                    write_error(&path, &format!("JSONPath error: {}", e));
                                    return 1;
                                }
                            }
                        }
                        (json_val, None, false)
                    }
                    Err(e) => {
                        write_error(&path, &format!("JSON parse error: {}", e));
                        return 1;
                    }
                }
            } else {
                write_error(&path, "Invalid UTF-8 in JSON file");
                return 1;
            }
        }
        "csv" => {
            if let Ok(text) = str::from_utf8(&file_bytes) {
                let csv_result = parse_csv(text, csv_delimiter, csv_has_header);
                // Convert CSV result map to JSON
                let mut content_obj = alloc::collections::BTreeMap::new();
                for (k, v) in csv_result {
                    content_obj.insert(k, JsonValue::String(v));
                }
                (JsonValue::Object(content_obj), None, csv_result.contains_key("truncated"))
            } else {
                write_error(&path, "Invalid UTF-8 in CSV file");
                return 1;
            }
        }
        "toml" => {
            if let Ok(text) = str::from_utf8(&file_bytes) {
                match parse_toml_basic(text) {
                    Ok(json_val) => (json_val, None, false),
                    Err(e) => {
                        write_error(&path, &format!("TOML parse error: {}", e));
                        return 1;                    }
                }
            } else {
                write_error(&path, "Invalid UTF-8 in TOML file");
                return 1;
            }
        }
        "yaml" => {
            // Treat YAML as text for now (full YAML parsing is complex)
            let processed = process_text(&file_bytes, &InputParams {
                path: path.clone(),
                format: format.clone(),
                encoding: encoding.clone(),
                max_bytes,
                start_line,
                end_line,
                csv_delimiter,
                csv_has_header,
                json_path: json_path.clone(),
                pattern: pattern.clone(),
                include_metadata,
            });
            (JsonValue::String(processed.content), processed.line_count, processed.truncated)
        }
        "binary" => {
            // Cap at 256KB for base64 output
            let cap = 256 * 1024;
            let truncated = file_bytes.len() > cap;
            let encoded = base64_encode(if truncated { &file_bytes[..cap] } else { &file_bytes });
            (JsonValue::String(encoded), None, truncated)
        }
        _ => {
            write_error(&path, &format!("Unsupported format: {}", detected_format));
            return 1;
        }
    };
    
    // 6. Build metadata if requested
    let metadata = if include_metadata {
        let mut meta = alloc::collections::BTreeMap::new();
        if let Some(lc) = line_count {
            meta.insert("lines_returned".to_string(), lc.to_string());
        }
        if let Some(sl) = start_line {
            meta.insert("start_line".to_string(), sl.to_string());
        }
        if let Some(el) = end_line {
            meta.insert("end_line".to_string(), el.to_string());
        }
        meta.insert("encoding".to_string(), encoding.clone());        Some(meta)
    } else {
        None
    };
    
    // 7. Build and write output JSON
    let output_json = build_output_json(
        &path,
        detected_format,
        &content,
        line_count,
        byte_size,
        truncated,
        metadata.as_ref(),
        None,
    );
    
    let output_bytes = output_json.as_bytes();
    if output_bytes.len() > OUTPUT_BUFFER.len() {
        write_error(&path, "Output too large for buffer");
        return 1;
    }
    
    unsafe {
        ptr::copy_nonoverlapping(output_bytes.as_ptr(), OUTPUT_BUFFER.as_mut_ptr(), output_bytes.len());
        nexus_set_output_len(output_bytes.len() as u32);
    }
    
    log_info(&format!("Successfully read file: {} (format: {})", path, detected_format));
    0 // Success
}
