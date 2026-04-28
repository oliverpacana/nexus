// tools/http-fetch/src/lib.rs
// WASM tool for fetching and extracting content from HTTP/HTTPS URLs
// Compile with: cargo build --target wasm32-unknown-unknown --release

#![no_std]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
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
    
    fn nexus_http_post(
        url_ptr: *const c_char,
        url_len: i32,
        body_ptr: *const c_char,
        body_len: i32,
        out_ptr: *mut c_char,
        out_max: i32,
    ) -> i32;
    
    fn nexus_log(level: i32, msg_ptr: *const c_char, msg_len: i32);
}

// =============================================================================
// Static Buffers
// =============================================================================

static mut INPUT_BUFFER: [u8; 65536] = [0u8; 65536];
static mut OUTPUT_BUFFER: [u8; 524288] = [0u8; 524288]; // 512KB for page content
static mut INPUT_LEN: u32 = 0;
static mut OUTPUT_LEN: u32 = 0;
static mut HTTP_RESPONSE_BUFFER: [u8; 524288] = [0u8; 524288]; // Intermediate buffer for HTTP response

// =============================================================================
// ABI Exports// =============================================================================

#[no_mangle]
pub extern "C" fn nexus_get_input_ptr() -> *mut u8 {
    unsafe { INPUT_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn nexus_get_input_len() -> u32 {
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
// Input/Output Structures
// =============================================================================

struct InputParams {
    url: String,
    method: String,
    body: Option<String>,
    extract_text: bool,
    max_chars: usize,
    follow_redirects: bool,
    include_headers: bool,
}

struct FetchResult {
    final_url: String,
    status: u16,
    content: Vec<u8>,    error: Option<String>,
}

// =============================================================================
// Minimal JSON Parser (no_std compatible)
// =============================================================================

/// Parses a JSON string value for a given key.
/// Returns None if key not found or value is not a string.
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
                        end += 2; // Skip escaped char
                    } else {
                        end += 1;
                    }
                }
            }
        }
    }
    None
}

/// Parses a JSON boolean value for a given key.
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
/// Parses a JSON integer value for a given key.
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

/// Parses the input JSON from INPUT_BUFFER into InputParams.
fn parse_input() -> Option<InputParams> {
    unsafe {
        let input_ptr = nexus_get_input_ptr();
        let input_len = nexus_get_input_len() as usize;
        if input_len == 0 || input_len > INPUT_BUFFER.len() {
            return None;
        }
        let input_str = str::from_utf8(&INPUT_BUFFER[..input_len]).ok()?;
        
        let url = parse_json_string(input_str, "url")?;
        let method = parse_json_string(input_str, "method").unwrap_or_else(|| "GET".to_string());
        let body = parse_json_string(input_str, "body");
        let extract_text = parse_json_bool(input_str, "extract_text", true);
        let max_chars = parse_json_int(input_str, "max_chars", 50000) as usize;
        let follow_redirects = parse_json_bool(input_str, "follow_redirects", true);
        let include_headers = parse_json_bool(input_str, "include_headers", false);
        
        Some(InputParams {
            url,
            method,
            body,
            extract_text,
            max_chars,
            follow_redirects,
            include_headers,
        })    }
}

// =============================================================================
// URL Validation
// =============================================================================

/// Validates a URL for security and format.
fn validate_url(url: &str) -> bool {
    // Must start with http:// or https://
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return false;
    }
    
    // Max length check
    if url.len() > 2048 {
        return false;
    }
    
    // Extract host (between :// and next / or end)
    let after_protocol = if url.starts_with("https://") {
        &url[8..]
    } else {
        &url[7..]
    };
    
    let host_end = after_protocol.find('/').unwrap_or(after_protocol.len());
    let host = &after_protocol[..host_end];
    
    // Remove port if present
    let host = host.split(':').next().unwrap_or(host);
    
    // Must have at least one dot (not localhost IP)
    if !host.contains('.') {
        return false;
    }
    
    // Block private/internal IPs
    if host.starts_with("127.") || 
       host.starts_with("192.168.") || 
       host.starts_with("10.") || 
       host.starts_with("169.254.") ||
       host == "localhost" ||
       host == "::1" {
        return false;
    }
    
    // No spaces or control characters
    if host.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return false;    }
    
    true
}

// =============================================================================
// HTTP Fetch
// =============================================================================

/// Performs the HTTP fetch using host functions.
fn do_fetch(params: &InputParams) -> Result<FetchResult, &'static str> {
    let url_cstr = alloc::ffi::CString::new(params.url.as_str()).map_err(|_| "URL conversion failed")?;
    let body_cstr = params.body.as_ref().map(|b| alloc::ffi::CString::new(b.as_str()).ok());
    
    let response_ptr = unsafe { HTTP_RESPONSE_BUFFER.as_mut_ptr() as *mut c_char };
    let response_max = HTTP_RESPONSE_BUFFER.len() as i32;
    
    let response_len = unsafe {
        if params.method == "POST" {
            if let Some(body) = &params.body {
                let body_ptr = body.as_ptr() as *const c_char;
                nexus_http_post(
                    url_cstr.as_ptr(),
                    params.url.len() as i32,
                    body_ptr,
                    body.len() as i32,
                    response_ptr,
                    response_max,
                )
            } else {
                return Err("POST method requires body");
            }
        } else {
            nexus_http_get(
                url_cstr.as_ptr(),
                params.url.len() as i32,
                response_ptr,
                response_max,
            )
        }
    };
    
    if response_len < 0 {
        let error_msg = match response_len {
            -1 => "Host not allowed by manifest",
            -2 => "HTTP request failed",
            -3 => "Response buffer too small",
            _ => "Unknown HTTP error",
        };
        return Err(error_msg);    }
    
    // For simplicity, we assume status 200 on success
    // In a real implementation, we'd parse HTTP status from response headers
    Ok(FetchResult {
        final_url: params.url.clone(), // Follow redirects would update this
        status: 200,
        content: unsafe { HTTP_RESPONSE_BUFFER[..response_len as usize].to_vec() },
        error: None,
    })
}

// =============================================================================
// HTML Text Extraction (State Machine Parser)
// =============================================================================

/// Extracts clean text, title, and links from HTML.
fn extract_text_from_html(html: &[u8]) -> (String, Option<String>, Vec<String>) {
    let html_str = str::from_utf8(html).unwrap_or("");
    
    // State machine for parsing
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut in_comment = false;
    let mut in_title = false;
    let mut in_a_tag = false;
    let mut current_attr = String::new();
    let mut in_href = false;
    
    let mut text_buffer = String::new();
    let mut title_buffer = String::new();
    let mut links: Vec<String> = Vec::with_capacity(50);
    let mut chars = html_str.chars().peekable();
    
    while let Some(ch) = chars.next() {
        // Handle HTML comments
        if !in_comment && ch == '<' && chars.peek() == Some(&'!') {
            let mut peek = chars.clone();
            peek.next();
            if peek.peek() == Some(&'-') {
                peek.next();
                if peek.peek() == Some(&'-') {
                    in_comment = true;
                    continue;
                }
            }
        }
        if in_comment {
            if ch == '>' && html_str[..].ends_with("-->") {                in_comment = false;
            }
            continue;
        }
        
        // Handle script/style tags
        if !in_script && !in_style && ch == '<' {
            let mut tag_name = String::new();
            let mut temp_chars = chars.clone();
            while let Some(&next_ch) = temp_chars.peek() {
                if next_ch.is_alphanumeric() || next_ch == '-' {
                    tag_name.push(temp_chars.next().unwrap());
                } else {
                    break;
                }
            }
            let tag_lower = tag_name.to_lowercase();
            if tag_lower == "script" {
                in_script = true;
                continue;
            } else if tag_lower == "style" {
                in_style = true;
                continue;
            } else if tag_lower == "title" {
                in_title = true;
                continue;
            } else if tag_lower == "a" {
                in_a_tag = true;
                in_tag = true;
                continue;
            }
        }
        
        if in_script {
            if ch == '<' {
                let mut temp = chars.clone();
                let mut tag = String::new();
                while let Some(&c) = temp.peek() {
                    if c.is_alphanumeric() || c == '-' {
                        tag.push(temp.next().unwrap());
                    } else {
                        break;
                    }
                }
                if tag.to_lowercase() == "/script" {
                    in_script = false;
                }
            }
            continue;
        }        
        if in_style {
            if ch == '<' {
                let mut temp = chars.clone();
                let mut tag = String::new();
                while let Some(&c) = temp.peek() {
                    if c.is_alphanumeric() || c == '-' {
                        tag.push(temp.next().unwrap());
                    } else {
                        break;
                    }
                }
                if tag.to_lowercase() == "/style" {
                    in_style = false;
                }
            }
            continue;
        }
        
        // Handle tags
        if ch == '<' {
            in_tag = true;
            continue;
        }
        if ch == '>' {
            in_tag = false;
            if in_title {
                in_title = false;
            }
            if in_a_tag {
                in_a_tag = false;
                in_href = false;
                current_attr.clear();
            }
            continue;
        }
        
        if in_tag {
            // Parse attributes
            if ch == ' ' || ch == '\t' || ch == '\n' {
                if in_href && !current_attr.is_empty() {
                    if links.len() < 50 {
                        links.push(current_attr.clone());
                    }
                    current_attr.clear();
                    in_href = false;
                }
                continue;
            }
            if ch == '=' {                if current_attr.to_lowercase() == "href" {
                    in_href = true;
                }
                current_attr.clear();
                continue;
            }
            if ch == '"' || ch == '\'' {
                if in_href {
                    let quote = ch;
                    let mut attr_value = String::new();
                    while let Some(&next_ch) = chars.peek() {
                        if next_ch == quote {
                            chars.next();
                            break;
                        }
                        attr_value.push(chars.next().unwrap());
                    }
                    if links.len() < 50 {
                        links.push(attr_value);
                    }
                    in_href = false;
                }
                current_attr.clear();
                continue;
            }
            current_attr.push(ch);
            continue;
        }
        
        // Handle text content
        if in_title {
            title_buffer.push(ch);
        } else {
            // Decode HTML entities
            if ch == '&' {
                let mut entity = String::new();
                while let Some(&next_ch) = chars.peek() {
                    if next_ch == ';' {
                        chars.next();
                        break;
                    }
                    entity.push(chars.next().unwrap());
                }
                let decoded = match entity.as_str() {
                    "amp" => '&',
                    "lt" => '<',
                    "gt" => '>',
                    "quot" => '"',
                    "apos" => '\'',
                    "nbsp" => ' ',                    _ => {
                        // Handle numeric entities
                        if entity.starts_with('#') {
                            let num_str = &entity[1..];
                            if num_str.starts_with('x') {
                                u32::from_str_radix(&num_str[1..], 16)
                                    .ok()
                                    .and_then(|c| char::from_u32(c))
                                    .unwrap_or(' ')
                            } else {
                                num_str.parse::<u32>()
                                    .ok()
                                    .and_then(|c| char::from_u32(c))
                                    .unwrap_or(' ')
                            }
                        } else {
                            ' '
                        }
                    }
                };
                text_buffer.push(decoded);
            } else {
                text_buffer.push(ch);
            }
        }
    }
    
    // Clean up whitespace
    let clean_text = text_buffer
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    
    let clean_title = if title_buffer.is_empty() {
        None
    } else {
        Some(title_buffer.trim().to_string())
    };
    
    (clean_text, clean_title, links)
}

// =============================================================================
// Content Truncation
// =============================================================================

/// Truncates a string at a UTF-8 character boundary, trying to break at a sentence boundary.
fn truncate_to_chars(s: &str, max_chars: usize) -> (String, bool) {
    if s.len() <= max_chars {
        return (s.to_string(), false);    }
    
    // Find truncation point
    let mut char_count = 0;
    let mut byte_idx = 0;
    for (idx, ch) in s.char_indices() {
        if char_count >= max_chars {
            byte_idx = idx;
            break;
        }
        char_count += 1;
        byte_idx = idx + ch.len_utf8();
    }
    
    // Try to find sentence boundary within 200 chars of limit
    let search_start = if char_count > 200 { char_count - 200 } else { 0 };
    let search_end = char_count.min(s.chars().count());
    
    let mut best_boundary = byte_idx;
    let mut last_period = None;
    
    for (idx, ch) in s.char_indices().skip(search_start).take(search_end - search_start) {
        if ch == '.' {
            last_period = Some(idx);
        }
        if ch == '.' && idx + 1 < s.len() && s.as_bytes()[idx + 1] == b' ' {
            best_boundary = idx + 2; // Include the space after period
            break;
        }
    }
    
    if let Some(period_idx) = last_period {
        if period_idx + 1 < s.len() && s.as_bytes()[period_idx + 1] == b' ' {
            best_boundary = period_idx + 2;
        }
    }
    
    (s[..best_boundary].to_string(), true)
}

// =============================================================================
// JSON Output Builder
// =============================================================================

/// Builds the output JSON string manually.
fn build_output_json(result: &FetchResult, params: &InputParams, extracted: Option<(String, Option<String>, Vec<String>)>) -> String {
    let mut json = String::from("{\"url\":\"");
    json.push_str(&escape_json_string(&result.final_url));
    json.push_str("\",\"status\":");
    json.push_str(&result.status.to_string());    json.push_str(",\"content\":\"");
    
    if let Some((text, _title, _links)) = &extracted {
        let (truncated_text, was_truncated) = truncate_to_chars(text, params.max_chars);
        json.push_str(&escape_json_string(&truncated_text));
        json.push_str("\",\"content_length\":");
        json.push_str(&truncated_text.len().to_string());
        json.push_str(",\"truncated\":");
        json.push_str(if was_truncated { "true" } else { "false" });
        
        // Add title if extracted
        json.push_str(",\"title\":");
        if let Some(title) = _title {
            json.push('"');
            json.push_str(&escape_json_string(title));
            json.push('"');
        } else {
            json.push_str("null");
        }
        
        // Add links (max 50)
        json.push_str(",\"links\":[");
        let links_to_show = _links.iter().take(50);
        for (i, link) in links_to_show.enumerate() {
            if i > 0 { json.push(','); }
            json.push('"');
            json.push_str(&escape_json_string(link));
            json.push('"');
        }
        json.push(']');
    } else {
        // Raw content
        let content_str = str::from_utf8(&result.content).unwrap_or("");
        let (truncated_content, was_truncated) = truncate_to_chars(content_str, params.max_chars);
        json.push_str(&escape_json_string(&truncated_content));
        json.push_str("\",\"content_length\":");
        json.push_str(&truncated_content.len().to_string());
        json.push_str(",\"truncated\":");
        json.push_str(if was_truncated { "true" } else { "false" });
        json.push_str(",\"title\":null,\"links\":[]");
    }
    
    json.push_str(",\"error\":");
    if let Some(error) = &result.error {
        json.push('"');
        json.push_str(&escape_json_string(error));
        json.push('"');
    } else {
        json.push_str("null");
    }    
    json.push('}');
    json
}

/// Escapes a string for safe inclusion in JSON.
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

/// Writes an error JSON to the output buffer.
fn write_error(url: &str, error_msg: &str) {
    let error_json = format!(
        "{{\"url\":\"{}\",\"status\":0,\"content\":\"\",\"content_length\":0,\"truncated\":false,\"title\":null,\"links\":[],\"error\":\"{}\"}}",
        escape_json_string(url),
        escape_json_string(error_msg)
    );
    let bytes = error_json.as_bytes();
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), OUTPUT_BUFFER.as_mut_ptr(), bytes.len());
        nexus_set_output_len(bytes.len() as u32);
    }
}

/// Logs a message via the host nexus_log function.
fn log_info(msg: &str) {
    unsafe {
        nexus_log(2, msg.as_ptr() as *const c_char, msg.len() as i32);
    }
}

// =============================================================================
// Main Entry Point
// =============================================================================
#[no_mangle]
pub extern "C" fn _nexus_run() -> i32 {
    // 1. Parse input
    let params = match parse_input() {
        Some(p) => p,
        None => {
            write_error("", "Failed to parse input JSON");
            return 1;
        }
    };
    
    log_info(&format!("Fetching URL: {} (method: {})", params.url, params.method));
    
    // 2. Validate URL
    if !validate_url(&params.url) {
        write_error(&params.url, "Invalid or blocked URL");
        return 1;
    }
    
    // 3. Perform HTTP fetch
    let fetch_result = match do_fetch(&params) {
        Ok(res) => res,
        Err(e) => {
            write_error(&params.url, e);
            return 1;
        }
    };
    
    // 4. Extract text if requested
    let extracted = if params.extract_text {
        Some(extract_text_from_html(&fetch_result.content))
    } else {
        None
    };
    
    // 5. Build and write output JSON
    let output_json = build_output_json(&fetch_result, &params, extracted);
    let output_bytes = output_json.as_bytes();
    
    if output_bytes.len() > OUTPUT_BUFFER.len() {
        write_error(&params.url, "Output too large for buffer");
        return 1;
    }
    
    unsafe {
        ptr::copy_nonoverlapping(output_bytes.as_ptr(), OUTPUT_BUFFER.as_mut_ptr(), output_bytes.len());
        nexus_set_output_len(output_bytes.len() as u32);
    }
    
    log_info("HTTP fetch completed successfully");    0 // Success
}
