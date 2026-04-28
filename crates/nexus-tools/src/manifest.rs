use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;

use nexus_proto::tool::{ResourceLimits, ToolCapabilityRequirement, ToolManifest as ProtoManifest, ToolId};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, instrument};

// =============================================================================
// ToolError — Manifest Parsing and Validation Errors
// =============================================================================

/// Errors that can occur during tool manifest parsing or validation.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("failed to read manifest file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse TOML: {0}")]
    TomlError(#[from] toml::de::Error),

    #[error("invalid manifest field '{field}': {reason}")]
    ManifestInvalid {
        field: String,
        reason: String,
    },

    #[error("invalid JSON schema for {schema_type}: {reason}")]
    SchemaInvalid {
        schema_type: String,
        reason: String,
    },

    #[error("manifest file not found at: {0}")]
    NotFound(PathBuf),

    #[error("checksum mismatch for WASM binary: expected {expected}, got {actual}")]
    ChecksumMismatch {
        expected: String,
        actual: String,
    },
}

pub type Result<T> = std::result::Result<T, ToolError>;

// =============================================================================
// RawManifest — TOML-Deserialized Structure
// =============================================================================
/// The raw structure matching the TOML manifest file format.
/// This is an intermediate representation before conversion to `ProtoManifest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RawManifest {
    /// Core tool metadata.
    pub tool: ToolSection,

    /// List of required host capabilities.
    #[serde(default, rename = "capabilities_required")]
    pub capabilities_required: Vec<RawCapability>,

    /// Resource limits for WASM sandbox execution.
    #[serde(default)]
    pub resource_limits: RawResourceLimits,

    /// JSON Schema for validating tool input arguments.
    #[serde(default)]
    pub input_schema: serde_json::Value,

    /// JSON Schema for validating tool output results.
    #[serde(default)]
    pub output_schema: serde_json::Value,

    /// Optional filesystem path to the compiled WASM binary.
    #[serde(default)]
    pub wasm_path: Option<String>,

    /// Optional SHA-256 hex digest for WASM integrity verification.
    #[serde(default)]
    pub checksum_sha256: Option<String>,
}

/// Core metadata section from the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolSection {
    /// Human-readable tool name (e.g., "web-search").
    pub name: String,

    /// Semantic version string (e.g., "1.0.0").
    pub version: String,

    /// Description of the tool's purpose and behavior.
    pub description: String,

    /// Optional author name or organization.
    pub author: Option<String>,

    /// Optional SPDX license identifier.    pub license: Option<String>,
}

/// A single capability requirement as specified in TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum RawCapability {
    /// Network access with optional host allowlist.
    Network {
        #[serde(default)]
        allowed_hosts: Vec<String>,
    },

    /// Read-only filesystem access to specified paths.
    FilesystemRead {
        paths: Vec<String>,
    },

    /// Read-write filesystem access to specified paths.
    FilesystemWrite {
        paths: Vec<String>,
    },

    /// Access to non-deterministic random number generation.
    Random,

    /// Access to wall-clock time and monotonic clocks.
    Clock,

    /// Permission to emit structured log entries.
    Logging,
}

/// Resource limits as specified in TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RawResourceLimits {
    #[serde(default = "default_max_memory_mb")]
    pub max_memory_mb: u32,

    #[serde(default = "default_max_fuel")]
    pub max_fuel: u64,

    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    #[serde(default = "default_max_output_bytes")]
    pub max_output_bytes: usize,
}
fn default_max_memory_mb() -> u32 { 64 }
fn default_max_fuel() -> u64 { 10_000_000 }
fn default_timeout_ms() -> u64 { 5000 }
fn default_max_output_bytes() -> usize { 1_048_576 }

impl Default for RawResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: default_max_memory_mb(),
            max_fuel: default_max_fuel(),
            timeout_ms: default_timeout_ms(),
            max_output_bytes: default_max_output_bytes(),
        }
    }
}

// =============================================================================
// Conversion: RawManifest → ProtoManifest
// =============================================================================

impl RawManifest {
    /// Converts this raw manifest into the canonical `ProtoManifest` type.
    ///
    /// # Arguments
    /// * `default_wasm_path` - Optional fallback path if not specified in manifest
    ///
    /// # Returns
    /// * `Ok(ProtoManifest)` - If all fields are valid
    /// * `Err(ToolError)` - If validation fails
    fn into_proto(self, default_wasm_path: Option<PathBuf>) -> Result<ProtoManifest> {
        // Validate core fields
        validate_name(&self.tool.name)?;
        validate_version(&self.tool.version)?;

        // Convert capabilities
        let capabilities = self
            .capabilities_required
            .into_iter()
            .map(convert_capability)
            .collect::<Result<Vec<_>>>()?;

        // Convert resource limits
        let limits = convert_resource_limits(self.resource_limits)?;

        // Validate JSON schemas
        validate_json_schema(&self.input_schema, "input_schema")?;
        validate_json_schema(&self.output_schema, "output_schema")?;

        // Resolve WASM path
        let wasm_path = self            .wasm_path
            .map(PathBuf::from)
            .or(default_wasm_path)
            .map(|p| p.to_string_lossy().into_owned());

        Ok(ProtoManifest {
            name: self.tool.name,
            version: self.tool.version,
            description: self.tool.description,
            author: self.tool.author,
            license: self.license,
            capabilities_required: capabilities,
            resource_limits: limits,
            input_schema: self.input_schema,
            output_schema: self.output_schema,
            wasm_path,
            checksum_sha256: self.checksum_sha256,
        })
    }
}

/// Validates the tool name field.
fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(ToolError::ManifestInvalid {
            field: "tool.name".into(),
            reason: "name cannot be empty".into(),
        });
    }

    // Allow alphanumeric, hyphens, underscores; no spaces or special chars
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(ToolError::ManifestInvalid {
            field: "tool.name".into(),
            reason: "name must contain only alphanumeric characters, hyphens, or underscores".into(),
        });
    }

    Ok(())
}

/// Validates the version string is semver-ish (x.y.z format).
fn validate_version(version: &str) -> Result<()> {
    if version.is_empty() {
        return Err(ToolError::ManifestInvalid {
            field: "tool.version".into(),
            reason: "version cannot be empty".into(),
        });
    }
    // Basic semver-ish check: at least two dots with numeric segments
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() < 2 {
        return Err(ToolError::ManifestInvalid {
            field: "tool.version".into(),
            reason: "version must be in x.y[.z] format".into(),
        });
    }

    // First two parts must be numeric
    for (i, part) in parts.iter().take(2).enumerate() {
        if part.parse::<u32>().is_err() {
            return Err(ToolError::ManifestInvalid {
                field: "tool.version".into(),
                reason: format!("version segment {} must be numeric", i + 1),
            });
        }
    }

    Ok(())
}

/// Converts a `RawCapability` to the canonical `ToolCapabilityRequirement`.
fn convert_capability(raw: RawCapability) -> Result<ToolCapabilityRequirement> {
    Ok(match raw {
        RawCapability::Network { allowed_hosts } => {
            ToolCapabilityRequirement::NetworkAccess { allowed_hosts }
        }
        RawCapability::FilesystemRead { paths } => {
            ToolCapabilityRequirement::FilesystemRead { paths }
        }
        RawCapability::FilesystemWrite { paths } => {
            ToolCapabilityRequirement::FilesystemWrite { paths }
        }
        RawCapability::Random => ToolCapabilityRequirement::RandomAccess,
        RawCapability::Clock => ToolCapabilityRequirement::ClockAccess,
        RawCapability::Logging => ToolCapabilityRequirement::LoggingAccess,
    })
}

/// Converts `RawResourceLimits` to canonical `ResourceLimits` with validation.
fn convert_resource_limits(raw: RawResourceLimits) -> Result<ResourceLimits> {
    if raw.max_memory_mb == 0 {
        return Err(ToolError::ManifestInvalid {
            field: "resource_limits.max_memory_mb".into(),
            reason: "must be positive".into(),
        });
    }
    if raw.max_fuel == 0 {
        return Err(ToolError::ManifestInvalid {            field: "resource_limits.max_fuel".into(),
            reason: "must be positive".into(),
        });
    }
    if raw.timeout_ms == 0 {
        return Err(ToolError::ManifestInvalid {
            field: "resource_limits.timeout_ms".into(),
            reason: "must be positive".into(),
        });
    }
    if raw.max_output_bytes == 0 {
        return Err(ToolError::ManifestInvalid {
            field: "resource_limits.max_output_bytes".into(),
            reason: "must be positive".into(),
        });
    }

    Ok(ResourceLimits {
        max_memory_mb: raw.max_memory_mb,
        max_fuel: raw.max_fuel,
        timeout_ms: raw.timeout_ms,
        max_output_bytes: raw.max_output_bytes,
    })
}

/// Validates that a value is a valid JSON Schema using jsonschema crate.
fn validate_json_schema(schema: &serde_json::Value, schema_type: &str) -> Result<()> {
    // Empty schema is valid (allows anything)
    if schema.is_null() || (schema.is_object() && schema.as_object().map_or(true, |o| o.is_empty())) {
        return Ok(());
    }

    match jsonschema::JSONSchema::compile(schema) {
        Ok(_) => Ok(()),
        Err(e) => Err(ToolError::SchemaInvalid {
            schema_type: schema_type.into(),
            reason: e.to_string(),
        }),
    }
}

// =============================================================================
// Public API: Parsing and Loading
// =============================================================================

/// Parses a tool manifest from a TOML string.
///
/// # Arguments
/// * `toml_str` - The raw TOML content to parse
/// * `default_wasm_path` - Optional fallback path if not specified in manifest///
/// # Returns
/// * `Ok(ToolManifest)` - If parsing and validation succeed
/// * `Err(ToolError)` - If any step fails
#[instrument(skip(toml_str), fields(name = ?extract_name(toml_str)))]
pub fn parse_manifest(toml_str: &str, default_wasm_path: Option<PathBuf>) -> Result<ProtoManifest> {
    debug!("parsing tool manifest from TOML string");

    // Parse TOML into raw structure
    let raw: RawManifest = toml::from_str(toml_str)
        .map_err(ToolError::TomlError)?;

    // Convert and validate
    raw.into_proto(default_wasm_path)
}

/// Helper to extract tool name for logging (best effort, doesn't fail).
fn extract_name(toml_str: &str) -> Option<String> {
    toml_str
        .lines()
        .find(|l| l.trim_start().starts_with("name"))
        .and_then(|l| l.split('=').nth(1))
        .map(|v| v.trim().trim_matches('"').to_string())
}

/// Loads and parses a manifest from a TOML file.
///
/// # Arguments
/// * `path` - Filesystem path to the manifest TOML file
///
/// # Returns
/// * `Ok(ToolManifest)` - If file read, parse, and validation succeed
/// * `Err(ToolError)` - If any step fails
#[instrument(skip(path), fields(path = ?path))]
pub fn load_manifest_from_file(path: &Path) -> Result<ProtoManifest> {
    debug!("loading tool manifest from file");

    if !path.exists() {
        return Err(ToolError::NotFound(path.to_path_buf()));
    }

    let toml_str = fs::read_to_string(path)?;
    let default_wasm_path = path.parent().map(|p| p.to_path_buf());

    parse_manifest(&toml_str, default_wasm_path)
}

/// Loads a manifest from a file located alongside a WASM binary.
///
/// # Search Strategy/// Given `wasm_path = "/tools/web-search/web-search.wasm"`, looks for:
/// 1. `/tools/web-search/web-search.toml` (same name, .toml extension)
/// 2. `/tools/web-search/manifest.toml` (fixed name)
/// 3. `/tools/web-search/WebSearch.toml` (PascalCase name)
///
/// # Arguments
/// * `wasm_path` - Path to the WASM binary; manifest is searched relative to this
///
/// # Returns
/// * `Ok(ToolManifest)` - If manifest found and valid
/// * `Err(ToolError)` - If not found or invalid
#[instrument(skip(wasm_path), fields(wasm_path = ?wasm_path))]
pub fn load_manifest_alongside_wasm(wasm_path: &Path) -> Result<ProtoManifest> {
    debug!("searching for tool manifest alongside WASM binary");

    let parent = wasm_path.parent().ok_or_else(|| ToolError::NotFound(wasm_path.to_path_buf()))?;

    // Candidate manifest paths in priority order
    let candidates = {
        let stem = wasm_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("tool");

        vec![
            parent.join(format!("{}.toml", stem)),      // web-search.toml
            parent.join("manifest.toml"),                // manifest.toml
            parent.join(format!("{}Manifest.toml", capitalize(stem))), // WebSearchManifest.toml
        ]
    };

    // Try each candidate
    for candidate in &candidates {
        if candidate.exists() {
            debug!(path = ?candidate, "found manifest candidate");
            return load_manifest_from_file(candidate);
        }
    }

    Err(ToolError::NotFound(candidates[0].clone()))
}

/// Capitalizes the first letter of a string (for PascalCase conversion).
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
// =============================================================================
// Utility: Compute SHA-256 Checksum for WASM Binary
// =============================================================================

/// Computes the SHA-256 hex digest of a file's contents.
///
/// # Arguments
/// * `path` - Path to the file to hash
///
/// # Returns
/// * `Ok(String)` - Lowercase hex digest of the file's SHA-256 hash
/// * `Err(ToolError)` - If file cannot be read or hashed
#[instrument(skip(path), fields(path = ?path))]
pub fn compute_wasm_checksum(path: &Path) -> Result<String> {
    use sha2::{Sha256, Digest};

    let contents = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    let result = hasher.finalize();

    Ok(hex::encode(result))
}

/// Verifies that a WASM binary matches its expected checksum.
///
/// # Arguments
/// * `wasm_path` - Path to the WASM binary
/// * `expected` - Expected SHA-256 hex digest (lowercase)
///
/// # Returns
/// * `Ok(())` - If checksum matches
/// * `Err(ToolError::ChecksumMismatch)` - If checksum doesn't match
/// * `Err(ToolError)` - If file cannot be read or hashed
#[instrument(skip(wasm_path, expected), fields(path = ?wasm_path))]
pub fn verify_wasm_checksum(wasm_path: &Path, expected: &str) -> Result<()> {
    let actual = compute_wasm_checksum(wasm_path)?;

    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(ToolError::ChecksumMismatch {
            expected: expected.to_lowercase(),
            actual,
        })
    }
}

// =============================================================================
// Tests// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const VALID_MANIFEST: &str = r#"
[tool]
name = "web-search"
version = "1.2.3"
description = "Search the web"
author = "Test Team"
license = "MIT"

[[capabilities_required]]
type = "network"
allowed_hosts = ["api.example.com"]

[resource_limits]
max_memory_mb = 128
max_fuel = 20000000
timeout_ms = 10000
max_output_bytes = 2097152

[input_schema]
type = "object"
required = ["query"]
[input_schema.properties.query]
type = "string"

[output_schema]
type = "object"
[output_schema.properties.results]
type = "array"
"#;

    #[test]
    fn test_parse_valid_manifest() {
        let result = parse_manifest(VALID_MANIFEST, None);
        assert!(result.is_ok());

        let manifest = result.unwrap();
        assert_eq!(manifest.name, "web-search");
        assert_eq!(manifest.version, "1.2.3");
        assert_eq!(manifest.resource_limits.max_memory_mb, 128);
        assert!(!manifest.capabilities_required.is_empty());
    }

    #[test]    fn test_validate_name() {
        assert!(validate_name("valid-name").is_ok());
        assert!(validate_name("valid_name").is_ok());
        assert!(validate_name("valid123").is_ok());

        assert!(validate_name("").is_err());
        assert!(validate_name("invalid name").is_err());
        assert!(validate_name("invalid@name").is_err());
    }

    #[test]
    fn test_validate_version() {
        assert!(validate_version("1.0").is_ok());
        assert!(validate_version("1.2.3").is_ok());
        assert!(validate_version("0.0.1-beta").is_ok());

        assert!(validate_version("").is_err());
        assert!(validate_version("1").is_err());
        assert!(validate_version("a.b.c").is_err());
    }

    #[test]
    fn test_validate_resource_limits() {
        let valid = RawResourceLimits {
            max_memory_mb: 64,
            max_fuel: 1000,
            timeout_ms: 5000,
            max_output_bytes: 1024,
        };
        assert!(convert_resource_limits(valid).is_ok());

        let invalid = RawResourceLimits {
            max_memory_mb: 0,
            ..Default::default()
        };
        assert!(convert_resource_limits(invalid).is_err());
    }

    #[test]
    fn test_validate_json_schema() {
        // Valid schema
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });
        assert!(validate_json_schema(&schema, "test").is_ok());

        // Empty schema (allows anything)        assert!(validate_json_schema(&serde_json::Value::Null, "test").is_ok());
        assert!(validate_json_schema(&serde_json::json!({}), "test").is_ok());

        // Invalid schema
        let bad = serde_json::json!({
            "type": "not_a_real_type"
        });
        assert!(validate_json_schema(&bad, "test").is_err());
    }

    #[test]
    fn test_load_from_file() {
        let tmp = TempDir::new().unwrap();
        let manifest_path = tmp.path().join("test.toml");
        fs::write(&manifest_path, VALID_MANIFEST).unwrap();

        let result = load_manifest_from_file(&manifest_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name, "web-search");
    }

    #[test]
    fn test_load_alongside_wasm() {
        let tmp = TempDir::new().unwrap();
        let wasm_path = tmp.path().join("my-tool.wasm");
        let manifest_path = tmp.path().join("my-tool.toml");

        // Create dummy WASM file
        fs::write(&wasm_path, b"\0asm").unwrap();
        // Create manifest
        fs::write(&manifest_path, VALID_MANIFEST).unwrap();

        let result = load_manifest_alongside_wasm(&wasm_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_checksum() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("test.wasm");
        fs::write(&file, b"hello world").unwrap();

        let checksum = compute_wasm_checksum(&file).unwrap();
        // SHA-256 of "hello world"
        assert_eq!(
            checksum,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }
    #[test]
    fn test_verify_checksum() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("test.wasm");
        fs::write(&file, b"hello world").unwrap();

        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert!(verify_wasm_checksum(&file, expected).is_ok());
        assert!(verify_wasm_checksum(&file, "wrong").is_err());
    }

    #[test]
    fn test_capability_conversion() {
        let raw = RawCapability::Network {
            allowed_hosts: vec!["example.com".into()],
        };
        let converted = convert_capability(raw).unwrap();
        match converted {
            ToolCapabilityRequirement::NetworkAccess { allowed_hosts } => {
                assert_eq!(allowed_hosts, vec!["example.com"]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_manifest_id() {
        let manifest = parse_manifest(VALID_MANIFEST, None).unwrap();
        let id = manifest.id();
        assert_eq!(id.name(), "web-search");
        assert_eq!(id.version(), "1.2.3");
        assert_eq!(id.to_string(), "web-search@1.2.3");
    }
}
