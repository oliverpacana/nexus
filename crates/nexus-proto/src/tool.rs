use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::agent::AgentId;

// =============================================================================
// Tool Identification
// =============================================================================

/// Unique identifier for a tool in the format `name@version`.
/// Newtype wrapper ensuring consistent parsing and display.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolId(String);

impl ToolId {
    /// Constructs a `ToolId` from a name and semantic version.
    /// Format: `"name@version"` (e.g., `"web-search@2.1.0"`)
    pub fn new(name: &str, version: &str) -> Self {
        Self(format!("{}@{}", name, version))
    }

    /// Returns the tool name portion (before `@`).
    pub fn name(&self) -> &str {
        self.0
            .split_once('@')
            .map(|(name, _)| name)
            .unwrap_or(&self.0)
    }

    /// Returns the version portion (after `@`), or empty string if malformed.
    pub fn version(&self) -> &str {
        self.0
            .split_once('@')
            .map(|(_, version)| version)
            .unwrap_or("")
    }
}

impl fmt::Display for ToolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ToolId {
    fn from(s: &str) -> Self {
        Self(s.to_string())    }
}

impl From<String> for ToolId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// =============================================================================
// Capability Requirements
// =============================================================================

/// Declares what host resources or permissions a tool requires to execute.
/// The runtime enforces these at load time and during execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "capability", rename_all = "snake_case")]
pub enum ToolCapabilityRequirement {
    /// Network access with optional host allowlist.
    /// Empty `allowed_hosts` vector means all hosts are permitted.
    NetworkAccess { allowed_hosts: Vec<String> },

    /// Read-only filesystem access to specified path prefixes.
    FilesystemRead { paths: Vec<String> },

    /// Read-write filesystem access to specified path prefixes.
    FilesystemWrite { paths: Vec<String> },

    /// Access to non-deterministic random number generation.
    RandomAccess,

    /// Access to wall-clock time and monotonic clocks.
    ClockAccess,

    /// Permission to emit structured log entries to host observability.
    LoggingAccess,
}

// =============================================================================
// Resource Limits
// =============================================================================

/// Hard limits imposed on WASM tool execution for safety and fairness.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum heap memory in megabytes the WASM module may allocate.
    pub max_memory_mb: u32,

    /// Maximum wasmtime "fuel" units (approximate CPU instruction budget).
    pub max_fuel: u64,
    /// Hard timeout in milliseconds before the tool is terminated.
    pub timeout_ms: u64,

    /// Maximum size in bytes of serialized output the tool may return.
    pub max_output_bytes: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 64,
            max_fuel: 10_000_000,
            timeout_ms: 5000,
            max_output_bytes: 1_048_576, // 1 MiB
        }
    }
}

// =============================================================================
// Tool Manifest
// =============================================================================

/// Complete metadata and specification for a WASM tool plugin.
/// This is the contract between tool authors and the Nexus runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolManifest {
    /// Human-readable tool name (e.g., `"web-search"`).
    pub name: String,

    /// Semantic version string (e.g., `"2.1.0"`).
    pub version: String,

    /// Description of the tool's purpose and behavior.
    pub description: String,

    /// Optional author name or organization.
    pub author: Option<String>,

    /// Optional SPDX license identifier (e.g., `"MIT"`, `"Apache-2.0"`).
    pub license: Option<String>,

    /// List of host capabilities this tool requires to function.
    pub capabilities_required: Vec<ToolCapabilityRequirement>,

    /// Resource constraints enforced by the WASM sandbox.
    pub resource_limits: ResourceLimits,

    /// JSON Schema defining the expected structure of tool input arguments.
    pub input_schema: serde_json::Value,
    /// JSON Schema defining the structure of successful tool output.
    pub output_schema: serde_json::Value,

    /// Optional filesystem path to the compiled `.wasm` binary.
    /// Populated at load time by the tool registry.
    pub wasm_path: Option<String>,

    /// Optional SHA-256 hex digest of the WASM binary for integrity verification.
    pub checksum_sha256: Option<String>,
}

impl ToolManifest {
    /// Returns the canonical `ToolId` for this manifest.
    pub fn id(&self) -> ToolId {
        ToolId::new(&self.name, &self.version)
    }
}

// =============================================================================
// Tool Invocation Types
// =============================================================================

/// A request to execute a specific tool with given arguments.
/// Generated by an agent and validated by the kernel before sandbox execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique correlation ID for this invocation (matches model tool_call_id).
    pub id: Uuid,

    /// Name of the tool to invoke (version resolved by registry).
    pub tool_name: String,

    /// Arguments to pass to the tool, validated against `input_schema`.
    pub arguments: serde_json::Value,

    /// ID of the agent requesting this tool execution.
    pub agent_id: AgentId,

    /// Trace ID for observability correlation across subsystems.
    pub trace_id: Uuid,
}

/// The result of executing a `ToolCall`, returned to the requesting agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Matches the `ToolCall.id` that produced this result.
    pub call_id: Uuid,

    /// Name of the tool that was executed.    pub tool_name: String,

    /// Output payload, validated against `output_schema` if `is_error` is false.
    pub output: serde_json::Value,

    /// Whether this result represents an error condition.
    pub is_error: bool,

    /// Human-readable error message if `is_error` is true.
    pub error_message: Option<String>,

    /// Wall-clock execution time in milliseconds.
    pub execution_time_ms: u64,

    /// Wasmtime fuel units consumed during execution.
    pub fuel_consumed: u64,

    /// Peak heap memory usage in megabytes during execution.
    pub memory_peak_mb: u32,
}

// =============================================================================
// Registry Types
// =============================================================================

/// Entry stored in the tool registry after a WASM module is loaded.
/// Contains both metadata and the compiled binary for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRegistryEntry {
    /// The validated manifest describing the tool.
    pub manifest: ToolManifest,

    /// Raw bytes of the compiled WASM module, ready for `wasmtime` instantiation.
    pub wasm_bytes: Vec<u8>,

    /// Timestamp when this tool was loaded into the registry.
    pub loaded_at: DateTime<Utc>,
}
