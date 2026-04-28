use thiserror::Error;
use uuid::Uuid;

/// Top-level error type for the Nexus AI agent runtime.
///
/// All subsystem errors ultimately convert into this enum, providing
/// a unified error handling interface across the entire system.
/// Each variant includes a descriptive message suitable for logging
/// and user-facing error reporting.
#[derive(Debug, Error)]
pub enum NexusError {
    /// Kernel subsystem error: agent lifecycle, scheduling, or supervision failure.
    #[error("kernel error: {0}")]
    KernelError(String),

    /// Memory subsystem error: L1-L4 storage operation failure.
    #[error("memory error: {0}")]
    MemoryError(String),

    /// Tool execution error: WASM sandbox, capability, or runtime failure.
    #[error("tool error: {0}")]
    ToolError(String),

    /// Model routing error: provider selection, API call, or response parsing failure.
    #[error("router error: {0}")]
    RouterError(String),

    /// Mesh networking error: peer discovery, message routing, or CRDT sync failure.
    #[error("mesh error: {0}")]
    MeshError(String),

    /// Workflow engine error: DAG execution, branching, or checkpoint failure.
    #[error("flow error: {0}")]
    FlowError(String),

    /// Observability error: tracing, metrics, or logging subsystem failure.
    #[error("observability error: {0}")]
    ObsError(String),

    /// Agent attempted to use a capability not granted in its security context.
    #[error("capability denied: agent '{agent_id}' requested '{capability}'")]
    CapabilityDenied {
        /// The UUID of the agent that was denied.
        agent_id: String,
        /// The capability that was requested but not granted.
        capability: String,
    },

    /// Lookup failed for an agent with the given UUID.
    #[error("agent not found: {0}")]    AgentNotFound(Uuid),

    /// Tool lookup failed: no module registered with the given name.
    #[error("tool not found in registry: '{0}'")]
    ToolNotFound(String),

    /// Model provider endpoint unreachable, unauthenticated, or rate-limited.
    #[error("model provider unavailable: {0}")]
    ProviderUnavailable(String),

    /// Tool input or output failed JSON schema validation.
    #[error("schema validation failed for tool '{tool}': field '{field}' - {message}")]
    SchemaValidation {
        /// The name of the tool whose schema failed validation.
        tool: String,
        /// The specific field that failed validation.
        field: String,
        /// Human-readable description of the validation failure.
        message: String,
    },

    /// Operation exceeded its allocated time budget.
    #[error("timeout: operation '{operation}' exceeded {duration_ms}ms")]
    Timeout {
        /// The name or description of the operation that timed out.
        operation: String,
        /// The timeout duration in milliseconds.
        duration_ms: u64,
    },

    /// JSON serialization or deserialization failure.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Filesystem or I/O operation failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration parsing, validation, or loading failure.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// Runtime is shutting down; operation cannot complete.
    #[error("runtime is shutting down")]
    ShuttingDown,

    /// Unexpected internal error indicating a potential bug in Nexus itself.
    #[error("internal error: {0}")]
    Internal(String),
}

/// A specialized Result type for Nexus operations.
pub type Result<T> = std::result::Result<T, NexusError>;

impl From<anyhow::Error> for NexusError {
    fn from(err: anyhow::Error) -> Self {
        NexusError::Internal(err.to_string())
    }
}

impl NexusError {
    /// Returns `true` if this error represents a transient condition that
    /// may succeed if retried, `false` for permanent failures.
    ///
    /// # Transient (retryable)
    /// - [`ProviderUnavailable`]: Network blip, rate limit, temporary outage
    /// - [`Timeout`]: Operation may complete on retry with more time
    ///
    /// # Permanent (non-retryable)
    /// - [`CapabilityDenied`], [`AgentNotFound`], [`SchemaValidation`]: 
    ///   Logic or configuration errors that won't resolve without intervention
    /// - All other variants indicate structural failures requiring debugging
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            NexusError::ProviderUnavailable(_) | NexusError::Timeout { .. }
        )
    }

    /// Returns a short, machine-readable error code string for this variant.
    ///
    /// Useful for:
    /// - Structured logging and metrics aggregation
    /// - API error response bodies (e.g., `{"code": "CAPABILITY_DENIED", ...}`)
    /// - Alerting rules and dashboards
    /// - Client-side error handling logic
    pub fn error_code(&self) -> &'static str {
        match self {
            NexusError::KernelError(_) => "KERNEL_ERROR",
            NexusError::MemoryError(_) => "MEMORY_ERROR",
            NexusError::ToolError(_) => "TOOL_ERROR",
            NexusError::RouterError(_) => "ROUTER_ERROR",
            NexusError::MeshError(_) => "MESH_ERROR",
            NexusError::FlowError(_) => "FLOW_ERROR",
            NexusError::ObsError(_) => "OBS_ERROR",
            NexusError::CapabilityDenied { .. } => "CAPABILITY_DENIED",
            NexusError::AgentNotFound(_) => "AGENT_NOT_FOUND",
            NexusError::ToolNotFound(_) => "TOOL_NOT_FOUND",
            NexusError::ProviderUnavailable(_) => "PROVIDER_UNAVAILABLE",
            NexusError::SchemaValidation { .. } => "SCHEMA_VALIDATION",
            NexusError::Timeout { .. } => "TIMEOUT",
            NexusError::Serialization(_) => "SERIALIZATION",
            NexusError::Io(_) => "IO_ERROR",            NexusError::Configuration(_) => "CONFIGURATION",
            NexusError::ShuttingDown => "SHUTTING_DOWN",
            NexusError::Internal(_) => "INTERNAL_ERROR",
        }
    }
}
