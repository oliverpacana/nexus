use thiserror::Error;
use uuid::Uuid;

use nexus_proto::NexusError;

/// Type alias for kernel-specific results.
pub type Result<T> = std::result::Result<T, KernelError>;

/// Errors originating from the `nexus-kernel` agent process manager.
/// Covers lifecycle, scheduling, supervision, and capability enforcement failures.
#[derive(Debug, Error)]
pub enum KernelError {
    /// Requested agent does not exist in the registry.
    #[error("agent not found: {0}")]
    AgentNotFound(Uuid),

    /// Attempted to spawn an agent with a UUID that is already registered.
    #[error("agent already exists: {0}")]
    AgentAlreadyExists(Uuid),

    /// Agent attempted to use a capability not declared in its security manifest.
    #[error("capability denied: agent '{agent_id}' requested '{capability}'")]
    CapabilityDenied {
        agent_id: Uuid,
        capability: String,
    },

    /// Referenced supervisor node does not exist in the hierarchy.
    #[error("supervisor not found: '{0}'")]
    SupervisorNotFound(String),

    /// Kernel scheduler cannot accept new agents due to resource limits.
    #[error("scheduler at capacity: max agents limit reached ({max_agents})")]
    SchedulerAtCapacity { max_agents: usize },

    /// Agent tokio task panicked, propagating the panic message.
    #[error("agent panicked: agent '{agent_id}' - {message}")]
    AgentPanicked {
        agent_id: Uuid,
        message: String,
    },

    /// Agent attempted an invalid lifecycle state transition.
    #[error("invalid state transition: agent '{agent_id}' cannot transition from '{from}' to '{to}'")]
    InvalidStateTransition {
        from: String,
        to: String,
        agent_id: Uuid,
    },

    /// Internal async channel was dropped or closed prematurely.
    #[error("channel closed: {0}")]
    ChannelClosed(String),

    /// Runtime is in the process of shutting down; new operations are rejected.
    #[error("kernel is shutting down")]
    ShuttingDown,

    /// Unexpected internal bug or invariant violation within the kernel.
    #[error("internal kernel error: {0}")]
    Internal(String),
}

impl From<NexusError> for KernelError {
    fn from(err: NexusError) -> Self {
        match err {
            NexusError::AgentNotFound(id) => KernelError::AgentNotFound(id),
            NexusError::CapabilityDenied { agent_id, capability } => {
                KernelError::CapabilityDenied {
                    agent_id: Uuid::parse_str(&agent_id).unwrap_or(Uuid::nil()),
                    capability,
                }
            }
            NexusError::ShuttingDown => KernelError::ShuttingDown,
            NexusError::Internal(msg) => KernelError::Internal(msg),
            _ => KernelError::Internal(err.to_string()),
        }
    }
}

impl From<KernelError> for NexusError {
    fn from(err: KernelError) -> Self {
        match err {
            KernelError::AgentNotFound(id) => NexusError::AgentNotFound(id),
            KernelError::CapabilityDenied { agent_id, capability } => NexusError::CapabilityDenied {
                agent_id: agent_id.to_string(),
                capability,
            },
            KernelError::ShuttingDown => NexusError::ShuttingDown,
            KernelError::Internal(msg) => NexusError::Internal(msg),
            _ => NexusError::KernelError(err.to_string()),
        }
    }
}

impl KernelError {
    /// Returns `true` if this error represents a transient condition that
    /// may succeed if retried (typically after backoff), `false` otherwise.
    ///
    /// # Retryable
    /// - `SchedulerAtCapacity`: Indicates temporary resource contention; 
    ///   retrying after exponential backoff usually succeeds.
    ///
    /// # Non-Retryable
    /// - All other variants represent structural, logical, or terminal failures
    ///   that require intervention or workflow rerouting rather than simple retry.
    pub fn is_retryable(&self) -> bool {
        matches!(self, KernelError::SchedulerAtCapacity { .. })
    }
}
