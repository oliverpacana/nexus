use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Unique identifier for an agent within the Nexus runtime.
/// Newtype wrapper around `uuid::Uuid` with serialization and lifecycle helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentId(Uuid);

impl AgentId {
    /// Generates a new random v4 UUID for an agent.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Returns a nil UUID (all zeros), useful for testing or sentinel values.
    pub fn nil() -> Self {
        Self(Uuid::nil())
    }

    /// Returns a reference to the underlying `Uuid`.
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Access level for memory tier capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryAccess {
    /// Read-only access to memory.
    Read,
    /// Write-only access to memory.
    Write,
    /// Both read and write access to memory.
    ReadWrite,
}

impl fmt::Display for MemoryAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryAccess::Read => write!(f, "read"),
            MemoryAccess::Write => write!(f, "write"),
            MemoryAccess::ReadWrite => write!(f, "read_write"),
        }
    }
}

/// The categorical type of an agent, defining its primary operational role.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    /// Agent specialized in gathering and synthesizing information.
    Research,
    /// Agent specialized in generating text or code.
    Writing,
    /// Agent specialized in reviewing and auditing code.
    CodeReview,
    /// Agent specialized in data analysis and reasoning.
    Analysis,
    /// Agent specialized in breaking down tasks and orchestrating workflows.
    Planning,
    /// A custom agent type with a specific name.
    Custom(String),
}

impl fmt::Display for AgentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentKind::Research => write!(f, "research"),
            AgentKind::Writing => write!(f, "writing"),
            AgentKind::CodeReview => write!(f, "code_review"),
            AgentKind::Analysis => write!(f, "analysis"),
            AgentKind::Planning => write!(f, "planning"),
            AgentKind::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

/// Scheduling priority for agent execution. Higher values receive more CPU share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentPriority {
    /// Highest priority; urgent tasks requiring immediate attention.
    Critical = 5,
    /// High priority; important tasks that should be prioritized.
    High = 4,
    /// Default priority for most tasks.
    Normal = 3,
    /// Low priority; non-urgent tasks that can wait.
    Low = 2,
    /// Lowest priority; background tasks performed when idle.
    Background = 1,
}

impl Default for AgentPriority {
    fn default() -> Self {
        AgentPriority::Normal
    }
}

impl AgentPriority {
    /// Returns the numeric weight used by the scheduler for proportional resource allocation.
    pub fn weight(&self) -> u32 {
        match self {
            AgentPriority::Critical => 5,
            AgentPriority::High => 4,
            AgentPriority::Normal => 3,
            AgentPriority::Low => 2,
            AgentPriority::Background => 1,
        }
    }
}

/// Represents the full lifecycle state of an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    /// Agent is created but not yet scheduled.
    Pending {
        /// Timestamp when the agent was created.
        created_at: DateTime<Utc>,
    },
    /// Agent is currently executing a task.
    Running {
        /// Timestamp when execution began.
        started_at: DateTime<Utc>,
        /// Optional ID of the task currently being executed.
        task_id: Option<String>,
    },
    /// Agent execution is paused.
    Suspended {
        /// Reason for suspension.
        reason: String,
        /// Timestamp when suspension occurred.
        suspended_at: DateTime<Utc>,
    },
    /// Agent completed its task successfully.
    Completed {
        /// Timestamp when the agent finished.
        finished_at: DateTime<Utc>,
        /// Whether the agent considers its execution successful.
        success: bool,
    },
    /// Agent encountered an error and stopped.
    Failed {
        /// Error message describing the failure.
        error: String,
        /// Timestamp when the failure occurred.
        failed_at: DateTime<Utc>,
        /// Number of times the agent has retried its current task.
        retries: u32,
    },
    /// Agent is in the process of being shut down.
    Terminating,
}

impl AgentStatus {
    /// Returns `true` if the agent has reached a terminal state (completed or failed).
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentStatus::Completed { .. } | AgentStatus::Failed { .. })
    }

    /// Returns `true` if the agent is currently executing.
    pub fn is_active(&self) -> bool {
        matches!(self, AgentStatus::Running { .. })
    }

    /// Returns the timestamp when the agent transitioned to the `Running` state, if applicable.
    pub fn started_at(&self) -> Option<DateTime<Utc>> {
        match self {
            AgentStatus::Running { started_at, .. } => Some(*started_at),
            _ => None,
        }
    }
}

/// Declares the set of capabilities an agent is authorized to use.
/// Internally backed by a `HashSet<String>` for O(1) lookup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentCapabilities(HashSet<String>);

impl AgentCapabilities {
    /// Creates a new empty capability set.
    pub fn new() -> Self {
        Self(HashSet::new())
    }

    /// Adds a tool capability.
    pub fn with_tool(self, tool_name: impl Into<String>) -> Self {
        let mut set = self.0;
        set.insert(format!("tool:{}", tool_name.into()));
        Self(set)
    }

    /// Adds a memory access capability.
    pub fn with_memory(self, scope: MemoryAccess) -> Self {
        let mut set = self.0;
        set.insert(format!("memory:{}", scope));
        Self(set)
    }

    /// Adds a model routing capability using a glob or exact pattern.
    pub fn with_model(self, pattern: impl Into<String>) -> Self {
        let mut set = self.0;
        set.insert(format!("model:{}", pattern.into()));
        Self(set)
    }

    /// Grants permission to delegate work across the P2P mesh.
    pub fn with_mesh_delegate(self) -> Self {
        let mut set = self.0;
        set.insert("mesh:delegate".to_string());
        Self(set)
    }

    /// Checks if the agent possesses the specified capability string.
    pub fn has(&self, capability: &str) -> bool {
        self.0.contains(capability)
    }
    /// Returns an iterator over all declared capabilities.
    pub fn all(&self) -> impl Iterator<Item = &String> {
        self.0.iter()
    }
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable metadata snapshot representing an agent's configuration and state.
/// Used for cross-subsystem communication, supervision, and observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AgentMeta {
    /// Unique identifier for this agent instance.
    pub id: AgentId,

    /// The categorical role this agent fulfills.
    pub kind: AgentKind,

    /// Human-readable name for logging, debugging, and UI display.
    pub name: String,

    /// Scheduler priority determining resource allocation.
    pub priority: AgentPriority,

    /// Current lifecycle state.
    pub status: AgentStatus,

    /// Declared security capabilities and permissions.
    pub capabilities: AgentCapabilities,

    /// Timestamp when the agent was initially spawned.
    pub created_at: DateTime<Utc>,

    /// Optional parent agent ID for hierarchical supervision trees.
    pub parent_id: Option<AgentId>,

    /// Arbitrary key-value pairs for tagging, routing hints, or user annotations.
    pub tags: HashMap<String, String>,
}
