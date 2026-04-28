use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::agent::{AgentId, AgentPriority, AgentStatus};
use crate::error::NexusError;
use crate::memory::MemoryKey;

// =============================================================================
// Message Identification
// =============================================================================

/// Unique identifier for a message in the Nexus messaging system.
/// Newtype wrapper around `uuid::Uuid` for type safety and serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MessageId(Uuid);

impl MessageId {
    /// Generates a new random v4 UUID for a message.
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

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }}

// =============================================================================
// Control Signals
// =============================================================================

/// Control signals for agent lifecycle management.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "signal", rename_all = "snake_case")]
pub enum ControlKind {
    /// Suspend agent execution; preserve state for later resume.
    Suspend,
    /// Resume a previously suspended agent.
    Resume,
    /// Terminate agent immediately with optional reason.
    Kill { reason: String },
    /// Dynamically adjust agent scheduling priority.
    SetPriority(AgentPriority),
}

// =============================================================================
// Message Kinds
// =============================================================================

/// The semantic type of a message, defining its payload structure and handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MessageKind {
    /// Assign a workflow step to an agent for execution.
    TaskAssignment {
        workflow_id: Uuid,
        step_id: String,
    },

    /// Report completion of an assigned workflow step.
    TaskResult {
        workflow_id: Uuid,
        step_id: String,
        success: bool,
    },

    /// Request delegation of work to another agent with specific capability.
    DelegateRequest {
        capability: String,
        payload: serde_json::Value,
    },

    /// Response to a delegation request.
    DelegateResponse {
        request_id: MessageId,        result: serde_json::Value,
        error: Option<String>,
    },

    /// Broadcast a memory update to subscribed agents.
    MemoryBroadcast {
        key: MemoryKey,
        value: serde_json::Value,
    },

    /// Periodic liveness signal with current agent status.
    Heartbeat {
        agent_id: AgentId,
        status: AgentStatus,
    },

    /// Lifecycle control signal for agent management.
    ControlSignal(ControlKind),

    /// Escape hatch for custom message types not covered by variants.
    Custom {
        kind_name: String,
        payload: serde_json::Value,
    },
}

// =============================================================================
// Message Envelope
// =============================================================================

/// Outer wrapper for all inter-agent messages in the Nexus runtime.
/// Provides routing metadata, tracing correlation, and delivery semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    /// Unique identifier for this message instance.
    pub id: MessageId,

    /// Agent ID of the message sender.
    pub from: AgentId,

    /// Agent ID of the intended recipient; `nil()` for broadcast messages.
    pub to: AgentId,

    /// The semantic payload and type of this message.
    pub kind: MessageKind,

    /// Distributed trace ID for observability correlation across subsystems.
    pub trace_id: Uuid,

    /// Timestamp when the message was sent.    pub sent_at: DateTime<Utc>,

    /// Optional time-to-live in milliseconds; message should be discarded after expiry.
    pub ttl_ms: Option<u64>,

    /// Optional ID of the message this is replying to, for request/response correlation.
    pub reply_to: Option<MessageId>,
}

impl Envelope {
    /// Constructs a new envelope for point-to-point messaging.
    pub fn new(from: AgentId, to: AgentId, kind: MessageKind) -> Self {
        Self {
            id: MessageId::new(),
            from,
            to,
            kind,
            trace_id: Uuid::new_v4(),
            sent_at: Utc::now(),
            ttl_ms: None,
            reply_to: None,
        }
    }

    /// Constructs a broadcast envelope (to = nil) for mesh-wide dissemination.
    pub fn broadcast(from: AgentId, kind: MessageKind) -> Self {
        Self {
            id: MessageId::new(),
            from,
            to: AgentId::nil(),
            kind,
            trace_id: Uuid::new_v4(),
            sent_at: Utc::now(),
            ttl_ms: None,
            reply_to: None,
        }
    }

    /// Returns `true` if this message has exceeded its TTL and should be discarded.
    pub fn is_expired(&self) -> bool {
        self.ttl_ms
            .map(|ttl| {
                let elapsed = Utc::now()
                    .signed_duration_since(self.sent_at)
                    .num_milliseconds();
                elapsed > ttl as i64
            })
            .unwrap_or(false)
    }
    /// Creates a reply envelope with this message's ID as `reply_to` and swapped sender/recipient.
    pub fn reply(&self, kind: MessageKind) -> Self {
        Self {
            id: MessageId::new(),
            from: self.to,
            to: self.from,
            kind,
            trace_id: self.trace_id,
            sent_at: Utc::now(),
            ttl_ms: self.ttl_ms,
            reply_to: Some(self.id),
        }
    }

    /// Returns `true` if this envelope is a broadcast (recipient is nil).
    pub fn is_broadcast(&self) -> bool {
        self.to == AgentId::nil()
    }
}

// =============================================================================
// Typed Async Channel Abstraction
// =============================================================================

/// Transmitter half of a typed async message channel.
pub struct ChannelTx<T> {
    inner: mpsc::Sender<T>,
}

impl<T> Clone for ChannelTx<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> ChannelTx<T> {
    /// Asynchronously sends a message, returning an error if the channel is closed.
    pub async fn send(&self, msg: T) -> Result<(), NexusError> {
        self.inner
            .send(msg)
            .await
            .map_err(|_| NexusError::Internal("channel closed".into()))
    }

    /// Attempts to send a message without awaiting; returns error if channel is full or closed.
    pub fn try_send(&self, msg: T) -> Result<(), NexusError> {
        self.inner
            .try_send(msg)            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => {
                    NexusError::Timeout {
                        operation: "channel send".into(),
                        duration_ms: 0,
                    }
                }
                mpsc::error::TrySendError::Closed(_) => {
                    NexusError::Internal("channel closed".into())
                }
            })
    }

    /// Returns `true` if the receiver half has been dropped.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

/// Receiver half of a typed async message channel.
pub struct ChannelRx<T> {
    inner: mpsc::Receiver<T>,
}

impl<T> ChannelRx<T> {
    /// Asynchronously receives the next message, returning `None` if the channel is closed.
    pub async fn recv(&mut self) -> Option<T> {
        self.inner.recv().await
    }

    /// Receives with a timeout; returns `Ok(None)` on timeout, `Err` if channel closed.
    pub async fn recv_timeout(&mut self, ms: u64) -> Result<Option<T>, NexusError> {
        match timeout(Duration::from_millis(ms), self.inner.recv()).await {
            Ok(msg) => Ok(msg),
            Err(_) => Err(NexusError::Timeout {
                operation: "channel recv".into(),
                duration_ms: ms,
            }),
        }
    }
}

/// Typed async channel abstraction built on `tokio::sync::mpsc`.
pub struct Channel<T>;

impl<T: Send + 'static> Channel<T> {
    /// Creates a bounded channel with the given capacity.
    /// Returns transmitter and receiver halves.
    pub fn bounded(capacity: usize) -> (ChannelTx<T>, ChannelRx<T>) {
        let (tx, rx) = mpsc::channel(capacity);        (ChannelTx { inner: tx }, ChannelRx { inner: rx })
    }
}
