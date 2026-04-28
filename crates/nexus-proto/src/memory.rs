use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::agent::AgentId;

// =============================================================================
// Memory Tier & Scope
// =============================================================================

/// The four-tier memory hierarchy in Nexus, modeled after CPU cache levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    /// L1: Working memory — in-process, sub-microsecond access, per-agent task context.
    Working,
    /// L2: Episodic memory — embedded SQLite, millisecond access, chronological event log.
    Episodic,
    /// L3: Semantic memory — vector index, similarity search across agents.
    Semantic,
    /// L4: Procedural memory — structured knowledge graph, versioned global facts.
    Procedural,
}

impl fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryTier::Working => write!(f, "working"),
            MemoryTier::Episodic => write!(f, "episodic"),
            MemoryTier::Semantic => write!(f, "semantic"),
            MemoryTier::Procedural => write!(f, "procedural"),
        }
    }
}

impl MemoryTier {
    /// Returns the typical latency class for this tier (for documentation/scheduling).
    pub fn latency_class(&self) -> &'static str {
        match self {
            MemoryTier::Working => "microseconds",
            MemoryTier::Episodic => "milliseconds",
            MemoryTier::Semantic => "milliseconds",
            MemoryTier::Procedural => "milliseconds",
        }
    }
}

/// Access scope controlling which agents can read/write a memory entry.#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// Only the owning agent can access this entry.
    Private,
    /// Agents under the same supervisor tree can access this entry.
    Group,
    /// Any agent in the system can access this entry.
    Global,
}

impl fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryScope::Private => write!(f, "private"),
            MemoryScope::Group => write!(f, "group"),
            MemoryScope::Global => write!(f, "global"),
        }
    }
}

// =============================================================================
// Memory Key
// =============================================================================

/// Structured key for addressing memory entries: `namespace::key`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryKey {
    pub namespace: String,
    pub key: String,
}

impl MemoryKey {
    /// Constructs a key from namespace and key components.
    pub fn new(namespace: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            key: key.into(),
        }
    }

    /// Constructs a key scoped to a specific agent (namespace = agent ID).
    pub fn agent_key(agent_id: AgentId, key: impl Into<String>) -> Self {
        Self {
            namespace: format!("agent:{}", agent_id),
            key: key.into(),
        }
    }

    /// Constructs a globally-scoped key.    pub fn global_key(key: impl Into<String>) -> Self {
        Self {
            namespace: "global".to_string(),
            key: key.into(),
        }
    }

impl fmt::Display for MemoryKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.namespace, self.key)
    }
}

impl FromStr for MemoryKey {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.splitn(2, "::").collect();
        if parts.len() != 2 {
            return Err(format!("invalid MemoryKey format: expected 'namespace::key', got '{}'", s));
        }
        Ok(Self {
            namespace: parts[0].to_string(),
            key: parts[1].to_string(),
        })
    }
}

// =============================================================================
// Embedding Vector
// =============================================================================

/// A dense float32 embedding vector for semantic memory (L3).
/// Wrapper around `Vec<f32>` with similarity computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingVector(Vec<f32>);

impl EmbeddingVector {
    /// Creates a zero-initialized vector with the given dimensionality.
    pub fn new(dims: usize) -> Self {
        Self(vec![0.0; dims])
    }

    /// Wraps an existing `Vec<f32>` as an embedding vector.
    pub fn from_vec(v: Vec<f32>) -> Self {
        Self(v)
    }
    /// Returns the dimensionality of this embedding.
    pub fn dims(&self) -> usize {
        self.0.len()
    }

    /// Returns a slice reference to the underlying float values.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Computes cosine similarity between this vector and another.
    ///
    /// Formula: dot(a, b) / (||a|| * ||b||)
    /// Returns 0.0 if either vector has zero magnitude (undefined similarity).
    pub fn cosine_similarity(&self, other: &EmbeddingVector) -> f32 {
        if self.0.len() != other.0.len() {
            return 0.0; // Dimension mismatch: treat as orthogonal
        }

        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();

        let norm_a: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

impl From<Vec<f32>> for EmbeddingVector {
    fn from(v: Vec<f32>) -> Self {
        Self::from_vec(v)
    }
}

impl AsRef<[f32]> for EmbeddingVector {
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

// =============================================================================
// Memory Entry
// =============================================================================

/// A generic entry in the Nexus memory hierarchy.
/// Fields are populated based on tier: L3 uses `embedding`, L2 uses `event_type`, etc.#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Structured address for this entry.
    pub key: MemoryKey,

    /// Which tier this entry resides in.
    pub tier: MemoryTier,

    /// Access scope controlling visibility.
    pub scope: MemoryScope,

    /// Agent ID that owns/created this entry.
    pub owner_id: AgentId,

    /// The actual stored value (JSON for flexibility).
    pub value: serde_json::Value,

    /// Optional embedding vector for semantic (L3) search.
    pub embedding: Option<EmbeddingVector>,

    /// Timestamp when this entry was first created.
    pub created_at: DateTime<Utc>,

    /// Timestamp of the last modification.
    pub updated_at: DateTime<Utc>,

    /// Optional expiration timestamp; entry should be evicted after this.
    pub expires_at: Option<DateTime<Utc>>,

    /// Optimistic locking version for concurrent updates.
    pub version: u64,

    /// Arbitrary tags for filtering and organization.
    pub tags: Vec<String>,
}

impl MemoryEntry {
    /// Returns `true` if this entry has expired based on `expires_at`.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Utc::now() > exp)
            .unwrap_or(false)
    }
}

// =============================================================================
// Semantic Search (L3)
// =============================================================================

/// Query parameters for vector similarity search in semantic memory.#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchQuery {
    /// The query embedding to compare against indexed vectors.
    pub query_embedding: EmbeddingVector,

    /// Optional original query text for logging/debugging.
    pub query_text: Option<String>,

    /// Maximum number of results to return.
    pub top_k: usize,

    /// Minimum cosine similarity threshold (0.0 to 1.0); results below are filtered.
    pub min_similarity: f32,

    /// Optional scope filter to restrict results by access level.
    pub scope_filter: Option<MemoryScope>,

    /// Optional owner filter to restrict results to a specific agent.
    pub owner_filter: Option<AgentId>,

    /// Optional tags that results must contain (AND logic).
    pub tag_filter: Vec<String>,
}

/// A single result from a semantic search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchResult {
    /// The matched memory entry.
    pub entry: MemoryEntry,

    /// Cosine similarity score between query and entry embedding.
    pub similarity: f32,

    /// Rank in the result set (1 = most similar).
    pub rank: usize,
}

// =============================================================================
// Episodic Memory (L2)
// =============================================================================

/// Categorizes the type of event logged in episodic memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodicEventType {
    AgentStarted,
    AgentFinished,
    ToolCalled,
    ToolResult,
    ModelRequest,    ModelResponse,
    MemoryWrite,
    MemoryRead,
    WorkflowStep,
    CustomEvent(String),
}

/// A single event in an agent's episodic log (L2 memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEvent {
    /// Unique identifier for this event.
    pub id: Uuid,

    /// Agent that generated this event.
    pub agent_id: AgentId,

    /// Categorization of the event type.
    pub event_type: EpisodicEventType,

    /// Structured payload data for the event.
    pub payload: serde_json::Value,

    /// Timestamp when the event occurred.
    pub timestamp: DateTime<Utc>,

    /// Session ID grouping events from a single agent execution run.
    pub session_id: Uuid,

    /// Monotonic sequence number within the session for ordering.
    pub sequence: u64,
}

impl EpisodicEvent {
    /// Constructs a new episodic event with auto-generated ID and timestamp.
    pub fn new(
        agent_id: AgentId,
        event_type: EpisodicEventType,
        payload: serde_json::Value,
        session_id: Uuid,
        sequence: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            agent_id,
            event_type,
            payload,
            timestamp: Utc::now(),
            session_id,
            sequence,
        }    }
}
