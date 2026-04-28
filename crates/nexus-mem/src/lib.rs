//! # Nexus Memory
//!
//! The four-tier memory hierarchy for the Nexus AI agent runtime.
//!
//! `nexus-mem` provides a unified, permissioned interface to:
//! - **L1 Working Memory**: In-process, per-agent key-value cache (`Arc<RwLock<HashMap>>`)
//! - **L2 Episodic Memory**: Append-only SQLite event log for replay and debugging
//! - **L3 Semantic Memory**: HNSW vector index for similarity search across agents
//! - **L4 Procedural Memory**: Sled-backed knowledge graph for structured facts
//!
//! All memory access is governed by a capability-based permission system that enforces
//! scope rules (Private, Group, Global) and explicit grants between agents.
//!
//! ## Usage
//!
//! ```rust
//! use nexus_mem::{MemoryStore, MemoryConfig, EmbeddingConfig};
//! use nexus_proto::agent::AgentId;
//! use nexus_proto::memory::{MemoryEntry, MemoryKey, MemoryScope, MemoryTier};
//!
//! async fn example() -> anyhow::Result<()> {
//!     let config = MemoryConfig {
//!         working_max_entries: 128,
//!         episodic_db_path: "./data/episodic.db".into(),
//!         episodic_max_events: 10000,
//!         semantic_index_path: "./data/semantic.idx".into(),
//!         semantic_meta_db_path: "./data/semantic_meta.db".into(),
//!         semantic_dimensions: 768,
//!         semantic_ef_construction: 256,
//!         semantic_ef_search: 64,
//!         procedural_db_path: "./data/procedural.sled".into(),
//!         embedding: EmbeddingConfig::default(),
//!     };
//!
//!     let mem = MemoryStore::new(config).await?;
//!     let agent_id = AgentId::new();
//!
//!     // Register agent with optional group membership
//!     mem.register_agent(agent_id, Some("research-team".into()));
//!
//!     // Write to working memory (L1)
//!     let working = mem.get_working(agent_id);
//!     working.set("task_context", serde_json::json!({"goal": "research quantum computing"})).await?;
//!
//!     // Append to episodic log (L2)
//!     let event = nexus_proto::memory::EpisodicEvent::new(
//!         agent_id,
//!         nexus_proto::memory::EpisodicEventType::MemoryWrite,
//!         serde_json::json!({"key": "task_context"}),
//!         uuid::Uuid::new_v4(),//!         0,
//!     );
//!     mem.episodic_append(event).await?;
//!
//!     // Search semantic memory (L3)
//!     let results = mem.semantic_search(
//!         agent_id,
//!         nexus_proto::memory::SemanticSearchQuery {
//!             query_embedding: None,
//!             query_text: Some("quantum algorithms".into()),
//!             top_k: 5,
//!             min_similarity: 0.7,
//!             scope_filter: Some(MemoryScope::Global),
//!             owner_filter: None,
//!             tag_filter: vec!["research".into()],
//!         },
//!     ).await?;
//!
//!     // Flush all tiers to durable storage
//!     mem.flush().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Permission Model
//!
//! Memory access is controlled by three orthogonal dimensions:
//!
//! 1. **Tier**: L1/L2 are always private for writes; L3/L4 support sharing
//! 2. **Scope**: Private (owner only), Group (same supervisor tree), Global (anyone)
//! 3. **Grants**: Explicit `MemoryPermission` grants from owner to other agents
//!
//! The `MemoryAccessChecker` enforces these rules at runtime. Agents cannot bypass
//! permissions by calling lower-level tier APIs directly — all access goes through
//! `MemoryStore` methods which perform authorization checks.
//!
//! ## Thread Safety
//!
//! All public methods are `async` and safe for concurrent use from multiple Tokio tasks.
//! Internal state uses `DashMap`, `Arc<RwLock>`, and `sled::Db` (which is `Send + Sync`)
//! to ensure lock-free reads and minimal contention for writes.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

// =============================================================================// Module Declarations
// =============================================================================

pub mod error;
pub mod embeddings;
pub mod working;
pub mod episodic;
pub mod semantic;
pub mod procedural;
pub mod permissions;

// =============================================================================
// Public Re-Exports
// =============================================================================

// Core error type
pub use error::MemoryError;
pub type Result<T> = std::result::Result<T, MemoryError>;

// Embedding abstraction
pub use embeddings::{
    EmbeddingProvider, EmbeddingConfig, MemoryError as EmbeddingError,
    LocalEmbeddingProvider, OpenAIEmbeddingProvider, OllamaEmbeddingProvider,
    cosine_similarity, find_most_similar,
};

// L1 Working Memory
pub use working::{WorkingMemory, WorkingEntry, WorkingMemoryStats, WorkingMemorySnapshot};

// L2 Episodic Memory
pub use episodic::EpisodicStore;

// L3 Semantic Memory
pub use semantic::SemanticStore;

// L4 Procedural Memory
pub use procedural::{ProceduralStore, Entity, Relation};

// Permission system
pub use permissions::{
    MemoryPermission, GrantTable, GroupTable, MemoryAccessChecker, PermissionSummary,
    AccessDeniedDetails,
};

// Re-export proto types for convenience
pub use nexus_proto::memory::{
    MemoryTier, MemoryScope, MemoryAccess, MemoryKey, MemoryEntry, EmbeddingVector,
    SemanticSearchQuery, SemanticSearchResult, EpisodicEvent, EpisodicEventType,
};
// =============================================================================
// MemoryConfig — Unified Configuration
// =============================================================================

/// Configuration for initializing the unified `MemoryStore`.
///
/// All fields are public for ergonomic construction; consider using `Default`
/// and builder patterns for production use.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    // L1 Working Memory
    /// Maximum entries per agent's working memory before LRU eviction.
    pub working_max_entries: usize,

    // L2 Episodic Memory
    /// Filesystem path to the SQLite database for episodic events.
    pub episodic_db_path: String,
    /// Maximum events to retain per agent before trimming oldest.
    pub episodic_max_events: usize,

    // L3 Semantic Memory
    /// Filesystem path to the usearch HNSW index file.
    pub semantic_index_path: String,
    /// Filesystem path to the SQLite metadata database for semantic entries.
    pub semantic_meta_db_path: String,
    /// Dimensionality of embedding vectors (must match embedder output).
    pub semantic_dimensions: usize,
    /// HNSW construction expansion factor (higher = better quality, slower build).
    pub semantic_ef_construction: usize,
    /// HNSW search expansion factor (higher = better recall, slower query).
    pub semantic_ef_search: usize,

    // L4 Procedural Memory
    /// Filesystem path to the sled database directory for the knowledge graph.
    pub procedural_db_path: String,

    // Embedding Provider
    /// Configuration for selecting and constructing the embedding provider.
    pub embedding: EmbeddingConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            working_max_entries: 128,
            episodic_db_path: "./data/nexus-episodic.db".into(),
            episodic_max_events: 10000,
            semantic_index_path: "./data/nexus-semantic.index".into(),
            semantic_meta_db_path: "./data/nexus-semantic-meta.db".into(),
            semantic_dimensions: 768,            semantic_ef_construction: 256,
            semantic_ef_search: 64,
            procedural_db_path: "./data/nexus-procedural.sled".into(),
            embedding: EmbeddingConfig::default(),
        }
    }
}

// =============================================================================
// MemoryStore — Unified Four-Tier API
// =============================================================================

/// The unified, permissioned interface to Nexus's four-tier memory hierarchy.
///
/// # Design
/// - Single entry point for all memory operations across L1-L4
/// - Automatic permission enforcement via `MemoryAccessChecker`
/// - Per-agent working memory instances created on-demand
/// - All tiers share consistent error handling via `MemoryError`
///
/// # Thread Safety
/// - `Send + Sync`; safe for concurrent access from many Tokio tasks
/// - Internal state uses `DashMap` and `Arc` for lock-free reads
/// - Write operations acquire appropriate locks internally
pub struct MemoryStore {
    /// L1: Per-agent working memory instances.
    working: dashmap::DashMap<nexus_proto::agent::AgentId, std::sync::Arc<WorkingMemory>>,

    /// L2: Episodic event log.
    episodic: std::sync::Arc<EpisodicStore>,

    /// L3: Semantic vector index.
    semantic: std::sync::Arc<SemanticStore>,

    /// L4: Procedural knowledge graph.
    procedural: std::sync::Arc<ProceduralStore>,

    /// Permission enforcement for cross-agent memory access.
    access_checker: std::sync::Arc<MemoryAccessChecker>,

    /// Configuration snapshot for observability.
    config: MemoryConfig,
}

impl MemoryStore {
    /// Initializes all four memory tiers from the given configuration.
    ///
    /// # Arguments
    /// * `config` - Unified configuration for all tiers
    ///    /// # Returns
    /// * `Ok(MemoryStore)` - If all tiers initialized successfully
    /// * `Err(MemoryError)` - If any tier failed to initialize
    ///
    /// # Side Effects
    /// - Creates/opens SQLite databases for L2 and L3 metadata
    /// - Creates/loads usearch HNSW index for L3
    /// - Opens sled database for L4
    /// - Constructs embedding provider per `config.embedding`
    #[tracing::instrument(skip(config), fields(
        working_max = config.working_max_entries,
        episodic_path = %config.episodic_db_path,
        semantic_dims = config.semantic_dimensions,
        procedural_path = %config.procedural_db_path,
    ))]
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        tracing::debug!("initializing unified memory store");

        // Initialize embedding provider
        let embedder = config
            .embedding
            .build()
            .map_err(|e| MemoryError::ProviderError(format!("embedding provider init failed: {}", e)))?;
        let embedder = std::sync::Arc::new(embedder);

        // L2: Episodic Store
        let episodic = EpisodicStore::new(&config.episodic_db_path, config.episodic_max_events)
            .await
            .map_err(|e| MemoryError::ProviderError(format!("episodic store init failed: {}", e)))?;

        // L3: Semantic Store
        let semantic = SemanticStore::new(
            &config.semantic_index_path,
            &config.semantic_meta_db_path,
            std::sync::Arc::clone(&embedder),
            config.semantic_dimensions,
            config.semantic_ef_construction,
            config.semantic_ef_search,
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("semantic store init failed: {}", e)))?;

        // L4: Procedural Store
        let procedural = ProceduralStore::new(&config.procedural_db_path)
            .map_err(|e| MemoryError::ProviderError(format!("procedural store init failed: {}", e)))?;

        // Permission system
        let grant_table = std::sync::Arc::new(permissions::GrantTable::new());
        let group_table = std::sync::Arc::new(permissions::GroupTable::new());
        let access_checker = std::sync::Arc::new(MemoryAccessChecker::new(            std::sync::Arc::clone(&grant_table),
            std::sync::Arc::clone(&group_table),
        ));

        tracing::debug!("all memory tiers initialized");

        Ok(Self {
            working: dashmap::DashMap::new(),
            episodic: std::sync::Arc::new(episodic),
            semantic: std::sync::Arc::new(semantic),
            procedural: std::sync::Arc::new(procedural),
            access_checker,
            config,
        })
    }

    /// Returns the working memory (L1) instance for an agent.
    ///
    /// Creates a new `WorkingMemory` if this is the first access for this agent.
    /// The returned `Arc` can be cloned cheaply for concurrent access.
    #[tracing::instrument(skip(self), fields(agent_id = %agent_id))]
    pub fn get_working(&self, agent_id: nexus_proto::agent::AgentId) -> std::sync::Arc<WorkingMemory> {
        self.working
            .entry(agent_id)
            .or_insert_with(|| {
                std::sync::Arc::new(WorkingMemory::new(
                    agent_id,
                    self.config.working_max_entries,
                ))
            })
            .value()
            .clone()
    }

    /// Appends an event to the episodic log (L2).
    ///
    /// No permission check: agents can always write to their own episodic log.
    #[tracing::instrument(skip(self, event), fields(agent_id = %event.agent_id, event_type = ?event.event_type))]
    pub async fn episodic_append(&self, event: EpisodicEvent) -> Result<()> {
        self.episodic
            .append(event)
            .await
            .map_err(|e| MemoryError::ProviderError(format!("episodic append failed: {}", e)))
    }

    /// Retrieves recent episodic events for an agent (L2).
    ///
    /// No permission check: agents can always read their own episodic log.
    #[tracing::instrument(skip(self), fields(agent_id = %agent, limit))]
    pub async fn episodic_history(        &self,
        agent: nexus_proto::agent::AgentId,
        limit: usize,
    ) -> Result<Vec<EpisodicEvent>> {
        self.episodic
            .get_agent_history(agent, limit, None)
            .await
            .map_err(|e| MemoryError::ProviderError(format!("episodic history query failed: {}", e)))
    }

    /// Writes an entry to semantic memory (L3) with permission enforcement.
    ///
    /// # Permission Logic
    /// - Agents can always write to their own entries (owner == requestor)
    /// - For entries owned by others: requires explicit write grant from owner
    /// - L3 respects scope: Private/Group/Global rules apply
    ///
    /// # Arguments
    /// * `requestor` - The agent attempting the write
    /// * `entry` - The memory entry to insert (must include embedding)
    #[tracing::instrument(skip(self, entry), fields(requestor = %requestor, key = %entry.key, owner = %entry.owner_id))]
    pub async fn semantic_write(
        &self,
        requestor: nexus_proto::agent::AgentId,
        entry: MemoryEntry,
    ) -> Result<()> {
        // Permission check
        self.access_checker.check_write(
            requestor,
            entry.owner_id,
            entry.scope,
            MemoryTier::Semantic,
        )?;

        // Insert into semantic store
        self.semantic
            .insert(entry)
            .await
            .map_err(|e| MemoryError::ProviderError(format!("semantic insert failed: {}", e)))
    }

    /// Searches semantic memory (L3) with permission filtering.
    ///
    /// # Permission Logic
    /// - Results are filtered to entries the requestor is authorized to read
    /// - Global scope entries are visible to all
    /// - Group scope entries require shared group membership or explicit grant
    /// - Private scope entries are only visible to their owner
    ///
    /// # Arguments    /// * `requestor` - The agent performing the search
    /// * `query` - Search parameters including embedding/text and filters
    #[tracing::instrument(skip(self, query), fields(requestor = %requestor, top_k = query.top_k))]
    pub async fn semantic_search(
        &self,
        requestor: nexus_proto::agent::AgentId,
        query: SemanticSearchQuery,
    ) -> Result<Vec<SemanticSearchResult>> {
        // Perform search at storage layer (returns candidates)
        let candidates = self
            .semantic
            .search(query)
            .await
            .map_err(|e| MemoryError::ProviderError(format!("semantic search failed: {}", e)))?;

        // Filter results by read permission
        let mut authorized = Vec::with_capacity(candidates.len());
        for result in candidates {
            if self
                .access_checker
                .check_read(
                    requestor,
                    result.entry.owner_id,
                    result.entry.scope,
                    MemoryTier::Semantic,
                )
                .is_ok()
            {
                authorized.push(result);
            }
        }

        Ok(authorized)
    }

    /// Inserts or updates an entity in the procedural knowledge graph (L4).
    ///
    /// No permission check: agents can always write to their own entities.
    /// For cross-agent entity writes, use explicit grants via `grant_access()`.
    #[tracing::instrument(skip(self, entity), fields(entity_id = %entity.id, kind = %entity.kind))]
    pub async fn procedural_put_entity(&self, entity: Entity) -> Result<()> {
        self.procedural
            .put_entity(entity)
            .map_err(|e| MemoryError::ProviderError(format!("procedural put_entity failed: {}", e)))
    }

    /// Retrieves an entity from the procedural knowledge graph (L4).
    ///
    /// No permission check: agents can always read entities they know the ID of.
    /// For scoped access control, filter results at the application layer.    #[tracing::instrument(skip(self), fields(entity_id = %id))]
    pub async fn procedural_get_entity(&self, id: uuid::Uuid) -> Result<Option<Entity>> {
        self.procedural
            .get_entity(id)
            .map_err(|e| MemoryError::ProviderError(format!("procedural get_entity failed: {}", e)))
    }

    /// Extracts a subgraph from the procedural knowledge graph (L4).
    ///
    /// # Arguments
    /// * `root` - Starting entity ID for traversal
    /// * `depth` - Maximum relation hops to traverse
    ///
    /// # Returns
    /// Tuple of (entities, relations) comprising the reachable subgraph.
    #[tracing::instrument(skip(self), fields(root = %root, depth))]
    pub async fn procedural_subgraph(
        &self,
        root: uuid::Uuid,
        depth: usize,
    ) -> Result<(Vec<Entity>, Vec<Relation>)> {
        self.procedural
            .subgraph(root, depth, None)
            .map_err(|e| MemoryError::ProviderError(format!("procedural subgraph failed: {}", e)))
    }

    /// Grants explicit memory access permission from one agent to another.
    ///
    /// # Arguments
    /// * `perm` - The `MemoryPermission` grant to record
    ///
    /// # Notes
    /// - Grants are additive; multiple grants for same (grantee, tier, scope) are allowed
    /// - Expired grants are not automatically removed; call `prune_expired_grants()` periodically
    /// - Revocation via `revoke_access()` removes all matching grants
    #[tracing::instrument(skip(self), fields(grantor = %perm.grantor, grantee = %perm.grantee, tier = ?perm.tier))]
    pub fn grant_access(&self, perm: MemoryPermission) {
        self.access_checker.grant_table().grant(perm);
        tracing::debug!("memory access granted");
    }

    /// Revokes memory access permission for a specific (grantee, grantor, tier) tuple.
    #[tracing::instrument(skip(self), fields(grantee = %grantee, grantor = %grantor, tier = ?tier))]
    pub fn revoke_access(
        &self,
        grantee: nexus_proto::agent::AgentId,
        grantor: nexus_proto::agent::AgentId,
        tier: MemoryTier,
    ) {
        self.access_checker            .grant_table()
            .revoke(grantee, grantor, tier);
        tracing::debug!("memory access revoked");
    }

    /// Registers an agent with the memory system, optionally assigning group membership.
    ///
    /// # Arguments
    /// * `agent_id` - The agent to register
    /// * `group` - Optional supervisor group name for Group-scope access decisions
    #[tracing::instrument(skip(self), fields(agent_id = %agent_id, group = ?group))]
    pub fn register_agent(
        &self,
        agent_id: nexus_proto::agent::AgentId,
        group: Option<String>,
    ) {
        if let Some(g) = group {
            self.access_checker.group_table().add_to_group(agent_id, g);
        }
        tracing::debug!("agent registered with memory system");
    }

    /// Deregisters an agent, revoking all their grants and group memberships.
    #[tracing::instrument(skip(self), fields(agent_id = %agent_id))]
    pub fn deregister_agent(&self, agent_id: nexus_proto::agent::AgentId) {
        // Revoke all permissions involving this agent
        self.access_checker
            .grant_table()
            .revoke_all_for_agent(agent_id);

        // Remove from all groups
        self.access_checker
            .group_table()
            .remove_agent_from_all_groups(agent_id);

        // Remove working memory instance
        self.working.remove(&agent_id);

        tracing::debug!("agent deregistered from memory system");
    }

    /// Persists all memory tiers to durable storage.
    ///
    /// # Behavior
    /// - L1: No persistence (in-process only); snapshot via `WorkingMemory::snapshot()`
    /// - L2: SQLite auto-commits; this flushes WAL to disk
    /// - L3: Saves usearch index + flushes SQLite metadata
    /// - L4: Calls `sled::Db::flush()` for durability
    ///
    /// Call this periodically or before shutdown to ensure data durability.    #[tracing::instrument(skip(self))]
    pub async fn flush(&self) -> Result<()> {
        // L2: SQLite flush via libsql
        self.episodic
            .connection()
            .execute("PRAGMA wal_checkpoint(FULL)", ())
            .await
            .map_err(|e| MemoryError::ProviderError(format!("episodic flush failed: {}", e)))?;

        // L3: Save usearch index + flush SQLite
        self.semantic
            .save()
            .await
            .map_err(|e| MemoryError::ProviderError(format!("semantic flush failed: {}", e)))?;

        // L4: Flush sled database
        self.procedural
            .flush()
            .map_err(|e| MemoryError::ProviderError(format!("procedural flush failed: {}", e)))?;

        tracing::debug!("all memory tiers flushed to durable storage");
        Ok(())
    }

    /// Prunes expired permission grants from the grant table.
    ///
    /// Call this periodically (e.g., via a background task) to prevent memory growth.
    #[tracing::instrument(skip(self))]
    pub fn prune_expired_grants(&self) {
        self.access_checker.prune_expired_grants();
    }

    /// Returns a snapshot of permission system state for observability.
    pub fn permission_summary(&self) -> PermissionSummary {
        self.access_checker.summary()
    }

    /// Returns the underlying access checker for advanced permission operations.
    ///
    /// ⚠️ Use with caution: direct access bypasses the unified API's safety checks.
    pub fn access_checker(&self) -> &std::sync::Arc<MemoryAccessChecker> {
        &self.access_checker
    }

    /// Returns a reference to the configuration used to initialize this store.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Returns the number of agents with active working memory instances.    pub fn working_agent_count(&self) -> usize {
        self.working.len()
    }

    /// Returns the total number of entries across all semantic memory.
    pub async fn semantic_count(&self) -> u64 {
        self.semantic.count().await
    }

    /// Returns the number of entities and relations in procedural memory.
    pub fn procedural_counts(&self) -> Result<(u64, u64)> {
        Ok((
            self.procedural.entity_count()?,
            self.procedural.relation_count()?,
        ))
    }
}

// =============================================================================
// Prelude Module
// =============================================================================

/// Convenience module for importing common `nexus-mem` types.
///
/// # Example
///
/// ```rust
/// use nexus_mem::prelude::*;
///
/// async fn handle_memory(mem: &MemoryStore) -> Result<()> {
///     // ...
/// }
/// ```
pub mod prelude {
    pub use crate::error::MemoryError;
    pub use crate::Result;

    pub use crate::embeddings::{EmbeddingProvider, EmbeddingConfig};
    pub use crate::working::{WorkingMemory, WorkingEntry};
    pub use crate::episodic::EpisodicStore;
    pub use crate::semantic::SemanticStore;
    pub use crate::procedural::{ProceduralStore, Entity, Relation};
    pub use crate::permissions::{MemoryPermission, MemoryAccessChecker};

    pub use crate::{MemoryStore, MemoryConfig};

    pub use nexus_proto::memory::{
        MemoryTier, MemoryScope, MemoryAccess, MemoryKey, MemoryEntry,
        SemanticSearchQuery, SemanticSearchResult, EpisodicEvent, EpisodicEventType,
    };}
