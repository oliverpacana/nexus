use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use libsql::{Connection, params};
use nexus_proto::agent::AgentId;
use nexus_proto::memory::{
    EmbeddingVector, MemoryEntry, MemoryKey, MemoryScope, MemoryTier, SemanticSearchQuery,
    SemanticSearchResult,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, instrument, warn};
use uuid::Uuid;

use crate::embeddings::{EmbeddingProvider, MemoryError};

pub type Result<T> = std::result::Result<T, MemoryError>;

// =============================================================================
// SemanticStore — Vector Index + Metadata Store (L3 Memory)
// =============================================================================

/// L3 semantic memory: HNSW vector index backed by `usearch` with SQLite metadata.
///
/// # Design
/// - Vectors stored in memory-mapped `usearch` HNSW index for fast similarity search
/// - Metadata (keys, scopes, values, tags, expiry) stored in SQLite for filtering
/// - Tombstone pattern for deletion (usearch doesn't support native removal)
/// - Embeddings computed on-write via pluggable `EmbeddingProvider`
///
/// # Thread Safety
/// - Index protected by `Arc<RwLock>` for concurrent read/write safety
/// - Tombstones and counters use atomic/sync primitives
/// - SQLite connection is `Send + Sync` per libsql design
pub struct SemanticStore {
    /// HNSW vector index (wrapped for thread safety)
    index: Arc<RwLock<usearch::Index>>,

    /// SQLite connection for metadata storage
    meta_db: Connection,

    /// Embedding provider for text→vector conversion
    embedder: Arc<dyn EmbeddingProvider + Send + Sync>,

    /// Dimensionality of embedding vectors
    dimensions: usize,

    /// Monotonically increasing internal ID allocator for usearch    next_id: AtomicU64,

    /// Tombstone set: internal IDs marked as deleted (usearch doesn't support removal)
    tombstones: Arc<RwLock<HashSet<u64>>>,

    /// Filesystem path for persisting the index
    index_path: String,
}

impl SemanticStore {
    /// Creates or loads a semantic memory store.
    ///
    /// # Arguments
    /// * `index_path` - Path to the usearch index file (created if missing)
    /// * `meta_db_path` - Path to SQLite metadata database
    /// * `embedder` - Provider for computing embeddings
    /// * `dimensions` - Expected vector dimensionality
    /// * `ef_construction` - HNSW graph construction expansion factor
    /// * `ef_search` - HNSW search expansion factor
    #[instrument(skip(embedder), fields(dimensions, ef_construction, ef_search))]
    pub async fn new(
        index_path: &str,
        meta_db_path: &str,
        embedder: Arc<dyn EmbeddingProvider + Send + Sync>,
        dimensions: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Result<Self> {
        debug!("initializing semantic memory store");

        // Initialize or load usearch index
        let index = if std::path::Path::new(index_path).exists() {
            debug!(path = %index_path, "loading existing usearch index");
            let opts = usearch::IndexOptions {
                dimensions,
                metric: usearch::MetricKind::Cos,
                quantization: usearch::ScalarKind::F32,
                connectivity: 16,
                expansion_add: ef_construction,
                expansion_search: ef_search,
            };
            usearch::Index::load(index_path, &opts)
                .map_err(|e| MemoryError::ProviderError(format!("failed to load index: {}", e)))?
        } else {
            debug!(path = %index_path, "creating new usearch index");
            let opts = usearch::IndexOptions {
                dimensions,
                metric: usearch::MetricKind::Cos,
                quantization: usearch::ScalarKind::F32,
                connectivity: 16,                expansion_add: ef_construction,
                expansion_search: ef_search,
            };
            usearch::Index::new(&opts)
                .map_err(|e| MemoryError::ProviderError(format!("failed to create index: {}", e)))?
        };

        let index = Arc::new(RwLock::new(index));

        // Open SQLite metadata database
        let meta_db = Database::open(meta_db_path)
            .map_err(|e| MemoryError::ProviderError(format!("failed to open meta DB: {}", e)))?
            .connect()
            .map_err(|e| MemoryError::ProviderError(format!("failed to connect to meta DB: {}", e)))?;

        let store = Self {
            index,
            meta_db,
            embedder,
            dimensions,
            next_id: AtomicU64::new(1),
            tombstones: Arc::new(RwLock::new(HashSet::new())),
            index_path: index_path.to_string(),
        };

        // Run migrations
        store.run_migrations().await?;

        // Initialize next_id counter from existing metadata
        let max_id: Option<i64> = store
            .meta_db
            .query_row("SELECT MAX(internal_id) FROM semantic_meta", (), |row| row.get(0))
            .await
            .ok()
            .flatten();

        if let Some(max) = max_id {
            store.next_id.store((max as u64) + 1, Ordering::Relaxed);
            debug!(next_id = max + 1, "initialized next_id from existing data");
        }

        debug!("semantic memory store initialized");
        Ok(store)
    }

    /// Creates metadata table and indexes if they don't exist.
    async fn run_migrations(&self) -> Result<()> {
        self.meta_db
            .execute(
                "CREATE TABLE IF NOT EXISTS semantic_meta (                    internal_id INTEGER PRIMARY KEY,
                    entry_id TEXT UNIQUE NOT NULL,
                    owner_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    key_ns TEXT NOT NULL,
                    key_name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    version INTEGER NOT NULL DEFAULT 1,
                    tier TEXT NOT NULL DEFAULT 'semantic',
                    embedding TEXT  -- base64 encoded for retrieval
                )",
                (),
            )
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to create meta table: {}", e)))?;

        self.meta_db
            .execute("CREATE INDEX IF NOT EXISTS idx_sem_owner ON semantic_meta(owner_id)", ())
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to create owner index: {}", e)))?;

        self.meta_db
            .execute("CREATE INDEX IF NOT EXISTS idx_sem_scope ON semantic_meta(scope)", ())
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to create scope index: {}", e)))?;

        self.meta_db
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_sem_expiry ON semantic_meta(expires_at)",
                (),
            )
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to create expiry index: {}", e)))?;

        Ok(())
    }

    /// Inserts a memory entry into the semantic index and metadata store.
    ///
    /// # Flow
    /// 1. Serialize value to string & embed
    /// 2. Allocate internal ID
    /// 3. Add vector to HNSW index
    /// 4. Insert metadata row to SQLite
    /// 5. Persist index to disk
    #[instrument(skip(self, entry), fields(owner_id = %entry.owner_id, key = %entry.key))]
    pub async fn insert(&self, entry: MemoryEntry) -> Result<()> {        if entry.embedding.is_none() || entry.embedding.as_ref().unwrap().dims() != self.dimensions {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimensions,
                actual: entry.embedding.map(|e| e.dims()).unwrap_or(0),
            });
        }

        let internal_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let vector = entry.embedding.as_ref().unwrap().as_slice();

        // Add to HNSW index
        {
            let mut idx = self.index.write().await;
            idx.add(internal_id, vector)
                .map_err(|e| MemoryError::ProviderError(format!("failed to add to index: {}", e)))?;
        }

        // Serialize embedding for storage (base64 or JSON array)
        let embedding_json = serde_json::to_string(vector)
            .map_err(|e| MemoryError::SerializationError(e))?;

        // Insert metadata
        let row = entry_to_row(&entry, internal_id, &embedding_json);
        self.meta_db
            .execute(
                "INSERT INTO semantic_meta (
                    internal_id, entry_id, owner_id, scope, key_ns, key_name,
                    value, tags, created_at, expires_at, version, tier, embedding
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'semantic', ?12)",
                params![
                    row.0 as i64, row.1, row.2, row.3, row.4, row.5,
                    row.6, row.7, row.8, row.9, row.10 as i64, row.11
                ],
            )
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to insert metadata: {}", e)))?;

        // Persist index
        self.save().await?;

        debug!(
            internal_id,
            owner_id = %entry.owner_id,
            key = %entry.key,
            "semantic entry inserted"
        );
        Ok(())
    }

    /// Searches the semantic index for similar entries.    ///
    /// # Flow
    /// 1. Embed query text if provided
    /// 2. Search HNSW for `top_k * 3` candidates (oversample for filtering)
    /// 3. Fetch metadata from SQLite
    /// 4. Apply scope, owner, tag filters
    /// 5. Compute similarity, apply min_similarity threshold
    /// 6. Sort descending, take top_k
    #[instrument(skip(self, query), fields(top_k = query.top_k, min_sim = query.min_similarity))]
    pub async fn search(&self, query: SemanticSearchQuery) -> Result<Vec<SemanticSearchResult>> {
        // 1. Get query embedding
        let query_vec = if let Some(emb) = query.query_embedding.as_ref() {
            emb.as_slice().to_vec()
        } else if let Some(text) = &query.query_text {
            let emb = self.embedder.embed(text).await?;
            emb.as_slice().to_vec()
        } else {
            return Err(MemoryError::ProviderError("search requires query_embedding or query_text".into()));
        };

        // 2. Search HNSW index (oversample)
        let oversample = query.top_k.saturating_mul(3).max(10);
        let candidates = {
            let idx = self.index.read().await;
            let tombstones = self.tombstones.read().await;

            idx.search(&query_vec, oversample)
                .map_err(|e| MemoryError::ProviderError(format!("search failed: {}", e)))?
                .into_iter()
                .filter(|(id, _)| !tombstones.contains(id))
                .collect::<Vec<_>>()
        };

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // 3. Load metadata for candidates
        let ids: Vec<String> = candidates.iter().map(|(id, _)| id.to_string()).collect();
        let placeholders: Vec<&str> = (0..ids.len()).map(|_| "?").collect();
        let in_clause = placeholders.join(",");

        let query_sql = format!(
            "SELECT internal_id, entry_id, owner_id, scope, key_ns, key_name, value, tags, 
                    created_at, expires_at, version, tier, embedding 
             FROM semantic_meta WHERE internal_id IN ({}) ORDER BY internal_id",
            in_clause
        );

        let param_values: Vec<libsql::Value> = ids.into_iter().map(Into::into).collect();        let mut rows = self.meta_db.query(&query_sql, param_values.as_slice())
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to query metadata: {}", e)))?;

        let mut meta_map = std::collections::HashMap::new();
        while let Some(row) = rows.next().await.map_err(|e| {
            MemoryError::ProviderError(format!("failed to fetch row: {}", e))
        })? {
            let internal_id: i64 = row.get(0)?;
            let entry = row_to_entry(row, internal_id as u64)?;
            meta_map.insert(internal_id as u64, entry);
        }

        // 4 & 5. Filter, compute similarity, apply thresholds
        let mut scored = Vec::with_capacity(candidates.len());
        for (internal_id, distance) in candidates {
            if let Some(entry) = meta_map.get(&internal_id) {
                // Apply scope/owner/tag filters
                if let Some(scope) = query.scope_filter {
                    if entry.scope != scope {
                        continue;
                    }
                }
                if let Some(owner) = query.owner_filter {
                    if entry.owner_id != owner {
                        continue;
                    }
                }
                if !query.tag_filter.is_empty() {
                    if !entry.tags.iter().any(|t| query.tag_filter.contains(t)) {
                        continue;
                    }
                }

                // Cosine similarity: distance from usearch Cos metric = 1 - cosine
                let similarity = (1.0 - distance).max(0.0).min(1.0);
                if similarity >= query.min_similarity {
                    scored.push((entry.clone(), similarity));
                }
            }
        }

        // 6. Sort and take top_k
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let results = scored
            .into_iter()
            .take(query.top_k)
            .enumerate()
            .map(|(i, (entry, sim))| SemanticSearchResult {
                entry,                similarity: sim,
                rank: i + 1,
            })
            .collect();

        debug!(count = results.len(), "semantic search completed");
        Ok(results)
    }

    /// Retrieves a single entry by key and owner.
    #[instrument(skip(self, key, owner), fields(owner_id = %owner))]
    pub async fn get(&self, key: &MemoryKey, owner: AgentId) -> Result<Option<MemoryEntry>> {
        let row = self.meta_db
            .query_row(
                "SELECT internal_id, entry_id, owner_id, scope, key_ns, key_name, value, tags,
                        created_at, expires_at, version, tier, embedding
                 FROM semantic_meta WHERE owner_id = ?1 AND key_ns = ?2 AND key_name = ?3",
                params![owner.to_string(), key.namespace, key.key],
                |row| row,
            )
            .await;

        match row {
            Ok(r) => {
                let internal_id: i64 = r.get(0)?;
                let entry = row_to_entry(r, internal_id as u64)?;
                // Check tombstones
                let tombstones = self.tombstones.read().await;
                if tombstones.contains(&internal_id as u64) {
                    Ok(None)
                } else {
                    Ok(Some(entry))
                }
            }
            Err(libsql::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(MemoryError::ProviderError(format!("failed to get entry: {}", e))),
        }
    }

    /// Updates an existing entry: re-embeds, replaces vector, increments version.
    #[instrument(skip(self, entry), fields(key = %entry.key))]
    pub async fn update(&self, entry: MemoryEntry) -> Result<()> {
        let existing = self.get(&entry.key, entry.owner_id).await?;
        if existing.is_none() {
            return Err(MemoryError::ProviderError("entry not found for update".into()));
        }

        // Embed new value
        let value_str = serde_json::to_string(&entry.value)
            .unwrap_or_else(|_| entry.key.key.clone());        let emb = self.embedder.embed(&value_str).await?;

        let mut updated_entry = entry;
        updated_entry.embedding = Some(emb);
        updated_entry.version = existing.unwrap().version + 1;
        updated_entry.updated_at = Utc::now();

        // Insert new vector (overwrite not supported, so delete + insert)
        self.delete(&updated_entry.key, updated_entry.owner_id).await?;
        self.insert(updated_entry).await
    }

    /// Marks an entry as deleted (tombstone + SQLite removal).
    #[instrument(skip(self, key, owner), fields(owner_id = %owner))]
    pub async fn delete(&self, key: &MemoryKey, owner: AgentId) -> Result<bool> {
        let row = self.meta_db
            .query_row(
                "SELECT internal_id FROM semantic_meta WHERE owner_id = ?1 AND key_ns = ?2 AND key_name = ?3",
                params![owner.to_string(), key.namespace, key.key],
                |row| row.get::<i64, _>(0),
            )
            .await;

        match row {
            Ok(internal_id) => {
                let id_u64 = internal_id as u64;
                // Add to tombstones
                self.tombstones.write().await.insert(id_u64);

                // Remove from SQLite
                self.meta_db
                    .execute(
                        "DELETE FROM semantic_meta WHERE internal_id = ?1",
                        params![internal_id],
                    )
                    .await
                    .map_err(|e| MemoryError::ProviderError(format!("failed to delete metadata: {}", e)))?;

                // Note: usearch doesn't support removal, tombstone handles it
                debug!(internal_id, "semantic entry tombstoned");
                Ok(true)
            }
            Err(libsql::Error::QueryReturnedNoRows) => Ok(false),
            Err(e) => Err(MemoryError::ProviderError(format!("failed to query for delete: {}", e))),
        }
    }

    /// Deletes all entries belonging to an owner.
    #[instrument(skip(self, owner), fields(owner_id = %owner))]
    pub async fn delete_by_owner(&self, owner: AgentId) -> Result<u64> {        let owner_str = owner.to_string();

        // Fetch all internal IDs for this owner
        let ids: Vec<i64> = {
            let mut rows = self.meta_db
                .query("SELECT internal_id FROM semantic_meta WHERE owner_id = ?1", params![owner_str])
                .await
                .map_err(|e| MemoryError::ProviderError(format!("failed to query owner entries: {}", e)))?;
            let mut vec = Vec::new();
            while let Some(row) = rows.next().await.map_err(|e| {
                MemoryError::ProviderError(format!("failed to fetch row: {}", e))
            })? {
                let id: i64 = row.get(0)?;
                vec.push(id);
            }
            vec
        };

        if ids.is_empty() {
            return Ok(0);
        }

        // Tombstone all
        let tombstone_ids: HashSet<u64> = ids.iter().map(|&id| id as u64).collect();
        self.tombstones.write().await.extend(tombstone_ids);

        // Delete from SQLite
        let placeholders: Vec<&str> = (0..ids.len()).map(|_| "?").collect();
        let sql = format!("DELETE FROM semantic_meta WHERE owner_id = ?1");
        let deleted = self.meta_db
            .execute(&sql, params![owner_str])
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to delete by owner: {}", e)))?;

        self.save().await?;
        debug!(owner_id = %owner, deleted, "deleted owner entries");
        Ok(deleted as u64)
    }

    /// Returns the total count of active (non-tombstoned) entries.
    pub async fn count(&self) -> u64 {
        let db_count: i64 = self.meta_db
            .query_row("SELECT COUNT(*) FROM semantic_meta", (), |row| row.get(0))
            .await
            .unwrap_or(0);

        let tombstone_count = self.tombstones.read().await.len();
        (db_count as u64).saturating_sub(tombstone_count as u64)
    }
    /// Persists the HNSW index to disk.
    pub async fn save(&self) -> Result<()> {
        let idx = self.index.read().await;
        idx.save(&self.index_path)
            .map_err(|e| MemoryError::ProviderError(format!("failed to save index: {}", e)))?;
        debug!(path = %self.index_path, "index persisted to disk");
        Ok(())
    }

    /// Finds and removes expired entries.
    #[instrument(skip(self))]
    pub async fn evict_expired(&self) -> Result<u64> {
        let now = Utc::now().to_rfc3339();

        let ids: Vec<i64> = {
            let mut rows = self.meta_db
                .query(
                    "SELECT internal_id FROM semantic_meta WHERE expires_at IS NOT NULL AND expires_at <= ?1",
                    params![now],
                )
                .await
                .map_err(|e| MemoryError::ProviderError(format!("failed to query expired: {}", e)))?;
            let mut vec = Vec::new();
            while let Some(row) = rows.next().await.map_err(|e| {
                MemoryError::ProviderError(format!("failed to fetch row: {}", e))
            })? {
                vec.push(row.get(0)?);
            }
            vec
        };

        if ids.is_empty() {
            return Ok(0);
        }

        // Tombstone
        let tombstone_ids: HashSet<u64> = ids.iter().map(|&id| id as u64).collect();
        self.tombstones.write().await.extend(tombstone_ids);

        // Delete from SQLite
        let placeholders: Vec<&str> = (0..ids.len()).map(|_| "?").collect();
        let sql = format!(
            "DELETE FROM semantic_meta WHERE internal_id IN ({})",
            placeholders.join(",")
        );
        let param_values: Vec<libsql::Value> = ids.into_iter().map(|id| (id as i64).into()).collect();

        let deleted = self.meta_db
            .execute(&sql, param_values.as_slice())
            .await            .map_err(|e| MemoryError::ProviderError(format!("failed to delete expired: {}", e)))?;

        self.save().await?;
        debug!(deleted, "evicted expired entries");
        Ok(deleted as u64)
    }
}

// =============================================================================
// Serialization Helpers
// =============================================================================

/// Converts a `MemoryEntry` to a row tuple for SQLite insertion.
fn entry_to_row(
    entry: &MemoryEntry,
    internal_id: u64,
    embedding_json: &str,
) -> (u64, String, String, String, String, String, String, String, String, Option<String>, u64, String) {
    (
        internal_id,
        entry.key.to_string(),
        entry.owner_id.to_string(),
        scope_to_string(&entry.scope),
        entry.key.namespace.clone(),
        entry.key.key.clone(),
        serde_json::to_string(&entry.value).unwrap_or_else(|_| "{}".into()),
        serde_json::to_string(&entry.tags).unwrap_or_else(|_| "[]".into()),
        entry.created_at.to_rfc3339(),
        entry.expires_at.map(|dt| dt.to_rfc3339()),
        entry.version,
        embedding_json.to_string(),
    )
}

/// Converts a SQLite row back to a `MemoryEntry`.
fn row_to_entry(row: libsql::Row, internal_id: u64) -> Result<MemoryEntry> {
    let entry_id: String = row.get(1)?;
    let owner_id: String = row.get(2)?;
    let scope_str: String = row.get(3)?;
    let key_ns: String = row.get(4)?;
    let key_name: String = row.get(5)?;
    let value_str: String = row.get(6)?;
    let tags_str: String = row.get(7)?;
    let created_at_str: String = row.get(8)?;
    let expires_at_opt: Option<String> = row.get(9)?;
    let version: i64 = row.get(10)?;
    let embedding_str: Option<String> = row.get(11)?;

    let owner_id = AgentId::from(owner_id.as_str());
    let scope = string_to_scope(&scope_str).ok_or_else(|| {        MemoryError::ProviderError(format!("invalid scope: {}", scope_str))
    })?;

    let key = MemoryKey::new(key_ns, key_name);
    let value: serde_json::Value = serde_json::from_str(&value_str)
        .map_err(|e| MemoryError::SerializationError(e))?;
    let tags: Vec<String> = serde_json::from_str(&tags_str)
        .unwrap_or_default();
    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| MemoryError::ProviderError(format!("invalid created_at: {}", e)))?;
    let expires_at = expires_at_opt
        .map(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| MemoryError::ProviderError(format!("invalid expires_at: {}", e)))
        })
        .transpose()?;

    let embedding = embedding_str
        .map(|s| {
            let vec: Vec<f32> = serde_json::from_str(&s)
                .map_err(|e| MemoryError::SerializationError(e))?;
            Ok(EmbeddingVector::from_vec(vec))
        })
        .transpose()?;

    Ok(MemoryEntry {
        key,
        tier: MemoryTier::Semantic,
        scope,
        owner_id,
        value,
        embedding,
        created_at,
        updated_at: Utc::now(),
        expires_at,
        version: version as u64,
        tags,
    })
}

#[inline]
fn scope_to_string(scope: &MemoryScope) -> String {
    match scope {
        MemoryScope::Private => "private",
        MemoryScope::Group => "group",
        MemoryScope::Global => "global",
    }
    .to_string()}

#[inline]
fn string_to_scope(s: &str) -> Option<MemoryScope> {
    match s {
        "private" => Some(MemoryScope::Private),
        "group" => Some(MemoryScope::Group),
        "global" => Some(MemoryScope::Global),
        _ => None,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::LocalEmbeddingProvider;
    use nexus_proto::memory::{EpisodicEvent, EpisodicEventType};
    use std::collections::HashMap;

    fn test_agent_id() -> AgentId {
        AgentId::new()
    }

    #[tokio::test]
    async fn test_semantic_store_lifecycle() {
        let embedder = Arc::new(LocalEmbeddingProvider::new(64));
        let store = SemanticStore::new(
            "/tmp/nexus_test_semantic.idx",
            "/tmp/nexus_test_semantic_meta.db",
            embedder,
            64,
            32,
            16,
        )
        .await
        .unwrap();

        let agent_id = test_agent_id();
        let key = MemoryKey::new("facts", "rust_is_safe");
        let entry = MemoryEntry {
            key: key.clone(),
            tier: MemoryTier::Semantic,
            scope: MemoryScope::Global,
            owner_id: agent_id,
            value: serde_json::json!("Rust guarantees memory safety without GC"),
            embedding: Some(embedder.embed("Rust guarantees memory safety without GC").await.unwrap()),            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
            version: 1,
            tags: vec!["programming", "rust".to_string()],
        };

        store.insert(entry).await.unwrap();
        assert_eq!(store.count().await, 1);

        // Search
        let results = store
            .search(SemanticSearchQuery {
                query_embedding: None,
                query_text: Some("memory safety programming".to_string()),
                top_k: 5,
                min_similarity: 0.1,
                scope_filter: None,
                owner_filter: None,
                tag_filter: vec![],
            })
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].entry.key, key);

        // Cleanup
        std::fs::remove_file("/tmp/nexus_test_semantic.idx").ok();
        std::fs::remove_file("/tmp/nexus_test_semantic_meta.db").ok();
    }

    #[tokio::test]
    async fn test_tombstone_deletion() {
        let embedder = Arc::new(LocalEmbeddingProvider::new(32));
        let store = SemanticStore::new(
            "/tmp/nexus_test_tombstone.idx",
            "/tmp/nexus_test_tombstone_meta.db",
            embedder,
            32,
            32,
            16,
        )
        .await
        .unwrap();

        let agent_id = test_agent_id();
        let key1 = MemoryKey::new("test", "key1");
        let key2 = MemoryKey::new("test", "key2");
        for (k, v) in [(&key1, "value one"), (&key2, "value two")] {
            let emb = embedder.embed(v).await.unwrap();
            let entry = MemoryEntry {
                key: k.clone(),
                tier: MemoryTier::Semantic,
                scope: MemoryScope::Private,
                owner_id: agent_id,
                value: serde_json::json!(v),
                embedding: Some(emb),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                expires_at: None,
                version: 1,
                tags: vec![],
            };
            store.insert(entry).await.unwrap();
        }

        assert_eq!(store.count().await, 2);

        // Delete one
        store.delete(&key1, agent_id).await.unwrap();
        assert_eq!(store.count().await, 1);

        // Verify it's filtered from search
        let results = store
            .search(SemanticSearchQuery {
                query_embedding: None,
                query_text: Some("test".to_string()),
                top_k: 5,
                min_similarity: 0.0,
                scope_filter: None,
                owner_filter: None,
                tag_filter: vec![],
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.key.key, "key2");

        // Cleanup
        std::fs::remove_file("/tmp/nexus_test_tombstone.idx").ok();
        std::fs::remove_file("/tmp/nexus_test_tombstone_meta.db").ok();
    }

    #[tokio::test]
    async fn test_evict_expired() {
        let embedder = Arc::new(LocalEmbeddingProvider::new(32));
        let store = SemanticStore::new(            "/tmp/nexus_test_expire.idx",
            "/tmp/nexus_test_expire_meta.db",
            embedder,
            32,
            32,
            16,
        )
        .await
        .unwrap();

        let agent_id = test_agent_id();
        let now = Utc::now();

        // Insert expired
        let key1 = MemoryKey::new("tmp", "old");
        let emb1 = embedder.embed("old data").await.unwrap();
        store
            .insert(MemoryEntry {
                key: key1,
                tier: MemoryTier::Semantic,
                scope: MemoryScope::Global,
                owner_id: agent_id,
                value: serde_json::json!("old"),
                embedding: Some(emb1),
                created_at: now - chrono::Duration::hours(2),
                updated_at: now,
                expires_at: Some(now - chrono::Duration::minutes(1)),
                version: 1,
                tags: vec![],
            })
            .await
            .unwrap();

        // Insert valid
        let key2 = MemoryKey::new("tmp", "new");
        let emb2 = embedder.embed("new data").await.unwrap();
        store
            .insert(MemoryEntry {
                key: key2,
                tier: MemoryTier::Semantic,
                scope: MemoryScope::Global,
                owner_id: agent_id,
                value: serde_json::json!("new"),
                embedding: Some(emb2),
                created_at: now,
                updated_at: now,
                expires_at: Some(now + chrono::Duration::hours(1)),
                version: 1,
                tags: vec![],
            })            .await
            .unwrap();

        assert_eq!(store.count().await, 2);
        let evicted = store.evict_expired().await.unwrap();
        assert_eq!(evicted, 1);
        assert_eq!(store.count().await, 1);

        // Cleanup
        std::fs::remove_file("/tmp/nexus_test_expire.idx").ok();
        std::fs::remove_file("/tmp/nexus_test_expire_meta.db").ok();
    }
}
