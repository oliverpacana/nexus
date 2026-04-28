use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use nexus_proto::agent::AgentId;
use serde::{Deserialize, Serialize};
use sled::{Config, Db, Tree};
use tracing::{debug, error, instrument, warn};
use uuid::Uuid;

use crate::embeddings::MemoryError;

pub type Result<T> = std::result::Result<T, MemoryError>;

// =============================================================================
// Entity — Knowledge Graph Node
// =============================================================================

/// A node in the procedural knowledge graph representing a typed entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier for this entity.
    pub id: Uuid,

    /// Semantic type/kind of this entity (e.g., "person", "concept", "tool").
    pub kind: String,

    /// Human-readable name or label.
    pub name: String,

    /// Arbitrary key-value properties describing this entity.
    pub properties: HashMap<String, serde_json::Value>,

    /// Timestamp when this entity was first created.
    pub created_at: DateTime<Utc>,

    /// Timestamp of the last modification.
    pub updated_at: DateTime<Utc>,

    /// Optimistic locking version for concurrent updates.
    pub version: u64,
}

impl Entity {
    /// Creates a new entity with the given kind and name.
    pub fn new(kind: impl Into<String>, name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            kind: kind.into(),            name: name.into(),
            properties: HashMap::new(),
            created_at: now,
            updated_at: now,
            version: 1,
        }
    }

    /// Adds or updates a property on this entity.
    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self.updated_at = Utc::now();
        self
    }

    /// Gets a property value by key, if present.
    pub fn get_property(&self, key: &str) -> Option<&serde_json::Value> {
        self.properties.get(key)
    }

    /// Returns the age of this entity in seconds.
    pub fn age_secs(&self) -> u64 {
        self.created_at.elapsed().unwrap_or_default().as_secs()
    }
}

// =============================================================================
// Relation — Knowledge Graph Edge
// =============================================================================

/// A directed, typed relationship between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Unique identifier for this relation.
    pub id: Uuid,

    /// Source entity ID (edge origin).
    pub from_id: Uuid,

    /// Target entity ID (edge destination).
    pub to_id: Uuid,

    /// Semantic type of this relationship (e.g., "works_at", "knows", "is_a").
    pub relation_type: String,

    /// Arbitrary properties describing this relation instance.
    pub properties: HashMap<String, serde_json::Value>,

    /// Timestamp when this relation was created.
    pub created_at: DateTime<Utc>,
    /// Confidence score for this relation (0.0 = uncertain, 1.0 = certain).
    pub confidence: f32,
}

impl Relation {
    /// Creates a new relation between two entities.
    pub fn new(
        from_id: Uuid,
        to_id: Uuid,
        relation_type: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            from_id,
            to_id,
            relation_type: relation_type.into(),
            properties: HashMap::new(),
            created_at: Utc::now(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Adds a property to this relation.
    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    /// Returns whether this relation points from `from` to `to`.
    pub fn connects(&self, from: Uuid, to: Uuid) -> bool {
        self.from_id == from && self.to_id == to
    }
}

// =============================================================================
// ProceduralStore — Sled-Backed Knowledge Graph (L4 Memory)
// =============================================================================

/// L4 procedural memory: a persistent, queryable knowledge graph stored in sled.
///
/// # Design
/// - Entities and relations stored as JSON in separate sled trees
/// - Secondary indexes for efficient lookups by kind, name, and connectivity
/// - Transactional writes ensure atomicity for entity+index updates
/// - Subgraph extraction via BFS/DFS for reasoning and context building
///
/// # Thread Safety
/// - `sled::Db` and `sled::Tree` are `Send + Sync` and safe for concurrent access/// - All public methods are synchronous but fast (sled is embedded, no network)
/// - Use async wrapper in calling code if needed for non-blocking I/O
pub struct ProceduralStore {
    /// Root sled database.
    db: Db,

    /// Tree: entity_id (bytes) → Entity (JSON)
    entities: Tree,

    /// Tree: relation_id (bytes) → Relation (JSON)
    relations: Tree,

    /// Index: "kind:{kind}:{entity_id}" → b"" for kind-based filtering
    kind_index: Tree,

    /// Index: "from:{from_id}:{relation_id}" → b"" for outgoing edge lookup
    from_index: Tree,

    /// Index: "to:{to_id}:{relation_id}" → b"" for incoming edge lookup
    to_index: Tree,

    /// Index: name (bytes) → entity_id (bytes) for name-based lookup
    /// Note: names are not guaranteed unique; returns first match
    name_index: Tree,
}

impl ProceduralStore {
    /// Opens or creates a procedural memory store at the given path.
    ///
    /// # Arguments
    /// * `db_path` - Filesystem path for the sled database directory
    ///
    /// # Returns
    /// * `Ok(ProceduralStore)` - If database opened and trees initialized
    /// * `Err(MemoryError)` - If sled access failed
    #[instrument(skip(db_path), fields(path = %db_path))]
    pub fn new(db_path: &str) -> Result<Self> {
        debug!("opening procedural memory database");

        let db = Config::new()
            .path(db_path)
            .mode(sled::Mode::HighThroughput)
            .cache_capacity(256 * 1024 * 1024) // 256 MB
            .flush_every_ms(Some(1000))
            .open()
            .map_err(|e| MemoryError::ProviderError(format!("failed to open sled db: {}", e)))?;

        // Open or create trees
        let entities = db
            .open_tree("entities")            .map_err(|e| MemoryError::ProviderError(format!("failed to open entities tree: {}", e)))?;

        let relations = db
            .open_tree("relations")
            .map_err(|e| MemoryError::ProviderError(format!("failed to open relations tree: {}", e)))?;

        let kind_index = db
            .open_tree("idx_kind")
            .map_err(|e| MemoryError::ProviderError(format!("failed to open kind_index: {}", e)))?;

        let from_index = db
            .open_tree("idx_from")
            .map_err(|e| MemoryError::ProviderError(format!("failed to open from_index: {}", e)))?;

        let to_index = db
            .open_tree("idx_to")
            .map_err(|e| MemoryError::ProviderError(format!("failed to open to_index: {}", e)))?;

        let name_index = db
            .open_tree("idx_name")
            .map_err(|e| MemoryError::ProviderError(format!("failed to open name_index: {}", e)))?;

        debug!("procedural memory initialized");
        Ok(Self {
            db,
            entities,
            relations,
            kind_index,
            from_index,
            to_index,
            name_index,
        })
    }

    /// Inserts or updates an entity with atomic index maintenance.
    ///
    /// # Transactional Guarantees
    /// - Entity write + kind_index + name_index updates are atomic
    /// - If any part fails, none are applied (sled transaction)
    #[instrument(skip(self, entity), fields(entity_id = %entity.id, kind = %entity.kind))]
    pub fn put_entity(&self, entity: Entity) -> Result<()> {
        let entity_id_bytes = entity.id.as_bytes();
        let entity_json = serde_json::to_vec(&entity)
            .map_err(|e| MemoryError::SerializationError(e))?;

        // Transactional write: entity + indexes
        self.db
            .transaction(|tx_db| {
                // Open trees within transaction context
                let entities = tx_db.open_tree("entities")?;                let kind_index = tx_db.open_tree("idx_kind")?;
                let name_index = tx_db.open_tree("idx_name")?;

                // Insert/update entity
                entities.insert(entity_id_bytes, &entity_json)?;

                // Update kind index
                let kind_key = format!("kind:{}:{}", entity.kind, entity.id);
                kind_index.insert(kind_key.as_bytes(), b"")?;

                // Update name index (name → entity_id)
                name_index.insert(entity.name.as_bytes(), entity_id_bytes)?;

                Ok(())
            })
            .map_err(|e| MemoryError::ProviderError(format!("transaction failed: {}", e)))?;

        debug!(entity_id = %entity.id, name = %entity.name, "entity stored");
        Ok(())
    }

    /// Retrieves an entity by its UUID.
    #[instrument(skip(self), fields(entity_id = %id))]
    pub fn get_entity(&self, id: Uuid) -> Result<Option<Entity>> {
        match self.entities.get(id.as_bytes())
            .map_err(|e| MemoryError::ProviderError(format!("failed to get entity: {}", e)))?
        {
            Some(data) => {
                let entity: Entity = serde_json::from_slice(&data)
                    .map_err(|e| MemoryError::SerializationError(e))?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Retrieves an entity by name (first match if multiple exist).
    #[instrument(skip(self), fields(name))]
    pub fn get_entity_by_name(&self, name: &str) -> Result<Option<Entity>> {
        match self.name_index.get(name.as_bytes())
            .map_err(|e| MemoryError::ProviderError(format!("failed to query name index: {}", e)))?
        {
            Some(id_bytes) => {
                let id = Uuid::from_slice(&id_bytes)
                    .map_err(|e| MemoryError::ProviderError(format!("invalid entity id in index: {}", e)))?;
                self.get_entity(id)
            }
            None => Ok(None),
        }
    }
    /// Finds all entities of a given kind.
    #[instrument(skip(self), fields(kind))]
    pub fn find_entities_by_kind(&self, kind: &str) -> Result<Vec<Entity>> {
        let prefix = format!("kind:{}:", kind);
        let mut entities = Vec::new();

        for item in self.kind_index.scan_prefix(prefix.as_bytes()) {
            let (key, _) = item.map_err(|e| {
                MemoryError::ProviderError(format!("failed to scan kind index: {}", e))
            })?;

            // Extract entity_id from key: "kind:{kind}:{entity_id}"
            if let Some(id_str) = key.strip_prefix(prefix.as_bytes()) {
                if let Ok(id) = Uuid::from_slice(id_str) {
                    if let Ok(Some(entity)) = self.get_entity(id) {
                        entities.push(entity);
                    }
                }
            }
        }

        debug!(kind, count = entities.len(), "found entities by kind");
        Ok(entities)
    }

    /// Deletes an entity and all its connected relations.
    ///
    /// # Returns
    /// `true` if the entity existed and was deleted, `false` otherwise.
    #[instrument(skip(self), fields(entity_id = %id))]
    pub fn delete_entity(&self, id: Uuid) -> Result<bool> {
        let id_bytes = id.as_bytes();

        // Check if entity exists
        if self.entities.get(id_bytes)?.is_none() {
            return Ok(false);
        }

        // Transactional delete: entity + all indexes + connected relations
        self.db
            .transaction(|tx_db| {
                let entities = tx_db.open_tree("entities")?;
                let relations = tx_db.open_tree("relations")?;
                let kind_index = tx_db.open_tree("idx_kind")?;
                let name_index = tx_db.open_tree("idx_name")?;
                let from_index = tx_db.open_tree("idx_from")?;
                let to_index = tx_db.open_tree("idx_to")?;

                // Get entity to know its kind and name for index cleanup                let entity_data = entities.get(id_bytes)?.ok_or(sled::Error::Conflict)?;
                let entity: Entity = serde_json::from_slice(&entity_data)
                    .map_err(|_| sled::Error::Compression)?;

                // Remove entity
                entities.remove(id_bytes)?;

                // Remove from kind index
                let kind_key = format!("kind:{}:{}", entity.kind, entity.id);
                kind_index.remove(kind_key.as_bytes())?;

                // Remove from name index
                name_index.remove(entity.name.as_bytes())?;

                // Find and delete all relations to/from this entity
                let from_prefix = format!("from:{}:", entity.id);
                for item in from_index.scan_prefix(from_prefix.as_bytes()) {
                    let (_, rel_id_bytes) = item?;
                    if let Ok(rel_id) = Uuid::from_slice(&rel_id_bytes) {
                        relations.remove(rel_id.as_bytes())?;
                        // Also clean up to_index entry
                        let to_key = format!("to:{}", rel_id);
                        to_index.remove(to_key.as_bytes())?;
                    }
                }
                from_index.remove_prefix(from_prefix.as_bytes())?;

                let to_prefix = format!("to:{}:", entity.id);
                for item in to_index.scan_prefix(to_prefix.as_bytes()) {
                    let (_, rel_id_bytes) = item?;
                    if let Ok(rel_id) = Uuid::from_slice(&rel_id_bytes) {
                        relations.remove(rel_id.as_bytes())?;
                        // Also clean up from_index entry
                        let from_key = format!("from:{}", rel_id);
                        from_index.remove(from_key.as_bytes())?;
                    }
                }
                to_index.remove_prefix(to_prefix.as_bytes())?;

                Ok(())
            })
            .map_err(|e| MemoryError::ProviderError(format!("delete transaction failed: {}", e)))?;

        debug!(entity_id = %id, "entity and connected relations deleted");
        Ok(true)
    }

    /// Inserts or updates a relation with atomic index maintenance.
    #[instrument(skip(self, relation), fields(relation_id = %relation.id))]
    pub fn put_relation(&self, relation: Relation) -> Result<()> {        let rel_id_bytes = relation.id.as_bytes();
        let rel_json = serde_json::to_vec(&relation)
            .map_err(|e| MemoryError::SerializationError(e))?;

        self.db
            .transaction(|tx_db| {
                let relations = tx_db.open_tree("relations")?;
                let from_index = tx_db.open_tree("idx_from")?;
                let to_index = tx_db.open_tree("idx_to")?;

                // Insert/update relation
                relations.insert(rel_id_bytes, &rel_json)?;

                // Update from_index
                let from_key = format!("from:{}:{}", relation.from_id, relation.id);
                from_index.insert(from_key.as_bytes(), rel_id_bytes)?;

                // Update to_index
                let to_key = format!("to:{}:{}", relation.to_id, relation.id);
                to_index.insert(to_key.as_bytes(), rel_id_bytes)?;

                Ok(())
            })
            .map_err(|e| MemoryError::ProviderError(format!("relation transaction failed: {}", e)))?;

        debug!(
            relation_id = %relation.id,
            from = %relation.from_id,
            to = %relation.to_id,
            r#type = %relation.relation_type,
            "relation stored"
        );
        Ok(())
    }

    /// Retrieves a relation by its UUID.
    #[instrument(skip(self), fields(relation_id = %id))]
    pub fn get_relation(&self, id: Uuid) -> Result<Option<Relation>> {
        match self.relations.get(id.as_bytes())
            .map_err(|e| MemoryError::ProviderError(format!("failed to get relation: {}", e)))?
        {
            Some(data) => {
                let relation: Relation = serde_json::from_slice(&data)
                    .map_err(|e| MemoryError::SerializationError(e))?;
                Ok(Some(relation))
            }
            None => Ok(None),
        }
    }
    /// Retrieves all outgoing relations from an entity.
    #[instrument(skip(self), fields(from_id = %entity_id))]
    pub fn get_relations_from(&self, entity_id: Uuid) -> Result<Vec<Relation>> {
        let prefix = format!("from:{}:", entity_id);
        let mut relations = Vec::new();

        for item in self.from_index.scan_prefix(prefix.as_bytes()) {
            let (_, rel_id_bytes) = item.map_err(|e| {
                MemoryError::ProviderError(format!("failed to scan from_index: {}", e))
            })?;

            if let Ok(rel_id) = Uuid::from_slice(&rel_id_bytes) {
                if let Ok(Some(rel)) = self.get_relation(rel_id) {
                    relations.push(rel);
                }
            }
        }

        debug!(from_id = %entity_id, count = relations.len(), "retrieved outgoing relations");
        Ok(relations)
    }

    /// Retrieves all incoming relations to an entity.
    #[instrument(skip(self), fields(to_id = %entity_id))]
    pub fn get_relations_to(&self, entity_id: Uuid) -> Result<Vec<Relation>> {
        let prefix = format!("to:{}:", entity_id);
        let mut relations = Vec::new();

        for item in self.to_index.scan_prefix(prefix.as_bytes()) {
            let (_, rel_id_bytes) = item.map_err(|e| {
                MemoryError::ProviderError(format!("failed to scan to_index: {}", e))
            })?;

            if let Ok(rel_id) = Uuid::from_slice(&rel_id_bytes) {
                if let Ok(Some(rel)) = self.get_relation(rel_id) {
                    relations.push(rel);
                }
            }
        }

        debug!(to_id = %entity_id, count = relations.len(), "retrieved incoming relations");
        Ok(relations)
    }

    /// Retrieves all relations connecting two specific entities.
    #[instrument(skip(self), fields(from = %from_id, to = %to_id))]
    pub fn get_relations_between(&self, from_id: Uuid, to_id: Uuid) -> Result<Vec<Relation>> {
        let prefix = format!("from:{}:", from_id);
        let mut relations = Vec::new();
        for item in self.from_index.scan_prefix(prefix.as_bytes()) {
            let (_, rel_id_bytes) = item.map_err(|e| {
                MemoryError::ProviderError(format!("failed to scan from_index: {}", e))
            })?;

            if let Ok(rel_id) = Uuid::from_slice(&rel_id_bytes) {
                if let Ok(Some(rel)) = self.get_relation(rel_id) {
                    if rel.to_id == to_id {
                        relations.push(rel);
                    }
                }
            }
        }

        Ok(relations)
    }

    /// Deletes a relation and its index entries.
    ///
    /// # Returns
    /// `true` if the relation existed and was deleted, `false` otherwise.
    #[instrument(skip(self), fields(relation_id = %id))]
    pub fn delete_relation(&self, id: Uuid) -> Result<bool> {
        let id_bytes = id.as_bytes();

        // Check if relation exists
        if self.relations.get(id_bytes)?.is_none() {
            return Ok(false);
        }

        self.db
            .transaction(|tx_db| {
                let relations = tx_db.open_tree("relations")?;
                let from_index = tx_db.open_tree("idx_from")?;
                let to_index = tx_db.open_tree("idx_to")?;

                // Get relation to know from/to for index cleanup
                let rel_data = relations.get(id_bytes)?.ok_or(sled::Error::Conflict)?;
                let relation: Relation = serde_json::from_slice(&rel_data)
                    .map_err(|_| sled::Error::Compression)?;

                // Remove relation
                relations.remove(id_bytes)?;

                // Remove from from_index
                let from_key = format!("from:{}:{}", relation.from_id, relation.id);
                from_index.remove(from_key.as_bytes())?;

                // Remove from to_index
                let to_key = format!("to:{}:{}", relation.to_id, relation.id);                to_index.remove(to_key.as_bytes())?;

                Ok(())
            })
            .map_err(|e| MemoryError::ProviderError(format!("delete relation transaction failed: {}", e)))?;

        debug!(relation_id = %id, "relation deleted");
        Ok(true)
    }

    /// Extracts a subgraph rooted at `root` up to `max_depth` hops.
    ///
    /// # Arguments
    /// * `root` - Starting entity ID for traversal
    /// * `max_depth` - Maximum number of relation hops to traverse
    /// * `relation_types` - Optional filter: only traverse relations of these types
    ///
    /// # Returns
    /// Tuple of (entities, relations) comprising the subgraph.
    #[instrument(skip(self), fields(root = %root, max_depth))]
    pub fn subgraph(
        &self,
        root: Uuid,
        max_depth: usize,
        relation_types: Option<&[&str]>,
    ) -> Result<(Vec<Entity>, Vec<Relation>)> {
        if max_depth == 0 {
            // Return only the root entity
            return match self.get_entity(root)? {
                Some(e) => Ok((vec![e], Vec::new())),
                None => Ok((Vec::new(), Vec::new())),
            };
        }

        let mut visited_entities = HashSet::new();
        let mut visited_relations = HashSet::new();
        let mut entities = Vec::new();
        let mut relations = Vec::new();

        // BFS queue: (entity_id, current_depth)
        let mut queue = VecDeque::new();
        queue.push_back((root, 0));
        visited_entities.insert(root);

        while let Some((current_id, depth)) = queue.pop_front() {
            // Add entity if not already included
            if !visited_entities.contains(&current_id) {
                continue;
            }
            if let Some(entity) = self.get_entity(current_id)? {
                if !entities.iter().any(|e| e.id == entity.id) {
                    entities.push(entity);
                }
            }

            // Stop traversing if at max depth
            if depth >= max_depth {
                continue;
            }

            // Traverse outgoing relations
            for rel in self.get_relations_from(current_id)? {
                if visited_relations.contains(&rel.id) {
                    continue;
                }

                // Filter by relation type if specified
                if let Some(types) = relation_types {
                    if !types.contains(&rel.relation_type.as_str()) {
                        continue;
                    }
                }

                visited_relations.insert(rel.id);
                if !relations.iter().any(|r| r.id == rel.id) {
                    relations.push(rel.clone());
                }

                // Queue target entity if not visited
                if !visited_entities.contains(&rel.to_id) {
                    visited_entities.insert(rel.to_id);
                    queue.push_back((rel.to_id, depth + 1));
                }
            }

            // Also traverse incoming relations for bidirectional exploration
            for rel in self.get_relations_to(current_id)? {
                if visited_relations.contains(&rel.id) {
                    continue;
                }

                if let Some(types) = relation_types {
                    if !types.contains(&rel.relation_type.as_str()) {
                        continue;
                    }
                }

                visited_relations.insert(rel.id);
                if !relations.iter().any(|r| r.id == rel.id) {                    relations.push(rel.clone());
                }

                // Queue source entity if not visited
                if !visited_entities.contains(&rel.from_id) {
                    visited_entities.insert(rel.from_id);
                    queue.push_back((rel.from_id, depth + 1));
                }
            }
        }

        debug!(
            root = %root,
            max_depth,
            entity_count = entities.len(),
            relation_count = relations.len(),
            "subgraph extracted"
        );
        Ok((entities, relations))
    }

    /// Merges new properties into an existing entity, preserving unchanged fields.
    ///
    /// # Arguments
    /// * `id` - Entity to update
    /// * `new_properties` - Properties to merge (only specified keys are updated)
    ///
    /// # Returns
    /// The updated entity with incremented version.
    #[instrument(skip(self, new_properties), fields(entity_id = %id))]
    pub fn merge_entity(
        &self,
        id: Uuid,
        new_properties: HashMap<String, serde_json::Value>,
    ) -> Result<Entity> {
        let mut entity = self
            .get_entity(id)?
            .ok_or_else(|| MemoryError::ProviderError("entity not found for merge".into()))?;

        // Merge properties
        for (key, value) in new_properties {
            entity.properties.insert(key, value);
        }

        entity.updated_at = Utc::now();
        entity.version += 1;

        // Persist update
        self.put_entity(entity.clone())?;
        debug!(entity_id = %id, version = entity.version, "entity merged");
        Ok(entity)
    }

    /// Flushes all pending writes to durable storage.
    ///
    /// Call this periodically or before shutdown to ensure data durability.
    pub fn flush(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| MemoryError::ProviderError(format!("flush failed: {}", e)))?;
        debug!("procedural memory flushed to disk");
        Ok(())
    }

    /// Returns the number of entities in the knowledge graph.
    pub fn entity_count(&self) -> Result<u64> {
        Ok(self.entities.len() as u64)
    }

    /// Returns the number of relations in the knowledge graph.
    pub fn relation_count(&self) -> Result<u64> {
        Ok(self.relations.len() as u64)
    }

    /// Returns the underlying sled database for advanced operations.
    ///
    /// ⚠️ Use with caution: direct access bypasses the type-safe API.
    pub fn db(&self) -> &Db {
        &self.db
    }

    /// Clears all data from the knowledge graph.
    ///
    /// ⚠️ Destructive: use only in tests or during reinitialization.
    pub fn clear(&self) -> Result<()> {
        self.entities.clear()?;
        self.relations.clear()?;
        self.kind_index.clear()?;
        self.from_index.clear()?;
        self.to_index.clear()?;
        self.name_index.clear()?;
        self.flush()?;
        debug!("procedural memory cleared");
        Ok(())
    }
}

// =============================================================================
// Tests// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_store() -> (ProceduralStore, TempDir) {
        let tmp = TempDir::new().unwrap();
        let store = ProceduralStore::new(tmp.path().to_str().unwrap()).unwrap();
        (store, tmp)
    }

    #[test]
    fn test_entity_lifecycle() {
        let (store, _tmp) = test_store();

        let mut entity = Entity::new("person", "Alice");
        entity = entity.with_property("age", serde_json::json!(30));
        entity = entity.with_property("role", serde_json::json!("engineer"));

        store.put_entity(entity.clone()).unwrap();

        // Retrieve by ID
        let retrieved = store.get_entity(entity.id).unwrap().unwrap();
        assert_eq!(retrieved.name, "Alice");
        assert_eq!(retrieved.kind, "person");
        assert_eq!(retrieved.get_property("age"), Some(&serde_json::json!(30)));

        // Retrieve by name
        let by_name = store.get_entity_by_name("Alice").unwrap().unwrap();
        assert_eq!(by_name.id, entity.id);

        // Find by kind
        let by_kind = store.find_entities_by_kind("person").unwrap();
        assert_eq!(by_kind.len(), 1);
        assert_eq!(by_kind[0].id, entity.id);

        // Merge properties
        let mut updates = HashMap::new();
        updates.insert("location".to_string(), serde_json::json!("SF"));
        updates.insert("age".to_string(), serde_json::json!(31)); // update existing

        let merged = store.merge_entity(entity.id, updates).unwrap();
        assert_eq!(merged.version, 2);
        assert_eq!(merged.get_property("age"), Some(&serde_json::json!(31)));
        assert_eq!(merged.get_property("location"), Some(&serde_json::json!("SF")));
        assert_eq!(merged.get_property("role"), Some(&serde_json::json!("engineer"))); // preserved
    }
    #[test]
    fn test_relation_lifecycle() {
        let (store, _tmp) = test_store();

        let alice = Entity::new("person", "Alice");
        let acme = Entity::new("company", "Acme Corp");
        store.put_entity(alice.clone()).unwrap();
        store.put_entity(acme.clone()).unwrap();

        let rel = Relation::new(alice.id, acme.id, "works_at", 0.95)
            .with_property("since", serde_json::json!("2020"));
        store.put_relation(rel.clone()).unwrap();

        // Get relations from Alice
        let from_alice = store.get_relations_from(alice.id).unwrap();
        assert_eq!(from_alice.len(), 1);
        assert_eq!(from_alice[0].relation_type, "works_at");
        assert_eq!(from_alice[0].to_id, acme.id);

        // Get relations to Acme
        let to_acme = store.get_relations_to(acme.id).unwrap();
        assert_eq!(to_acme.len(), 1);
        assert_eq!(to_acme[0].from_id, alice.id);

        // Get relations between specific entities
        let between = store.get_relations_between(alice.id, acme.id).unwrap();
        assert_eq!(between.len(), 1);

        // Delete relation
        assert!(store.delete_relation(rel.id).unwrap());
        assert!(!store.delete_relation(rel.id).unwrap()); // already deleted
        assert!(store.get_relations_from(alice.id).unwrap().is_empty());
    }

    #[test]
    fn test_entity_delete_cascades() {
        let (store, _tmp) = test_store();

        let a = Entity::new("person", "Alice");
        let b = Entity::new("person", "Bob");
        let c = Entity::new("company", "Acme");
        store.put_entity(a.clone()).unwrap();
        store.put_entity(b.clone()).unwrap();
        store.put_entity(c.clone()).unwrap();

        // Create relations: Alice -> Acme, Bob -> Acme, Alice -> Bob
        store.put_relation(Relation::new(a.id, c.id, "works_at", 1.0)).unwrap();
        store.put_relation(Relation::new(b.id, c.id, "works_at", 1.0)).unwrap();
        store.put_relation(Relation::new(a.id, b.id, "knows", 0.8)).unwrap();
        // Delete Alice - should cascade to her relations
        assert!(store.delete_entity(a.id).unwrap());

        // Alice's entity is gone
        assert!(store.get_entity(a.id).unwrap().is_none());

        // Alice's outgoing relations are gone
        assert!(store.get_relations_from(a.id).unwrap().is_empty());
        assert!(store.get_relations_between(a.id, c.id).unwrap().is_empty());

        // Bob and Acme still exist, and Bob->Acme relation remains
        assert!(store.get_entity(b.id).unwrap().is_some());
        assert!(store.get_entity(c.id).unwrap().is_some());
        assert_eq!(store.get_relations_from(b.id).unwrap().len(), 1);
    }

    #[test]
    fn test_subgraph_traversal() {
        let (store, _tmp) = test_store();

        // Build a small graph: A -> B -> C, A -> D
        let a = Entity::new("root", "A");
        let b = Entity::new("node", "B");
        let c = Entity::new("node", "C");
        let d = Entity::new("node", "D");
        for e in [&a, &b, &c, &d] {
            store.put_entity(e.clone()).unwrap();
        }

        store.put_relation(Relation::new(a.id, b.id, "links_to", 1.0)).unwrap();
        store.put_relation(Relation::new(b.id, c.id, "links_to", 1.0)).unwrap();
        store.put_relation(Relation::new(a.id, d.id, "links_to", 1.0)).unwrap();

        // Depth 1: should get A, B, D and their connecting relations
        let (ents, rels) = store.subgraph(a.id, 1, None).unwrap();
        assert!(ents.iter().any(|e| e.id == a.id));
        assert!(ents.iter().any(|e| e.id == b.id));
        assert!(ents.iter().any(|e| e.id == d.id));
        assert!(!ents.iter().any(|e| e.id == c.id)); // depth 2, not included
        assert_eq!(rels.len(), 2); // A->B, A->D

        // Depth 2: should include C
        let (ents, rels) = store.subgraph(a.id, 2, None).unwrap();
        assert!(ents.iter().any(|e| e.id == c.id));
        assert_eq!(rels.len(), 3); // all three relations

        // Filter by relation type
        store.put_relation(Relation::new(a.id, b.id, "also_links", 0.5)).unwrap();
        let (ents, rels) = store.subgraph(a.id, 1, Some(&["links_to"])).unwrap();
        assert_eq!(rels.iter().filter(|r| r.relation_type == "links_to").count(), 2);        assert_eq!(rels.iter().filter(|r| r.relation_type == "also_links").count(), 0);
    }

    #[test]
    fn test_flush_and_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap();

        // Create and populate
        {
            let store = ProceduralStore::new(path).unwrap();
            let entity = Entity::new("test", "Persistent");
            store.put_entity(entity.clone()).unwrap();
            store.flush().unwrap();
        }

        // Reopen and verify data persists
        {
            let store = ProceduralStore::new(path).unwrap();
            let retrieved = store.get_entity_by_name("Persistent").unwrap().unwrap();
            assert_eq!(retrieved.kind, "test");
        }
    }

    #[test]
    fn test_concurrent_access() {
        let (store, _tmp) = test_store();
        let mut handles = vec![];

        // Spawn writers
        for i in 0..10 {
            let store = store.db.clone(); // sled::Db is cloneable
            handles.push(std::thread::spawn(move || {
                let db = ProceduralStore {
                    db: store.clone(),
                    entities: store.open_tree("entities").unwrap(),
                    relations: store.open_tree("relations").unwrap(),
                    kind_index: store.open_tree("idx_kind").unwrap(),
                    from_index: store.open_tree("idx_from").unwrap(),
                    to_index: store.open_tree("idx_to").unwrap(),
                    name_index: store.open_tree("idx_name").unwrap(),
                };
                let entity = Entity::new("concurrent", format!("item-{}", i));
                db.put_entity(entity).unwrap();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
        // Verify all were written
        let items = store.find_entities_by_kind("concurrent").unwrap();
        assert_eq!(items.len(), 10);
    }
}
