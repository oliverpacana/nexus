use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use nexus_proto::agent::AgentId;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::embeddings::MemoryError;

pub type Result<T> = std::result::Result<T, MemoryError>;

// =============================================================================
// WorkingEntry — A Value in L1 Working Memory
// =============================================================================

/// A single entry in an agent's working memory (L1 tier).
/// Tracks access patterns for LRU eviction and observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingEntry {
    /// The stored JSON value.
    pub value: serde_json::Value,

    /// Timestamp when this entry was first created.
    pub created_at: Instant,

    /// Timestamp of the last modification.
    pub updated_at: Instant,

    /// Count of times this entry has been accessed (get/set).
    pub access_count: u64,

    /// Timestamp of the most recent access (for LRU ordering).
    pub last_accessed: Instant,
}

impl WorkingEntry {
    /// Creates a new entry with the given value.
    pub fn new(value: serde_json::Value) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            updated_at: now,
            access_count: 0,
            last_accessed: now,
        }
    }
    /// Updates access metadata: increments count and refreshes timestamp.
    /// Call this on every get/set to maintain accurate LRU ordering.
    pub fn touch(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
        self.last_accessed = Instant::now();
    }

    /// Returns the age of this entry in seconds.
    pub fn age_secs(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }

    /// Returns time since last access in seconds.
    pub fn idle_secs(&self) -> u64 {
        self.last_accessed.elapsed().as_secs()
    }
}

// =============================================================================
// WorkingMemory — L1 In-Process Per-Agent Memory
// =============================================================================

/// L1 working memory: fast, in-process, per-agent key-value store.
///
/// # Design
/// - Backed by `Arc<RwLock<HashMap>>` for async-safe concurrent access
/// - LRU eviction via `VecDeque` tracking access order
/// - Enforces `max_entries` limit to bound memory usage
/// - Entries track access patterns for observability and eviction decisions
///
/// # Thread Safety
/// - All public methods are `async` and acquire the RwLock appropriately
/// - Read operations use `read()` lock; writes use `write()` lock
/// - LRU queue is updated atomically with the HashMap under the same lock
pub struct WorkingMemory {
    /// The underlying store: key → entry.
    store: Arc<RwLock<HashMap<String, WorkingEntry>>>,

    /// LRU queue: front = most recently used, back = least recently used.
    lru_order: Arc<RwLock<VecDeque<String>>>,

    /// Maximum number of entries allowed before LRU eviction triggers.
    max_entries: usize,

    /// The agent ID that owns this working memory instance.
    owner_id: AgentId,
}

impl WorkingMemory {    /// Creates a new working memory instance for an agent.
    ///
    /// # Arguments
    /// * `owner_id` - The agent that owns this memory (for observability)
    /// * `max_entries` - Maximum entries before LRU eviction (recommended: 64-256)
    pub fn new(owner_id: AgentId, max_entries: usize) -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::with_capacity(max_entries))),
            lru_order: Arc::new(RwLock::new(VecDeque::with_capacity(max_entries))),
            max_entries,
            owner_id,
        }
    }

    /// Retrieves a clone of the entry for the given key, if present.
    /// Updates access metadata (LRU position) on hit.
    pub async fn get(&self, key: &str) -> Option<WorkingEntry> {
        let mut store = self.store.write().await;
        let mut lru = self.lru_order.write().await;

        if let Some(entry) = store.get_mut(key) {
            // Touch the entry to update LRU metadata
            entry.touch();

            // Update LRU order: remove and re-insert at front
            if let Some(pos) = lru.iter().position(|k| k == key) {
                lru.remove(pos);
            }
            lru.push_front(key.to_string());

            Some(entry.clone())
        } else {
            None
        }
    }

    /// Stores or updates a value under the given key.
    ///
    /// If the store is at capacity, evicts the least-recently-used entry first.
    /// Returns `Ok(())` on success, or `MemoryError` if eviction fails (shouldn't happen).
    pub async fn set(&self, key: impl Into<String>, value: serde_json::Value) -> Result<()> {
        let key = key.into();
        let mut store = self.store.write().await;
        let mut lru = self.lru_order.write().await;

        let is_new = !store.contains_key(&key);

        // If new entry and at capacity, evict LRU first
        if is_new && store.len() >= self.max_entries {
            self.evict_lru(&mut store, &mut lru).await?;        }

        let now = Instant::now();
        let entry = if let Some(existing) = store.get_mut(&key) {
            // Update existing entry
            existing.value = value;
            existing.updated_at = now;
            existing.touch();
            existing.clone()
        } else {
            // Insert new entry
            let entry = WorkingEntry::new(value);
            store.insert(key.clone(), entry.clone());
            entry
        };

        // Update LRU order: ensure key is at front
        if let Some(pos) = lru.iter().position(|k| k == &key) {
            lru.remove(pos);
        }
        lru.push_front(key);

        debug!(
            agent_id = %self.owner_id,
            key = %key,
            is_new,
            store_size = store.len(),
            "working memory set"
        );

        Ok(())
    }

    /// Removes an entry by key.
    /// Returns `true` if the key existed and was removed.
    pub async fn delete(&self, key: &str) -> bool {
        let mut store = self.store.write().await;
        let mut lru = self.lru_order.write().await;

        if store.remove(key).is_some() {
            // Also remove from LRU queue
            if let Some(pos) = lru.iter().position(|k| k == key) {
                lru.remove(pos);
            }
            debug!(agent_id = %self.owner_id, key = %key, "working memory delete");
            true
        } else {
            false
        }
    }
    /// Returns `true` if the key exists in working memory.
    pub async fn contains(&self, key: &str) -> bool {
        let store = self.store.read().await;
        store.contains_key(key)
    }

    /// Returns a vector of all keys currently in working memory.
    /// Order is arbitrary (HashMap iteration order).
    pub async fn all_keys(&self) -> Vec<String> {
        let store = self.store.read().await;
        store.keys().cloned().collect()
    }

    /// Returns a full snapshot of the current working memory state.
    /// Useful for persisting to L2 episodic memory or debugging.
    ///
    /// Note: This clones all values; use judiciously for large stores.
    pub async fn snapshot(&self) -> HashMap<String, WorkingEntry> {
        let store = self.store.read().await;
        store.clone()
    }

    /// Clears all entries from working memory.
    pub async fn clear(&self) {
        let mut store = self.store.write().await;
        let mut lru = self.lru_order.write().await;
        store.clear();
        lru.clear();
        debug!(agent_id = %self.owner_id, "working memory cleared");
    }

    /// Returns the current number of entries in working memory.
    pub async fn len(&self) -> usize {
        let store = self.store.read().await;
        store.len()
    }

    /// Returns `true` if working memory is empty.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Returns the configured maximum entry limit.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Returns the owner agent ID.
    pub fn owner_id(&self) -> AgentId {        self.owner_id
    }

    /// Evicts the least-recently-used entry from the store.
    /// Must be called with both `store` and `lru` write-locks held.
    async fn evict_lru(
        &self,
        store: &mut HashMap<String, WorkingEntry>,
        lru: &mut VecDeque<String>,
    ) -> Result<()> {
        // LRU queue: back = least recently used
        if let Some(evict_key) = lru.pop_back() {
            if store.remove(&evict_key).is_some() {
                warn!(
                    agent_id = %self.owner_id,
                    key = %evict_key,
                    store_size = store.len(),
                    "working memory LRU eviction"
                );
            }
        }
        Ok(())
    }

    /// Returns statistics about the current working memory state.
    pub async fn stats(&self) -> WorkingMemoryStats {
        let store = self.store.read().await;
        let lru = self.lru_order.read().await;

        let (oldest_age_secs, newest_age_secs, total_accesses) = store.values().fold(
            (u64::MAX, 0u64, 0u64),
            |(oldest, newest, total), entry| {
                let age = entry.age_secs();
                (
                    oldest.min(age),
                    newest.max(age),
                    total + entry.access_count,
                )
            },
        );

        WorkingMemoryStats {
            entry_count: store.len(),
            max_entries: self.max_entries,
            utilization: store.len() as f64 / self.max_entries as f64,
            oldest_entry_age_secs: if oldest_age_secs == u64::MAX {
                0
            } else {
                oldest_age_secs
            },            newest_entry_age_secs: newest_age_secs,
            total_accesses,
            lru_order_snapshot: lru.iter().take(10).cloned().collect(),
        }
    }
}

// =============================================================================
// WorkingMemoryStats — Observability Snapshot
// =============================================================================

/// Statistics about the current state of working memory.
/// Used for observability, debugging, and the TUI dashboard.
#[derive(Debug, Clone)]
pub struct WorkingMemoryStats {
    /// Current number of entries.
    pub entry_count: usize,

    /// Configured maximum entries.
    pub max_entries: usize,

    /// Utilization ratio: entry_count / max_entries.
    pub utilization: f64,

    /// Age in seconds of the oldest entry (by creation time).
    pub oldest_entry_age_secs: u64,

    /// Age in seconds of the newest entry.
    pub newest_entry_age_secs: u64,

    /// Total access count across all entries.
    pub total_accesses: u64,

    /// First 10 keys in LRU order (most → least recently used).
    pub lru_order_snapshot: Vec<String>,
}

// =============================================================================
// WorkingMemorySnapshot — Serializable State for L2 Persistence
// =============================================================================

/// A serializable snapshot of working memory state.
/// Used to persist L1 state to L2 episodic memory or for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemorySnapshot {
    /// The agent ID that owns this memory.
    pub owner_id: AgentId,

    /// Map of key → value (stripped of access metadata for compactness).
    pub entries: HashMap<String, serde_json::Value>,
    /// Timestamp when this snapshot was taken.
    pub snapshot_at: DateTime<Utc>,
}

impl WorkingMemorySnapshot {
    /// Creates a new snapshot from the given data.
    pub fn new(owner_id: AgentId, entries: HashMap<String, serde_json::Value>) -> Self {
        Self {
            owner_id,
            entries,
            snapshot_at: Utc::now(),
        }
    }

    /// Converts a `WorkingMemory` instance into a snapshot.
    pub async fn from_working_memory(mem: &WorkingMemory) -> Self {
        let entries = mem
            .snapshot()
            .await
            .into_iter()
            .map(|(k, v)| (k, v.value))
            .collect();

        Self::new(mem.owner_id(), entries)
    }

    /// Returns the number of entries in this snapshot.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the snapshot is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_agent_id() -> AgentId {
        AgentId::new()
    }
    #[tokio::test]
    async fn test_working_memory_basic() {
        let mem = WorkingMemory::new(test_agent_id(), 10);

        // Initially empty
        assert!(mem.is_empty().await);
        assert_eq!(mem.len().await, 0);
        assert!(!mem.contains("foo").await);

        // Set a value
        mem.set("foo", json!("bar")).await.unwrap();
        assert!(mem.contains("foo").await);
        assert_eq!(mem.len().await, 1);

        // Get the value
        let entry = mem.get("foo").await.unwrap();
        assert_eq!(entry.value, json!("bar"));
        assert_eq!(entry.access_count, 1);

        // Update the value
        mem.set("foo", json!("baz")).await.unwrap();
        let entry = mem.get("foo").await.unwrap();
        assert_eq!(entry.value, json!("baz"));
        assert_eq!(entry.access_count, 2); // touched again

        // Delete
        assert!(mem.delete("foo").await);
        assert!(!mem.contains("foo").await);
        assert!(mem.is_empty().await);

        // Delete non-existent
        assert!(!mem.delete("foo").await);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let mem = WorkingMemory::new(test_agent_id(), 3);

        // Fill to capacity
        mem.set("a", json!(1)).await.unwrap();
        mem.set("b", json!(2)).await.unwrap();
        mem.set("c", json!(3)).await.unwrap();
        assert_eq!(mem.len().await, 3);

        // Access "a" to make it recently used
        let _ = mem.get("a").await;

        // Add "d" - should evict "b" (LRU: c was accessed after b, a was accessed most recently)
        mem.set("d", json!(4)).await.unwrap();
        assert!(!mem.contains("b").await); // evicted
        assert!(mem.contains("a").await);
        assert!(mem.contains("c").await);
        assert!(mem.contains("d").await);
        assert_eq!(mem.len().await, 3);

        // Verify LRU order: d (newest), a (accessed), c (oldest)
        let stats = mem.stats().await;
        // The snapshot shows most-recently-used first
        assert_eq!(stats.lru_order_snapshot.first(), Some(&"d".to_string()));
    }

    #[tokio::test]
    async fn test_lru_order_updates_on_access() {
        let mem = WorkingMemory::new(test_agent_id(), 3);

        mem.set("x", json!(1)).await.unwrap();
        mem.set("y", json!(2)).await.unwrap();
        mem.set("z", json!(3)).await.unwrap();

        // LRU order: z, y, x (x is least recently used)

        // Access x to make it most recent
        let _ = mem.get("x").await;

        // Now add "w" - should evict y (now LRU)
        mem.set("w", json!(4)).await.unwrap();

        assert!(!mem.contains("y").await);
        assert!(mem.contains("x").await);
        assert!(mem.contains("z").await);
        assert!(mem.contains("w").await);
    }

    #[tokio::test]
    async fn test_snapshot() {
        let mem = WorkingMemory::new(test_agent_id(), 10);

        mem.set("key1", json!("value1")).await.unwrap();
        mem.set("key2", json!(42)).await.unwrap();
        mem.set("key3", json!({"nested": true})).await.unwrap();

        let snapshot = WorkingMemorySnapshot::from_working_memory(&mem).await;

        assert_eq!(snapshot.owner_id, mem.owner_id());
        assert_eq!(snapshot.len(), 3);
        assert_eq!(snapshot.entries.get("key1"), Some(&json!("value1")));
        assert_eq!(snapshot.entries.get("key2"), Some(&json!(42)));
        assert_eq!(snapshot.entries.get("key3"), Some(&json!({"nested": true})));        assert!(snapshot.snapshot_at <= Utc::now());
    }

    #[tokio::test]
    async fn test_all_keys() {
        let mem = WorkingMemory::new(test_agent_id(), 10);

        mem.set("alpha", json!(1)).await.unwrap();
        mem.set("beta", json!(2)).await.unwrap();
        mem.set("gamma", json!(3)).await.unwrap();

        let keys = mem.all_keys().await;
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"alpha".to_string()));
        assert!(keys.contains(&"beta".to_string()));
        assert!(keys.contains(&"gamma".to_string()));
    }

    #[tokio::test]
    async fn test_clear() {
        let mem = WorkingMemory::new(test_agent_id(), 10);

        for i in 0..5 {
            mem.set(format!("key{}", i), json!(i)).await.unwrap();
        }
        assert_eq!(mem.len().await, 5);

        mem.clear().await;
        assert!(mem.is_empty().await);
        assert_eq!(mem.len().await, 0);
        assert!(mem.all_keys().await.is_empty());
    }

    #[tokio::test]
    async fn test_stats() {
        let mem = WorkingMemory::new(test_agent_id(), 10);

        // Initially empty stats
        let stats = mem.stats().await;
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.max_entries, 10);
        assert!((stats.utilization - 0.0).abs() < f64::EPSILON);

        // Add some entries and access them
        mem.set("a", json!(1)).await.unwrap();
        mem.set("b", json!(2)).await.unwrap();
        let _ = mem.get("a").await; // touch "a"

        let stats = mem.stats().await;
        assert_eq!(stats.entry_count, 2);        assert!((stats.utilization - 0.2).abs() < f64::EPSILON);
        assert!(stats.total_accesses >= 3); // 2 sets + 1 get
        assert_eq!(stats.lru_order_snapshot.first(), Some(&"a".to_string()));
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let mem = Arc::new(WorkingMemory::new(test_agent_id(), 100));
        let mut handles = vec![];

        // Spawn many concurrent writers
        for i in 0..20 {
            let mem = Arc::clone(&mem);
            handles.push(tokio::spawn(async move {
                for j in 0..10 {
                    mem.set(format!("key-{}-{}", i, j), json!(i * j)).await.unwrap();
                }
            }));
        }

        // Spawn many concurrent readers
        for i in 0..20 {
            let mem = Arc::clone(&mem);
            handles.push(tokio::spawn(async move {
                for j in 0..10 {
                    let _ = mem.get(&format!("key-{}-{}", i, j)).await;
                }
            }));
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify final state is consistent
        let stats = mem.stats().await;
        assert!(stats.entry_count <= 100); // max_entries enforced
    }

    #[test]
    fn test_working_entry_metadata() {
        use std::thread::sleep;
        use std::time::Duration;

        let mut entry = WorkingEntry::new(json!("test"));
        assert_eq!(entry.access_count, 0);

        entry.touch();
        assert_eq!(entry.access_count, 1);
        sleep(Duration::from_millis(10));
        entry.touch();
        assert_eq!(entry.access_count, 2);
        assert!(entry.idle_secs() == 0); // just touched

        assert!(entry.age_secs() >= 0);
    }
}
