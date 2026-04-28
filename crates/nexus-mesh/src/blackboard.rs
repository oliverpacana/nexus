use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{debug, instrument, warn};
use nexus_proto::agent::AgentId;

use crate::crdt::{LamportClock, LWWMap, ObservedRemoveSet};

// =============================================================================
// Blackboard Entry & Scope
// =============================================================================

/// A single entry posted to the distributed blackboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackboardEntry {
    /// The payload value stored at this key.
    pub value: serde_json::Value,

    /// The agent that authored this entry.
    pub author_id: AgentId,

    /// The mesh node ID that initially wrote this entry.
    pub node_id: String,

    /// Visibility/lifetime scope of the entry.
    pub scope: BlackboardScope,

    /// Optional expiration timestamp; entry is evicted after this time.
    pub expires_at: Option<DateTime<Utc>>,

    /// Arbitrary tags for filtering and discovery.
    pub tags: Vec<String>,
}

/// Defines the visibility and persistence scope of a blackboard entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BlackboardScope {
    /// Only visible on the local node; not replicated over the mesh.
    Local,
    /// Replicated to all nodes in the current mesh cluster.
    Cluster,
    /// Persisted across mesh reboots and reconnections.
    Persistent,
}

// =============================================================================
// Blackboard Change Events
// =============================================================================
/// Event emitted when the blackboard state changes.
#[derive(Debug, Clone)]
pub enum BlackboardChange {
    /// A key was set or updated.
    Set {
        key: String,
        value: serde_json::Value,
        node_id: String,
    },
    /// A key was logically deleted (tombstoned).
    Deleted {
        key: String,
        node_id: String,
    },
}

// =============================================================================
// Blackboard — Distributed Shared State
// =============================================================================

/// The in-node shared blackboard backed by CRDTs for coordination-free replication.
///
/// # Design
/// - `state`: LWW-Register Map providing conflict-free concurrent reads/writes
/// - `clock`: Lamport logical clock for causal ordering and delta sync
/// - `capability_members`: Tracks which capabilities exist across the mesh
/// - `change_tx`: Broadcast channel for local reaction to state changes
///
/// # Thread Safety
/// All mutable state is protected by `Arc<RwLock>` or `Arc<Mutex>`.
/// Public methods are `async` and carefully scope lock lifetimes to avoid
/// holding synchronization primitives across `.await` points or blocking calls.
pub struct Blackboard {
    /// CRDT-backed key-value store.
    pub state: Arc<RwLock<LWWMap>>,

    /// Logical clock for event ordering and sync boundaries.
    pub clock: Arc<Mutex<LamportClock>>,

    /// Unique identifier for this mesh node.
    pub node_id: String,

    /// Set of capabilities announced by nodes in the mesh.
    pub capability_members: Arc<RwLock<ObservedRemoveSet<String>>>,

    /// Broadcast channel for state change notifications.
    change_tx: broadcast::Sender<BlackboardChange>,
}
impl Blackboard {
    /// Creates a new blackboard instance for a mesh node.
    ///
    /// # Returns
    /// A tuple of `(Blackboard, broadcast::Receiver<BlackboardChange>)`.
    /// The receiver can be used to react to local or remote state changes.
    pub fn new(node_id: String) -> (Self, broadcast::Receiver<BlackboardChange>) {
        let (tx, rx) = broadcast::channel(512);
        let clock = LamportClock::new(node_id.clone());
        let state = LWWMap::new();
        let capability_members = ObservedRemoveSet::new();

        let blackboard = Self {
            state: Arc::new(RwLock::new(state)),
            clock: Arc::new(Mutex::new(clock)),
            node_id,
            capability_members: Arc::new(RwLock::new(capability_members)),
            change_tx: tx,
        };

        debug!(node_id = %blackboard.node_id, "blackboard initialized");
        (blackboard, rx)
    }

    /// Subscribes to blackboard change events.
    pub fn subscribe(&self) -> broadcast::Receiver<BlackboardChange> {
        self.change_tx.subscribe()
    }

    /// Sets or updates a key in the blackboard.
    ///
    /// Ticks the logical clock, inserts the serialized entry into the LWWMap,
    /// and broadcasts a `Set` event. Returns the new logical timestamp.
    #[instrument(skip(self, entry), fields(key = %key))]
    pub async fn set(&self, key: String, entry: BlackboardEntry) -> u64 {
        let ts = {
            let mut clock = self.clock.lock().await;
            clock.tick()
        };

        let value = serde_json::to_value(&entry).unwrap_or(serde_json::Value::Null);
        self.state.write().await.set(key.clone(), value.clone(), ts, &self.node_id);

        let _ = self.change_tx.send(BlackboardChange::Set {
            key,
            value,
            node_id: self.node_id.clone(),
        });

        ts    }

    /// Logically deletes a key by tombstoning it.
    #[instrument(skip(self), fields(key = %key))]
    pub async fn delete(&self, key: &str) {
        let ts = {
            let mut clock = self.clock.lock().await;
            clock.tick()
        };

        self.state.write().await.delete(key, ts, &self.node_id);

        let _ = self.change_tx.send(BlackboardChange::Deleted {
            key: key.to_string(),
            node_id: self.node_id.clone(),
        });
    }

    /// Retrieves the current value for a key, if present and not tombstoned.
    pub async fn get(&self, key: &str) -> Option<serde_json::Value> {
        self.state.read().await.get(key).cloned()
    }

    /// Returns all active keys that start with the given prefix.
    pub async fn keys_with_prefix(&self, prefix: &str) -> Vec<String> {
        let state = self.state.read().await;
        state.keys()
            .filter(|k| k.starts_with(prefix))
            .map(String::from)
            .collect()
    }

    /// Merges a remote delta into the local state and updates the logical clock.
    /// Broadcasts `BlackboardChange` events only for keys that actually changed.
    #[instrument(skip(self, delta), fields(remote_clock = remote_clock))]
    pub async fn merge_delta(&self, delta: LWWMap, remote_clock: u64) {
        // 1. Update logical clock (max(local, remote) + 1)
        {
            let mut clock = self.clock.lock().await;
            clock.update(remote_clock);
        }

        // 2. Compute changes that will actually apply, then merge
        let mut changes = Vec::new();
        {
            let mut state = self.state.write().await;

            for (key, lww_val) in &delta.entries {
                // Check if this remote entry supersedes the local entry
                let should_update = state.entries.get(key).map_or(true, |existing| {                    lww_val.timestamp > existing.timestamp ||
                    (lww_val.timestamp == existing.timestamp && lww_val.node_id > existing.node_id)
                });

                if should_update {
                    if let Some(entry_json) = lww_val.value.as_object() {
                        let node = entry_json.get("node_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("remote");
                        changes.push(BlackboardChange::Set {
                            key: key.clone(),
                            value: lww_val.value.clone(),
                            node_id: node.to_string(),
                        });
                    }
                }
            }

            for key in delta.tombstones.keys() {
                // Only broadcast delete if key actually exists locally
                if state.contains_key(key) {
                    changes.push(BlackboardChange::Deleted {
                        key: key.clone(),
                        node_id: "remote".into(),
                    });
                }
            }

            // Apply CRDT merge (idempotent, commutative, associative)
            state.merge(&delta);
        }

        // 3. Broadcast changes outside the lock
        for change in changes {
            let _ = self.change_tx.send(change);
        }

        debug!(changes = changes.len(), "remote delta merged");
    }

    /// Returns the complete current state for initial synchronization.
    pub async fn full_state(&self) -> LWWMap {
        self.state.read().await.clone()
    }

    /// Returns a delta containing only entries/tombstones created after `timestamp`.
    /// Used for efficient incremental sync between mesh nodes.
    pub async fn delta_since(&self, timestamp: u64) -> LWWMap {
        self.state.read().await.delta_since(timestamp)
    }
    /// Announces that this node supports a specific capability.
    /// Updates the capability set and publishes to the blackboard for discovery.
    #[instrument(skip(self), fields(capability = %capability))]
    pub async fn announce_capability(&self, capability: String) {
        // Tick clock for local write
        let ts = {
            let mut clock = self.clock.lock().await;
            clock.tick()
        };

        // Update local capability tracking set
        self.capability_members.write().await.add(capability.clone(), ts);

        // Publish to blackboard for cross-node discovery
        let key = format!("capabilities::{}::{}", self.node_id, capability);
        let entry = BlackboardEntry {
            value: serde_json::json!({ "capability": capability }),
            author_id: AgentId::nil(),
            node_id: self.node_id.clone(),
            scope: BlackboardScope::Cluster,
            expires_at: None,
            tags: vec!["capability".into()],
        };

        self.set(key, entry).await;
    }

    /// Returns list of node IDs that have announced a specific capability.
    pub async fn nodes_with_capability(&self, cap: &str) -> Vec<String> {
        let prefix = "capabilities::";
        let suffix = format!("::{}", cap);
        let keys = self.keys_with_prefix(prefix).await;

        let mut nodes = Vec::new();
        for key in keys {
            if key.ends_with(&suffix) {
                // Format: capabilities::{node_id}::{cap}
                let parts: Vec<&str> = key.split("::").collect();
                if parts.len() >= 3 {
                    nodes.push(parts[1].to_string());
                }
            }
        }
        nodes
    }

    /// Scans the blackboard and evicts entries that have passed their expiration time.
    #[instrument(skip(self))]
    pub async fn evict_expired(&self) {        let now = Utc::now();
        
        // Identify expired keys under read lock
        let expired_keys = {
            let state = self.state.read().await;
            let keys: Vec<String> = state.keys().map(String::from).collect();
            let mut expired = Vec::new();
            
            for key in &keys {
                if let Some(val) = state.get(key) {
                    if let Ok(entry) = serde_json::from_value::<BlackboardEntry>(val.clone()) {
                        if let Some(exp) = entry.expires_at {
                            if exp < now {
                                expired.push(key.clone());
                            }
                        }
                    }
                }
            }
            expired
        };

        // Delete expired keys under write lock
        for key in expired_keys {
            debug!(key = %key, "evicting expired entry");
            self.delete(&key).await;
        }
    }
}
