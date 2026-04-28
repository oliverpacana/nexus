use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

// =============================================================================
// Lamport Clock
// =============================================================================

/// A logical clock for causal ordering of events in a distributed system.
/// Monotonically increases with each tick or message reception.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LamportClock {
    pub node_id: String,
    pub counter: u64,
}

impl LamportClock {
    pub fn new(node_id: String) -> Self {
        Self { node_id, counter: 0 }
    }

    /// Increments the clock and returns the new value.
    /// Call before generating any local event.
    pub fn tick(&mut self) -> u64 {
        self.counter += 1;
        self.counter
    }

    /// Updates the clock upon receiving a message with a remote timestamp.
    /// Implements the Lamport update rule: max(self, received) + 1
    pub fn update(&mut self, received: u64) {
        self.counter = std::cmp::max(self.counter, received) + 1;
    }

    /// Returns the current logical timestamp.
    pub fn value(&self) -> u64 {
        self.counter
    }
}

// =============================================================================
// LWW Value
// =============================================================================

/// A Last-Write-Wins (LWW) register for any serializable type.
/// Conflicts are resolved by choosing the value with the highest timestamp.
/// Ties are broken lexicographically by `node_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LWWValue<T> {
    pub value: T,
    pub timestamp: u64,    pub node_id: String,
}

impl<T: PartialEq + std::fmt::Debug> LWWValue<T> {
    pub fn new(value: T, timestamp: u64, node_id: String) -> Self {
        Self { value, timestamp, node_id }
    }

    /// Merges two LWW registers.
    /// **CRDT Properties**:
    /// - Idempotent: `merge(a, a) == a`
    /// - Commutative: `merge(a, b) == merge(b, a)`
    /// - Associative: `merge(a, merge(b, c)) == merge(merge(a, b), c)`
    /// These hold because `max(timestamp)` with deterministic tie-breaking forms a semilattice.
    pub fn merge(&mut self, other: LWWValue<T>) {
        if other.timestamp > self.timestamp {
            self.timestamp = other.timestamp;
            self.node_id = other.node_id;
            self.value = other.value;
        } else if other.timestamp == self.timestamp && other.node_id > self.node_id {
            // Lexicographic tie-break ensures deterministic convergence
            self.node_id = other.node_id;
            self.value = other.value;
        }
    }
}

// =============================================================================
// LWW Map
// =============================================================================

/// A Last-Write-Wins map acting as the backbone of the shared blackboard.
/// Supports concurrent sets and deletes with eventual consistency.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LWWMap {
    /// Active key-value entries.
    pub entries: HashMap<String, LWWValue<serde_json::Value>>,
    /// Deletion markers: maps key to the timestamp of its deletion.
    pub tombstones: HashMap<String, u64>,
}

impl LWWMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets or updates a key with a new value and timestamp.
    pub fn set(&mut self, key: String, value: serde_json::Value, timestamp: u64, node_id: &str) {
        self.entries
            .entry(key)            .and_modify(|existing| {
                if timestamp > existing.timestamp
                    || (timestamp == existing.timestamp && node_id > &existing.node_id)
                {
                    existing.value = value.clone();
                    existing.timestamp = timestamp;
                    existing.node_id = node_id.to_string();
                }
            })
            .or_insert_with(|| LWWValue::new(value, timestamp, node_id.to_string()));
    }

    /// Marks a key as deleted at the given timestamp.
    /// The key is logically removed from the map for future reads.
    pub fn delete(&mut self, key: &str, timestamp: u64, node_id: &str) {
        self.tombstones
            .entry(key.to_string())
            .and_modify(|ts| {
                if timestamp > *ts {
                    *ts = timestamp;
                }
            })
            .or_insert(timestamp);
    }

    /// Retrieves the value for a key if it is present and not tombstoned.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        if self.contains_key(key) {
            self.entries.get(key).map(|v| &v.value)
        } else {
            None
        }
    }

    /// Checks if a key is present and not superseded by a tombstone.
    pub fn contains_key(&self, key: &str) -> bool {
        let entry_ts = self.entries.get(key).map(|v| v.timestamp).unwrap_or(0);
        let tombstone_ts = self.tombstones.get(key).copied().unwrap_or(0);
        entry_ts > tombstone_ts
    }

    /// Merges another map into this one.
    /// Applies LWW semantics independently per key and tombstone.
    pub fn merge(&mut self, other: &LWWMap) {
        // Merge entries
        for (k, v) in &other.entries {
            self.entries
                .entry(k.clone())
                .and_modify(|existing| {
                    if v.timestamp > existing.timestamp                        || (v.timestamp == existing.timestamp && v.node_id > existing.node_id)
                    {
                        existing.value = v.value.clone();
                        existing.timestamp = v.timestamp;
                        existing.node_id = v.node_id.clone();
                    }
                })
                .or_insert(v.clone());
        }

        // Merge tombstones
        for (k, ts) in &other.tombstones {
            self.tombstones
                .entry(k.clone())
                .and_modify(|t| {
                    if *ts > *t {
                        *t = *ts;
                    }
                })
                .or_insert(*ts);
        }

        // Compact: remove entries that are definitively superseded by tombstones
        self.entries.retain(|k, v| {
            self.tombstones.get(k).map_or(true, |&t_ts| v.timestamp > t_ts)
        });
    }

    /// Returns an iterator over all active keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().filter(|k| self.contains_key(k)).map(|k| k.as_str())
    }

    /// Returns the number of active keys.
    pub fn len(&self) -> usize {
        self.keys().count()
    }

    /// Returns a delta map containing only entries and tombstones created/updated after `timestamp`.
    /// Used for efficient incremental synchronization between mesh nodes.
    pub fn delta_since(&self, timestamp: u64) -> LWWMap {
        let mut delta = LWWMap::new();
        for (k, v) in &self.entries {
            if v.timestamp > timestamp {
                delta.entries.insert(k.clone(), v.clone());
            }
        }
        for (k, ts) in &self.tombstones {
            if *ts > timestamp {
                delta.tombstones.insert(k.clone(), *ts);            }
        }
        delta
    }
}

// =============================================================================
// GrowOnlySet
// =============================================================================

/// A set that can only grow. Elements can be added but never removed.
/// Merge is simply the union of both sets.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrowOnlySet<T> {
    pub members: HashSet<T>,
}

impl<T: Eq + std::hash::Hash + Clone + Serialize + for<'de> Deserialize<'de>> GrowOnlySet<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, item: T) {
        self.members.insert(item);
    }

    /// Merges another set by taking the union.
    /// **CRDT Properties**: Union is idempotent, commutative, and associative.
    pub fn merge(&mut self, other: &GrowOnlySet<T>) {
        self.members.extend(other.members.iter().cloned());
    }

    pub fn contains(&self, item: &T) -> bool {
        self.members.contains(item)
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.members.iter()
    }
}

// =============================================================================
// ObservedRemoveSet (2P-Set Variant)
// =============================================================================

/// A set supporting both addition and removal of elements./// Uses two underlying sets (adds and removes) with timestamps to resolve conflicts.
/// An element is considered present if its latest add timestamp is greater than its latest remove timestamp.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ObservedRemoveSet<T> {
    pub adds: HashMap<T, u64>,
    pub removes: HashMap<T, u64>,
}

impl<T: Eq + std::hash::Hash + Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>> ObservedRemoveSet<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an item at the given timestamp.
    pub fn add(&mut self, item: T, timestamp: u64) {
        self.adds
            .entry(item)
            .and_modify(|t| {
                if timestamp > *t {
                    *t = timestamp;
                }
            })
            .or_insert(timestamp);
    }

    /// Removes an item at the given timestamp.
    pub fn remove(&mut self, item: T, timestamp: u64) {
        self.removes
            .entry(item)
            .and_modify(|t| {
                if timestamp > *t {
                    *t = timestamp;
                }
            })
            .or_insert(timestamp);
    }

    /// Checks if an item is currently in the set.
    pub fn contains(&self, item: &T) -> bool {
        let add_ts = self.adds.get(item).copied().unwrap_or(0);
        let remove_ts = self.removes.get(item).copied().unwrap_or(0);
        add_ts > remove_ts
    }

    /// Merges another set by taking the maximum timestamp for each item in both add and remove maps.
    /// **CRDT Properties**: Pointwise `max` operation ensures idempotence, commutativity, and associativity.
    pub fn merge(&mut self, other: &ObservedRemoveSet<T>) {
        for (k, ts) in &other.adds {
            self.adds
                .entry(k.clone())                .and_modify(|t| {
                    if *ts > *t {
                        *t = *ts;
                    }
                })
                .or_insert(*ts);
        }
        for (k, ts) in &other.removes {
            self.removes
                .entry(k.clone())
                .and_modify(|t| {
                    if *ts > *t {
                        *t = *ts;
                    }
                })
                .or_insert(*ts);
        }
    }

    /// Returns an iterator over the current members of the set.
    pub fn members(&self) -> impl Iterator<Item = &T> {
        self.adds.iter().filter(|(k, v)| {
            self.removes.get(*k).copied().unwrap_or(0) < **v
        }).map(|(k, _)| k)
    }
}
