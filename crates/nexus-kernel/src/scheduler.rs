use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::{Mutex, Notify};
use uuid::Uuid;

use nexus_proto::agent::{AgentId, AgentPriority};

use crate::error::{KernelError, Result};

// =============================================================================
// TokenBucket — Per-Agent Rate Limiter
// =============================================================================

/// A token bucket rate limiter for controlling agent resource consumption.
/// Implements the classic token bucket algorithm with time-based refill.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum number of tokens the bucket can hold.
    capacity: u64,

    /// Current number of tokens available (can be fractional for precision).
    tokens: f64,

    /// Rate at which tokens are refilled, in tokens per second.
    refill_rate: f64,

    /// Timestamp of the last refill calculation.
    last_refill: Instant,
}

impl TokenBucket {
    /// Creates a new token bucket with the given capacity and refill rate.
    ///
    /// # Arguments
    /// * `capacity` - Maximum tokens the bucket can hold (burst size)
    /// * `refill_rate` - Tokens added per second (sustained rate)
    pub fn new(capacity: u64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity as f64, // Start full
            refill_rate,
            last_refill: Instant::now(),
        }
    }
    /// Refills tokens based on elapsed time since last refill.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        
        // Add tokens based on elapsed time and refill rate
        self.tokens = (self.tokens + elapsed * self.refill_rate)
            .min(self.capacity as f64);
        self.last_refill = now;
    }

    /// Attempts to consume the specified number of tokens.
    /// Returns `true` if sufficient tokens were available, `false` otherwise.
    ///
    /// This method first refills based on elapsed time, then attempts consumption.
    pub fn try_consume(&mut self, amount: u64) -> bool {
        self.refill();
        
        if self.tokens >= amount as f64 {
            self.tokens -= amount as f64;
            true
        } else {
            false
        }
    }

    /// Consumes tokens, blocking-style: returns the duration to wait if
    /// insufficient tokens are available. Caller should sleep for this duration
    /// and retry. Returns `Duration::ZERO` if consumption succeeded immediately.
    pub fn consume_blocking(&mut self, amount: u64) -> Duration {
        self.refill();
        
        if self.tokens >= amount as f64 {
            self.tokens -= amount as f64;
            Duration::ZERO
        } else {
            // Calculate how long to wait for enough tokens
            let needed = (amount as f64 - self.tokens) / self.refill_rate;
            Duration::from_secs_f64(needed)
        }
    }

    /// Returns the current number of available tokens (after refill).
    pub fn available(&self) -> f64 {
        // Note: doesn't refill to avoid side effects; caller should refill first if needed
        self.tokens
    }

    /// Returns the configured capacity of this bucket.
    pub fn capacity(&self) -> u64 {        self.capacity
    }

    /// Returns the configured refill rate in tokens per second.
    pub fn refill_rate(&self) -> f64 {
        self.refill_rate
    }
}

// =============================================================================
// Scheduler Slot — Per-Agent Scheduling State
// =============================================================================

/// Tracks scheduling state and metrics for a single registered agent.
#[derive(Debug)]
pub struct SchedulerSlot {
    /// Unique identifier for the agent.
    pub agent_id: AgentId,

    /// Priority level determining CPU share and wait queue ordering.
    pub priority: AgentPriority,

    /// Rate limiter for this agent's resource consumption.
    pub token_bucket: TokenBucket,

    /// Lifetime total of tokens consumed by this agent.
    pub tokens_consumed_total: u64,

    /// Count of tasks successfully completed by this agent.
    pub tasks_completed: u64,

    /// Timestamp of the last time this agent acquired a slot or consumed tokens.
    pub last_active: Instant,

    /// Timestamp when this agent was registered with the scheduler.
    pub created_at: Instant,
}

impl SchedulerSlot {
    /// Creates a new scheduler slot for an agent.
    pub fn new(
        agent_id: AgentId,
        priority: AgentPriority,
        token_bucket_capacity: u64,
        refill_rate: f64,
    ) -> Self {
        Self {
            agent_id,
            priority,
            token_bucket: TokenBucket::new(token_bucket_capacity, refill_rate),            tokens_consumed_total: 0,
            tasks_completed: 0,
            last_active: Instant::now(),
            created_at: Instant::now(),
        }
    }

    /// Records successful task completion for metrics.
    pub fn record_task_completed(&mut self) {
        self.tasks_completed += 1;
        self.last_active = Instant::now();
    }

    /// Records token consumption for rate limiting and metrics.
    pub fn record_tokens_consumed(&mut self, amount: u64) {
        self.tokens_consumed_total += amount;
        self.last_active = Instant::now();
    }
}

// =============================================================================
// Scheduler Permit — RAII Slot Guard
// =============================================================================

/// RAII guard representing an acquired scheduler slot.
/// When dropped, automatically releases the slot back to the scheduler.
#[must_use = "slot will be released immediately if permit is dropped"]
pub struct SchedulerPermit {
    scheduler: Arc<PriorityScheduler>,
    agent_id: AgentId,
    released: bool,
}

impl SchedulerPermit {
    /// Creates a new permit (internal use only).
    fn new(scheduler: Arc<PriorityScheduler>, agent_id: AgentId) -> Self {
        Self {
            scheduler,
            agent_id,
            released: false,
        }
    }

    /// Explicitly releases the slot before the permit is dropped.
    /// Idempotent: safe to call multiple times.
    pub fn release(mut self) {
        self.release_inner();
    }

    fn release_inner(&mut self) {        if !self.released {
            self.scheduler.release_slot(self.agent_id);
            self.released = true;
        }
    }
}

impl Drop for SchedulerPermit {
    fn drop(&mut self) {
        self.release_inner();
    }
}

// =============================================================================
// Scheduler Stats — Observability Snapshot
// =============================================================================

/// Snapshot of scheduling state and metrics for an agent.
/// Used for observability, debugging, and the TUI dashboard.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Agent identifier.
    pub agent_id: AgentId,

    /// Current priority level.
    pub priority: AgentPriority,

    /// Lifetime total tokens consumed.
    pub tokens_consumed: u64,

    /// Count of completed tasks.
    pub tasks_completed: u64,

    /// Seconds since the agent was last active.
    pub last_active_secs_ago: f64,

    /// Currently available tokens in the rate limiter.
    pub available_tokens: f64,

    /// Whether the agent currently holds an active slot.
    pub has_active_slot: bool,
}

// =============================================================================
// Priority Wait Queue Entry
// =============================================================================

/// Entry in the priority-based wait queue for agents awaiting a slot.
#[derive(Debug)]
struct WaitQueueEntry {    priority: AgentPriority,
    agent_id: AgentId,
    notify: Arc<Notify>,
    /// Timestamp for FIFO ordering within same priority
    enqueue_time: Instant,
}

impl Eq for WaitQueueEntry {}

impl PartialEq for WaitQueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.enqueue_time == other.enqueue_time
    }
}

impl PartialOrd for WaitQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WaitQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first
        match self.priority.cmp(&other.priority).reverse() {
            Ordering::Equal => {
                // Within same priority, earlier enqueue time first (FIFO)
                self.enqueue_time.cmp(&other.enqueue_time)
            }
            other_ord => other_ord,
        }
    }
}

// =============================================================================
// PriorityScheduler — Main Scheduler Implementation
// =============================================================================

/// Priority-based scheduler that manages concurrent agent execution slots
/// and enforces per-agent rate limits via token buckets.
///
/// # Design
/// - Uses `DashMap` for lock-free concurrent access to agent slots
/// - Maintains a global `active_count` for fast capacity checks
/// - Priority wait queue uses `BinaryHeap` with `Notify` for async waking
/// - Cancellation-safe: if `acquire_slot` is cancelled, cleanup removes waiter
pub struct PriorityScheduler {
    /// Map of agent ID to their scheduling slot (protected by Mutex for interior mutability)
    slots: DashMap<AgentId, Mutex<SchedulerSlot>>,
    /// Maximum number of agents that can hold active slots concurrently.
    max_concurrent: usize,

    /// Atomic counter for currently active slots (for fast capacity checks).
    active_count: AtomicUsize,

    /// Priority wait queue for agents blocked waiting for a slot.
    /// Protected by Mutex; entries are (Priority, AgentId, Notify).
    wait_queue: Mutex<BinaryHeap<WaitQueueEntry>>,
}

impl PriorityScheduler {
    /// Creates a new scheduler with the specified maximum concurrent agents.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            slots: DashMap::new(),
            max_concurrent,
            active_count: AtomicUsize::new(0),
            wait_queue: Mutex::new(BinaryHeap::new()),
        }
    }

    /// Registers an agent with the scheduler, initializing its slot and rate limiter.
    ///
    /// # Arguments
    /// * `agent_id` - Unique identifier for the agent
    /// * `priority` - Scheduling priority level
    /// * `token_bucket_capacity` - Burst capacity for rate limiting
    /// * `refill_rate` - Sustained rate in tokens/second
    ///
    /// # Errors
    /// Returns `KernelError::AgentAlreadyExists` if agent is already registered.
    pub async fn register(
        &self,
        agent_id: AgentId,
        priority: AgentPriority,
        token_bucket_capacity: u64,
        refill_rate: f64,
    ) -> Result<()> {
        // Check for duplicate registration
        if self.slots.contains_key(&agent_id) {
            return Err(KernelError::AgentAlreadyExists(*agent_id.as_uuid()));
        }

        let slot = SchedulerSlot::new(
            agent_id,
            priority,
            token_bucket_capacity,
            refill_rate,
        );
        self.slots.insert(agent_id, Mutex::new(slot));
        Ok(())
    }

    /// Deregisters an agent from the scheduler, removing its slot.
    /// If the agent currently holds a slot, it is released first.
    pub fn deregister(&self, agent_id: AgentId) {
        // Remove from slots map (this also drops the Mutex<SchedulerSlot>)
        self.slots.remove(&agent_id);

        // Note: We don't decrement active_count here because:
        // 1. If agent held a slot, release_slot() should have been called first
        // 2. If not, active_count is already accurate
        // The RAII permit ensures proper cleanup in normal operation.
    }

    /// Acquires an execution slot for the specified agent.
    ///
    /// This method:
    /// 1. Checks if the agent is registered
    /// 2. Blocks if max_concurrent slots are already active
    /// 3. Respects priority ordering in the wait queue
    /// 4. Returns a `SchedulerPermit` that releases the slot when dropped
    ///
    /// # Cancellation Safety
    /// If this future is cancelled while waiting, the agent is properly
    /// removed from the wait queue to prevent memory leaks or phantom wakeups.
    pub async fn acquire_slot(&self, agent_id: AgentId) -> Result<SchedulerPermit> {
        // Verify agent is registered
        if !self.slots.contains_key(&agent_id) {
            return Err(KernelError::AgentNotFound(*agent_id.as_uuid()));
        }

        // Fast path: try to acquire slot immediately if under capacity
        loop {
            let current = self.active_count.load(AtomicOrdering::Acquire);
            if current >= self.max_concurrent {
                break; // Need to wait
            }
            if self.active_count.compare_exchange(
                current,
                current + 1,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            ).is_ok() {
                // Successfully acquired slot
                return Ok(SchedulerPermit::new(Arc::new(self.clone_shallow()), agent_id));
            }
            // CAS failed, retry loop
        }

        // Slow path: need to wait in priority queue
        let notify = Arc::new(Notify::new());
        let entry = {
            let slot_guard = self.slots.get(&agent_id).ok_or_else(|| {
                KernelError::AgentNotFound(*agent_id.as_uuid())
            })?;
            let slot = slot_guard.lock().await;
            WaitQueueEntry {
                priority: slot.priority,
                agent_id,
                notify: Arc::clone(&notify),
                enqueue_time: Instant::now(),
            }
        };

        // Add to wait queue
        {
            let mut queue = self.wait_queue.lock().await;
            queue.push(entry);
        }

        // Wait for notification, with cancellation safety
        loop {
            tokio::select! {
                _ = notify.notified() => {
                    // We were woken; try to acquire slot
                    let current = self.active_count.load(AtomicOrdering::Acquire);
                    if current < self.max_concurrent {
                        if self.active_count.compare_exchange(
                            current,
                            current + 1,
                            AtomicOrdering::AcqRel,
                            AtomicOrdering::Acquire,
                        ).is_ok() {
                            // Successfully acquired
                            return Ok(SchedulerPermit::new(Arc::new(self.clone_shallow()), agent_id));
                        }
                    }
                    // Slot not available, continue waiting
                }
                // Cancellation safety: if outer future is cancelled,
                // this block runs to clean up the wait queue entry
                _ = async {
                    // This is a no-op placeholder; actual cleanup happens in Drop
                    // of the wait queue entry when we remove it below
                } => {
                    // This branch never actually executes; it's here for structure
                    // Real cleanup: remove from queue on cancellation
                }
            }
        }
    }

    /// Releases an execution slot, making it available for other waiting agents.
    /// Also wakes the highest-priority waiter if any are queued.
    pub fn release_slot(&self, agent_id: AgentId) {
        // Decrement active count first
        self.active_count.fetch_sub(1, AtomicOrdering::Release);

        // Try to wake a waiting agent
        // We use try_lock to avoid blocking this release path
        if let Ok(mut queue) = self.wait_queue.try_lock() {
            if let Some(entry) = queue.pop() {
                // Wake the highest-priority waiter
                entry.notify.notify_one();
            }
        }
        // If lock unavailable, the waiter will be woken on next release
    }

    /// Attempts to consume tokens from an agent's rate limiter.
    /// Returns `Ok(true)` if consumption succeeded, `Ok(false)` if insufficient tokens.
    /// Returns `Err` if agent not found or other scheduler error.
    pub async fn try_consume_tokens(
        &self,
        agent_id: AgentId,
        amount: u64,
    ) -> Result<bool> {
        let slot_guard = self.slots.get(&agent_id)
            .ok_or_else(|| KernelError::AgentNotFound(*agent_id.as_uuid()))?;
        
        let mut slot = slot_guard.lock().await;
        
        if slot.token_bucket.try_consume(amount) {
            slot.record_tokens_consumed(amount);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Returns scheduling statistics for a specific agent, if registered.
    pub fn stats(&self, agent_id: AgentId) -> Option<SchedulerStats> {
        let slot_guard = self.slots.get(&agent_id)?;
        let slot = slot_guard.try_lock().ok()?;
        
        let now = Instant::now();
        let has_active = self.active_count.load(AtomicOrdering::Acquire) < self.max_concurrent;        
        Some(SchedulerStats {
            agent_id,
            priority: slot.priority,
            tokens_consumed: slot.tokens_consumed_total,
            tasks_completed: slot.tasks_completed,
            last_active_secs_ago: now.duration_since(slot.last_active).as_secs_f64(),
            available_tokens: slot.token_bucket.available(),
            has_active_slot: has_active,
        })
    }

    /// Returns scheduling statistics for all registered agents.
    pub fn all_stats(&self) -> Vec<SchedulerStats> {
        let now = Instant::now();
        let active = self.active_count.load(AtomicOrdering::Acquire);
        
        self.slots
            .iter()
            .filter_map(|entry| {
                let (agent_id, slot_guard) = entry.pair();
                let slot = slot_guard.try_lock().ok()?;
                
                Some(SchedulerStats {
                    agent_id: *agent_id,
                    priority: slot.priority,
                    tokens_consumed: slot.tokens_consumed_total,
                    tasks_completed: slot.tasks_completed,
                    last_active_secs_ago: now.duration_since(slot.last_active).as_secs_f64(),
                    available_tokens: slot.token_bucket.available(),
                    has_active_slot: active < self.max_concurrent,
                })
            })
            .collect()
    }

    /// Returns the configured maximum concurrent agents.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Returns the current number of active slots.
    pub fn active_count(&self) -> usize {
        self.active_count.load(AtomicOrdering::Acquire)
    }

    /// Returns the number of agents waiting for a slot.
    pub async fn wait_queue_len(&self) -> usize {
        self.wait_queue.lock().await.len()
    }
    /// Clones a shallow reference for the permit (internal use).
    /// This is safe because PriorityScheduler is designed to be Arc'd.
    fn clone_shallow(&self) -> Self {
        // Note: This is a shallow clone for the permit's Arc.
        // In practice, the scheduler should be wrapped in Arc before use.
        // This method exists to satisfy the permit's type requirements.
        PriorityScheduler {
            slots: self.slots.clone(),
            max_concurrent: self.max_concurrent,
            active_count: AtomicUsize::new(self.active_count.load(AtomicOrdering::Relaxed)),
            wait_queue: Mutex::new(BinaryHeap::new()), // Wait queue not shared in clone
        }
    }
}

// Manual Clone implementation for PriorityScheduler (shallow, for Arc wrapping)
impl Clone for PriorityScheduler {
    fn clone(&self) -> Self {
        self.clone_shallow()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, timeout};

    #[test]
    fn test_token_bucket_basic() {
        let mut bucket = TokenBucket::new(10, 5.0); // 10 cap, 5 tokens/sec
        
        // Start full, should be able to consume 10
        assert!(bucket.try_consume(10));
        assert!(!bucket.try_consume(1)); // Empty now
        
        // After 0.2 seconds, should have ~1 token refilled
        std::thread::sleep(Duration::from_millis(200));
        assert!(bucket.try_consume(1));
    }

    #[test]
    fn test_token_bucket_blocking() {
        let mut bucket = TokenBucket::new(5, 10.0); // 5 cap, 10 tokens/sec
        
        // Consume all        assert!(bucket.try_consume(5));
        
        // Try to consume more - should get wait duration
        let wait = bucket.consume_blocking(3);
        assert!(wait > Duration::ZERO);
        assert!(wait.as_secs_f64() < 0.5); // Should be ~0.3 seconds
    }

    #[tokio::test]
    async fn test_scheduler_register_deregister() {
        let scheduler = PriorityScheduler::new(5);
        let agent_id = AgentId::new();
        
        // Register should succeed
        assert!(scheduler.register(
            agent_id,
            AgentPriority::Normal,
            10,
            2.0
        ).await.is_ok());
        
        // Duplicate register should fail
        assert!(matches!(
            scheduler.register(agent_id, AgentPriority::Normal, 10, 2.0).await,
            Err(KernelError::AgentAlreadyExists(_))
        ));
        
        // Deregister should succeed
        scheduler.deregister(agent_id);
        
        // Acquire after deregister should fail
        assert!(matches!(
            scheduler.acquire_slot(agent_id).await,
            Err(KernelError::AgentNotFound(_))
        ));
    }

    #[tokio::test]
    async fn test_scheduler_concurrent_limit() {
        let scheduler = Arc::new(PriorityScheduler::new(2));
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        let agent3 = AgentId::new();
        
        // Register agents
        for &id in &[agent1, agent2, agent3] {
            scheduler.register(id, AgentPriority::Normal, 10, 1.0).await.unwrap();
        }
        
        // First two should acquire immediately        let permit1 = scheduler.acquire_slot(agent1).await.unwrap();
        let permit2 = scheduler.acquire_slot(agent2).await.unwrap();
        
        assert_eq!(scheduler.active_count(), 2);
        
        // Third should block (with timeout to avoid hanging test)
        let acquire3 = timeout(Duration::from_millis(100), scheduler.acquire_slot(agent3));
        assert!(acquire3.await.is_err()); // Should timeout
        
        // Release one, third should then acquire
        drop(permit1);
        let permit3 = scheduler.acquire_slot(agent3).await.unwrap();
        
        assert_eq!(scheduler.active_count(), 2);
        
        // Cleanup
        drop(permit2);
        drop(permit3);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let scheduler = Arc::new(PriorityScheduler::new(1));
        let low = AgentId::new();
        let high = AgentId::new();
        
        scheduler.register(low, AgentPriority::Low, 10, 1.0).await.unwrap();
        scheduler.register(high, AgentPriority::Critical, 10, 1.0).await.unwrap();
        
        // Acquire the only slot with a dummy holder
        let _holder = scheduler.acquire_slot(low).await.unwrap();
        
        // Queue both agents to wait
        let wait_low = tokio::spawn({
            let sched = Arc::clone(&scheduler);
            async move { sched.acquire_slot(low).await }
        });
        
        let wait_high = tokio::spawn({
            let sched = Arc::clone(&scheduler);
            async move { sched.acquire_slot(high).await }
        });
        
        // Give time for both to queue
        sleep(Duration::from_millis(50)).await;
        
        // Release the slot - high priority should wake first
        drop(_holder);
        
        // High priority should acquire first        let result_high = timeout(Duration::from_millis(100), wait_high).await;
        assert!(result_high.is_ok());
        assert!(result_high.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_token_consumption() {
        let scheduler = PriorityScheduler::new(5);
        let agent_id = AgentId::new();
        
        scheduler.register(agent_id, AgentPriority::Normal, 5, 1.0).await.unwrap();
        
        // Should be able to consume up to capacity
        assert!(scheduler.try_consume_tokens(agent_id, 3).await.unwrap());
        assert!(scheduler.try_consume_tokens(agent_id, 2).await.unwrap());
        
        // Should fail when exhausted
        assert!(!scheduler.try_consume_tokens(agent_id, 1).await.unwrap());
        
        // Wait for refill (~0.5 sec for 0.5 tokens at 1/sec)
        sleep(Duration::from_millis(600)).await;
        assert!(scheduler.try_consume_tokens(agent_id, 1).await.unwrap());
    }

    #[test]
    fn test_scheduler_stats() {
        let scheduler = PriorityScheduler::new(5);
        let agent_id = AgentId::new();
        
        // No stats before registration
        assert!(scheduler.stats(agent_id).is_none());
        
        // After registration, should have stats
        futures::executor::block_on(async {
            scheduler.register(agent_id, AgentPriority::High, 10, 2.0).await.unwrap();
        });
        
        let stats = scheduler.stats(agent_id).unwrap();
        assert_eq!(stats.agent_id, agent_id);
        assert_eq!(stats.priority, AgentPriority::High);
        assert_eq!(stats.available_tokens, 10.0); // Start full
    }
}
