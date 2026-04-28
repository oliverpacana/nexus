use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use nexus_proto::agent::AgentId;

use crate::agent::{AgentRegistry, AgentTask};
use crate::capabilities::CapabilityGuard;
use crate::error::{KernelError, Result};
use crate::message::ChannelTx;

// =============================================================================
// RestartStrategy — Fault Tolerance Policies
// =============================================================================

/// Defines how a supervisor responds when a child agent fails.
/// Strategies are inspired by Erlang/OTP supervision trees.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum RestartStrategy {
    /// Restart only the failed child. If more than `max_restarts` occur within
    /// `window_secs`, the supervisor itself terminates (escalates failure up).
    OneForOne {
        max_restarts: u32,
        window_secs: u64,
    },

    /// When any child fails, restart ALL children under this supervisor.
    /// Rate-limited by `max_restarts` within `window_secs`.
    OneForAll {
        max_restarts: u32,
        window_secs: u64,
    },

    /// Restart the failed child AND all children that were started after it
    /// (in original insertion order). Rate-limited like other strategies.
    RestForOne {
        max_restarts: u32,
        window_secs: u64,
    },

    /// Always restart the child, with no rate limiting.
    /// ⚠️ Use carefully: can lead to infinite restart loops on persistent failures.
    Permanent,
    /// Restart only if the child terminated abnormally (panic, error).
    /// Normal completion (Ok return) is not restarted.
    Transient,

    /// Never restart the child. One-shot agents that run to completion.
    Temporary,
}

impl RestartStrategy {
    /// Returns the rate limit parameters for strategies that support them.
    /// Returns `None` for `Permanent`, `Transient`, `Temporary`.
    pub fn rate_limit(&self) -> Option<(u32, u64)> {
        match self {
            RestartStrategy::OneForOne { max_restarts, window_secs }
            | RestartStrategy::OneForAll { max_restarts, window_secs }
            | RestartStrategy::RestForOne { max_restarts, window_secs } => {
                Some((*max_restarts, *window_secs))
            }
            RestartStrategy::Permanent | RestartStrategy::Transient | RestartStrategy::Temporary => {
                None
            }
        }
    }

    /// Returns `true` if this strategy should restart on normal exit.
    pub fn restarts_on_normal_exit(&self) -> bool {
        matches!(self, RestartStrategy::Permanent)
    }

    /// Returns `true` if this strategy should restart on abnormal exit.
    pub fn restarts_on_abnormal_exit(&self) -> bool {
        matches!(
            self,
            RestartStrategy::Permanent | RestartStrategy::Transient | RestartStrategy::OneForOne { .. }
                | RestartStrategy::OneForAll { .. } | RestartStrategy::RestForOne { .. }
        )
    }
}

// =============================================================================
// RestartRecord — Rate Limiting for Restarts
// =============================================================================

/// Tracks restart timestamps for a child to enforce rate limiting.
/// Uses a sliding window: only restarts within `window_secs` count toward the limit.
#[derive(Debug, Clone)]
pub struct RestartRecord {
    /// Timestamps of recent restart attempts (most recent first).
    timestamps: VecDeque<Instant>,}

impl Default for RestartRecord {
    fn default() -> Self {
        Self::new()
    }
}

impl RestartRecord {
    /// Creates a new empty restart record.
    pub fn new() -> Self {
        Self {
            timestamps: VecDeque::new(),
        }
    }

    /// Records a new restart attempt at the current time.
    pub fn record(&mut self) {
        self.timestamps.push_front(Instant::now());
    }

    /// Returns the number of restarts within the specified time window.
    /// Automatically prunes timestamps outside the window.
    pub fn count_in_window(&mut self, window_secs: u64) -> u32 {
        self.prune_old(window_secs);
        self.timestamps.len() as u32
    }

    /// Removes timestamps older than the specified window.
    pub fn prune_old(&mut self, window_secs: u64) {
        let cutoff = Instant::now() - Duration::from_secs(window_secs);
        while self.timestamps.back().map_or(false, |&ts| ts < cutoff) {
            self.timestamps.pop_back();
        }
    }

    /// Returns `true` if the restart rate limit would be exceeded by another restart.
    pub fn would_exceed_limit(&mut self, max_restarts: u32, window_secs: u64) -> bool {
        self.count_in_window(window_secs) >= max_restarts
    }

    /// Clears all recorded restarts (useful for testing or manual reset).
    pub fn clear(&mut self) {
        self.timestamps.clear();
    }
}

// =============================================================================
// SupervisorChild — Agent Under Supervision
// =============================================================================
/// Represents an agent being supervised, with restart metadata and factory.
pub struct SupervisorChild {
    /// Unique identifier for the supervised agent.
    pub agent_id: AgentId,

    /// The restart strategy governing this child's fault tolerance.
    pub restart_strategy: RestartStrategy,

    /// History of restart attempts for rate limiting.
    pub restart_record: RestartRecord,

    /// Factory function to create a new instance of this agent on restart.
    /// Boxed trait object to support heterogeneous agent types.
    pub original_factory: Arc<dyn Fn() -> Box<dyn AgentTask> + Send + Sync>,

    /// Original insertion order for RestForOne strategy (lower = started earlier).
    pub start_order: u32,

    /// Capability guard to re-apply on restart.
    pub capability_guard: CapabilityGuard,
}

impl SupervisorChild {
    /// Creates a new supervised child entry.
    pub fn new(
        agent_id: AgentId,
        restart_strategy: RestartStrategy,
        factory: Arc<dyn Fn() -> Box<dyn AgentTask> + Send + Sync>,
        start_order: u32,
        capability_guard: CapabilityGuard,
    ) -> Self {
        Self {
            agent_id,
            restart_strategy,
            restart_record: RestartRecord::new(),
            original_factory: factory,
            start_order,
            capability_guard,
        }
    }

    /// Determines whether this child should be restarted given its exit condition.
    pub fn should_restart(&self, is_normal_exit: bool) -> bool {
        match &self.restart_strategy {
            RestartStrategy::Permanent => true,
            RestartStrategy::Transient => !is_normal_exit,
            RestartStrategy::Temporary => false,
            RestartStrategy::OneForOne { .. }
            | RestartStrategy::OneForAll { .. }            | RestartStrategy::RestForOne { .. } => !is_normal_exit,
        }
    }

    /// Checks if this child can be restarted without exceeding rate limits.
    /// Mutates the restart record to prune old entries.
    pub fn can_restart(&mut self) -> bool {
        if let Some((max_restarts, window_secs)) = self.restart_strategy.rate_limit() {
            !self.restart_record.would_exceed_limit(max_restarts, window_secs)
        } else {
            // Permanent/Transient/Temporary have no rate limit
            true
        }
    }

    /// Records a restart attempt and returns whether it was permitted.
    pub fn record_restart_attempt(&mut self) -> bool {
        if self.can_restart() {
            self.restart_record.record();
            true
        } else {
            false
        }
    }

    /// Creates a fresh agent instance using the original factory.
    pub fn recreate_agent(&self) -> Box<dyn AgentTask> {
        (self.original_factory)()
    }
}

// =============================================================================
// Supervisor — Supervision Tree Node
// =============================================================================

/// A supervisor node that manages a set of child agents with a restart strategy.
/// Implements hierarchical fault tolerance: failures can escalate to parent supervisors.
pub struct Supervisor {
    /// Human-readable identifier for this supervisor (for logging/debugging).
    pub id: String,

    /// The restart strategy applied to children of this supervisor.
    pub strategy: RestartStrategy,

    /// Map of child agent IDs to their supervision metadata.
    pub children: DashMap<AgentId, SupervisorChild>,

    /// Insertion order of children for RestForOne strategy.
    pub child_order: RwLock<Vec<AgentId>>,
    /// Channel to report child failures to the parent supervisor (if any).
    pub failure_tx: Option<ChannelTx<(AgentId, bool)>>,

    /// Optional parent supervisor for escalation of unhandled failures.
    pub parent: RwLock<Option<Arc<Supervisor>>>,

    /// Supervisor-level restart record for strategies that rate-limit at supervisor level.
    pub supervisor_restarts: RwLock<RestartRecord>,
}

impl Supervisor {
    /// Creates a new supervisor with the given ID and restart strategy.
    pub fn new(id: impl Into<String>, strategy: RestartStrategy) -> Self {
        Self {
            id: id.into(),
            strategy,
            children: DashMap::new(),
            child_order: RwLock::new(Vec::new()),
            failure_tx: None,
            parent: RwLock::new(None),
            supervisor_restarts: RwLock::new(RestartRecord::new()),
        }
    }

    /// Sets the failure reporting channel for escalation to parent.
    pub fn with_failure_channel(mut self, tx: ChannelTx<(AgentId, bool)>) -> Self {
        self.failure_tx = Some(tx);
        self
    }

    /// Sets the parent supervisor for this node in the tree.
    pub fn set_parent(&self, parent: Arc<Supervisor>) {
        *self.parent.blocking_write() = Some(parent);
    }

    /// Adds a child agent to this supervisor's management.
    pub fn add_child(&self, child: SupervisorChild) {
        let agent_id = child.agent_id;
        
        // Insert into children map
        self.children.insert(agent_id, child);
        
        // Track insertion order for RestForOne
        self.child_order.blocking_write().push(agent_id);
        
        debug!(
            supervisor = %self.id,
            child = %agent_id,
            "child added to supervisor"
        );    }

    /// Removes a child from supervision (e.g., when agent is explicitly terminated).
    pub fn remove_child(&self, agent_id: AgentId) {
        if self.children.remove(&agent_id).is_some() {
            // Remove from order list
            self.child_order.blocking_write().retain(|&id| id != agent_id);
            
            debug!(
                supervisor = %self.id,
                child = %agent_id,
                "child removed from supervisor"
            );
        }
    }

    /// Handles a child agent failure, applying the restart strategy.
    ///
    /// # Arguments
    /// * `agent_id` - The ID of the failed child
    /// * `is_normal` - Whether the agent exited normally (Ok) or abnormally (Err/panic)
    /// * `registry` - Reference to the agent registry for restarting agents
    ///
    /// # Returns
    /// * `Ok(Vec<AgentId>)` - List of agent IDs that should be restarted
    /// * `Err(KernelError)` - If the supervisor itself should terminate (rate limit exceeded)
    pub async fn handle_failure(
        &self,
        agent_id: AgentId,
        is_normal: bool,
        registry: &AgentRegistry,
    ) -> Result<Vec<AgentId>> {
        info!(
            supervisor = %self.id,
            child = %agent_id,
            is_normal,
            "handling child failure"
        );

        // Get mutable access to child metadata
        let mut child_opt = self.children.get_mut(&agent_id);
        let child = match child_opt.as_mut() {
            Some(c) => c,
            None => {
                warn!(
                    supervisor = %self.id,
                    child = %agent_id,
                    "failure reported for unknown child"
                );
                return Ok(Vec::new());            }
        };

        // Check if restart is warranted based on strategy and exit type
        if !child.should_restart(is_normal) {
            debug!(
                supervisor = %self.id,
                child = %agent_id,
                strategy = ?child.restart_strategy,
                "child exit does not warrant restart"
            );
            self.children.remove(&agent_id);
            return Ok(Vec::new());
        }

        // Check rate limits
        if !child.record_restart_attempt() {
            warn!(
                supervisor = %self.id,
                child = %agent_id,
                strategy = ?child.restart_strategy,
                "restart rate limit exceeded for child"
            );
            
            // Rate limit exceeded: remove child and potentially escalate
            self.children.remove(&agent_id);
            
            // For OneForOne/RestForOne, escalate to parent if configured
            if let RestartStrategy::OneForOne { .. } | RestartStrategy::RestForOne { .. } = &child.restart_strategy {
                if let Some(ref tx) = self.failure_tx {
                    let _ = tx.send((agent_id, false)).await;
                }
            }
            
            return Err(KernelError::Internal(format!(
                "restart rate limit exceeded for child {} in supervisor {}",
                agent_id, self.id
            )));
        }

        // Determine which agents to restart based on strategy
        let to_restart = match &child.restart_strategy {
            RestartStrategy::OneForOne { .. } => {
                vec![agent_id]
            }
            RestartStrategy::OneForAll { .. } => {
                // Restart all children
                self.child_order.read().await.clone()
            }
            RestartStrategy::RestForOne { .. } => {                // Restart failed child + all started after it
                let order = self.child_order.read().await;
                let mut restart_list = Vec::new();
                let mut found = false;
                
                for &id in order.iter() {
                    if found || id == agent_id {
                        found = true;
                        restart_list.push(id);
                    }
                }
                restart_list
            }
            RestartStrategy::Permanent | RestartStrategy::Transient | RestartStrategy::Temporary => {
                // These are handled by should_restart; shouldn't reach here for Temporary
                vec![agent_id]
            }
        };

        debug!(
            supervisor = %self.id,
            child = %agent_id,
            strategy = ?child.restart_strategy,
            restart_count = to_restart.len(),
            "restarting children per strategy"
        );

        Ok(to_restart)
    }

    /// Returns a vector of all child agent IDs under this supervisor.
    pub fn child_ids(&self) -> Vec<AgentId> {
        self.children.iter().map(|entry| *entry.key()).collect()
    }

    /// Returns the number of children currently supervised.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Checks if this supervisor has any children.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Gets a reference to a specific child's metadata, if present.
    pub fn get_child(&self, agent_id: AgentId) -> Option<dashmap::mapref::one::Ref<AgentId, SupervisorChild>> {
        self.children.get(&agent_id)
    }
    /// Returns the supervisor's human-readable ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the supervisor's restart strategy.
    pub fn strategy(&self) -> &RestartStrategy {
        &self.strategy
    }
}

// =============================================================================
// SupervisorTree — Root of Supervision Hierarchy
// =============================================================================

/// The root of the supervision tree, managing all supervisors and routing failures.
pub struct SupervisorTree {
    /// The root supervisor (always present).
    root: Arc<Supervisor>,

    /// Map of supervisor ID to supervisor instance for direct lookup.
    supervisors: DashMap<String, Arc<Supervisor>>,
}

impl Default for SupervisorTree {
    fn default() -> Self {
        Self::new()
    }
}

impl SupervisorTree {
    /// Creates a new supervision tree with a default root supervisor.
    pub fn new() -> Self {
        let root = Arc::new(Supervisor::new(
            "root",
            RestartStrategy::OneForOne {
                max_restarts: 10,
                window_secs: 60,
            },
        ));
        
        let mut tree = Self {
            root: Arc::clone(&root),
            supervisors: DashMap::new(),
        };
        
        // Register root in the map
        tree.supervisors.insert("root".to_string(), root);
        
        tree    }

    /// Adds a new supervisor to the tree, optionally as a child of an existing one.
    ///
    /// # Arguments
    /// * `id` - Unique human-readable identifier for the new supervisor
    /// * `strategy` - Restart strategy for this supervisor's children
    /// * `parent_id` - Optional ID of parent supervisor (None = child of root)
    ///
    /// # Returns
    /// * `Ok(Arc<Supervisor>)` - The newly created supervisor
    /// * `Err(KernelError)` - If parent not found or ID already exists
    pub fn add_supervisor(
        &self,
        id: &str,
        strategy: RestartStrategy,
        parent_id: Option<&str>,
    ) -> Result<Arc<Supervisor>> {
        // Check for duplicate ID
        if self.supervisors.contains_key(id) {
            return Err(KernelError::Internal(format!(
                "supervisor '{}' already exists",
                id
            )));
        }

        // Find parent
        let parent = match parent_id {
            Some(pid) => {
                self.supervisors.get(pid)
                    .map(|entry| Arc::clone(&entry))
                    .ok_or_else(|| KernelError::SupervisorNotFound(pid.to_string()))?
            }
            None => Arc::clone(&self.root),
        };

        // Create new supervisor
        let new_sup = Arc::new(Supervisor::new(id, strategy));
        new_sup.set_parent(Arc::clone(&parent));

        // Register in map
        self.supervisors.insert(id.to_string(), Arc::clone(&new_sup));

        info!(
            supervisor = %id,
            parent = %parent.id(),
            strategy = ?strategy,
            "supervisor added to tree"
        );
        Ok(new_sup)
    }

    /// Retrieves a supervisor by ID, if it exists in the tree.
    pub fn get_supervisor(&self, id: &str) -> Option<Arc<Supervisor>> {
        self.supervisors.get(id).map(|entry| Arc::clone(&entry))
    }

    /// Removes a supervisor from the tree.
    ///
    /// # Returns
    /// * `Ok(())` - If supervisor was removed
    /// * `Err(KernelError)` - If supervisor not found or is the root
    pub fn remove_supervisor(&self, id: &str) -> Result<()> {
        if id == "root" {
            return Err(KernelError::Internal("cannot remove root supervisor".into()));
        }

        if self.supervisors.remove(id).is_some() {
            info!(supervisor = %id, "supervisor removed from tree");
            Ok(())
        } else {
            Err(KernelError::SupervisorNotFound(id.to_string()))
        }
    }

    /// Finds which supervisor is responsible for a given agent ID.
    /// Searches all supervisors' children maps.
    pub fn which_supervisor(&self, agent_id: AgentId) -> Option<Arc<Supervisor>> {
        for entry in self.supervisors.iter() {
            if entry.value().children.contains_key(&agent_id) {
                return Some(Arc::clone(entry.value()));
            }
        }
        None
    }

    /// Handles an agent exit event, routing to the appropriate supervisor.
    ///
    /// # Arguments
    /// * `agent_id` - The ID of the exited agent
    /// * `is_normal` - Whether the exit was normal (Ok) or abnormal (Err/panic)
    /// * `registry` - Reference to the agent registry for potential restarts
    ///
    /// # Returns
    /// * `Ok(Vec<AgentId>)` - List of agents that will be restarted
    /// * `Err(KernelError)` - If handling failed or escalation occurred
    pub async fn handle_agent_exit(
        &self,
        agent_id: AgentId,        is_normal: bool,
        registry: &AgentRegistry,
    ) -> Result<Vec<AgentId>> {
        // Find the responsible supervisor
        let supervisor = match self.which_supervisor(agent_id) {
            Some(s) => s,
            None => {
                warn!(
                    agent = %agent_id,
                    "exit reported for agent with no supervisor"
                );
                return Ok(Vec::new());
            }
        };

        // Let the supervisor handle the failure
        match supervisor.handle_failure(agent_id, is_normal, registry).await {
            Ok(to_restart) => {
                debug!(
                    agent = %agent_id,
                    supervisor = %supervisor.id(),
                    restart_count = to_restart.len(),
                    "agent exit handled"
                );
                Ok(to_restart)
            }
            Err(e) => {
                // Supervisor couldn't handle it; escalate to parent if any
                if let Some(ref parent) = *supervisor.parent.read().await {
                    warn!(
                        agent = %agent_id,
                        supervisor = %supervisor.id(),
                        parent = %parent.id(),
                        error = %e,
                        "escalating failure to parent supervisor"
                    );
                    // Recursively handle at parent level
                    parent.handle_failure(agent_id, is_normal, registry).await
                } else {
                    // Reached root with unhandled failure
                    error!(
                        agent = %agent_id,
                        supervisor = %supervisor.id(),
                        error = %e,
                        "unhandled failure at root supervisor"
                    );
                    Err(e)
                }
            }
        }    }

    /// Returns the root supervisor.
    pub fn root(&self) -> &Arc<Supervisor> {
        &self.root
    }

    /// Returns the total number of supervisors in the tree (including root).
    pub fn supervisor_count(&self) -> usize {
        self.supervisors.len()
    }

    /// Returns all supervisor IDs in the tree.
    pub fn all_supervisor_ids(&self) -> Vec<String> {
        self.supervisors.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Returns a summary of the supervision tree for observability.
    pub fn summary(&self) -> SupervisorTreeSummary {
        SupervisorTreeSummary {
            supervisor_count: self.supervisors.len(),
            root_id: self.root.id().to_string(),
            root_strategy: format!("{:?}", self.root.strategy()),
            total_children: self.supervisors.iter()
                .map(|entry| entry.value().child_count())
                .sum(),
        }
    }
}

/// Summary statistics for the supervision tree.
#[derive(Debug, Clone)]
pub struct SupervisorTreeSummary {
    pub supervisor_count: usize,
    pub root_id: String,
    pub root_strategy: String,
    pub total_children: usize,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::{CapabilitySet, CapabilityGuard};

    // Dummy agent factory for testing
    fn dummy_factory() -> Box<dyn AgentTask> {        // In real code, this would return a concrete AgentTask implementation
        // For tests, we use a placeholder that would be replaced via mocking
        unimplemented!("dummy factory for tests")
    }

    #[test]
    fn test_restart_record_rate_limiting() {
        let mut record = RestartRecord::new();
        
        // Record 3 restarts
        record.record();
        record.record();
        record.record();
        
        // With window of 10 seconds and max 5, should not exceed
        assert!(!record.would_exceed_limit(5, 10));
        
        // With max 2, should exceed
        assert!(record.would_exceed_limit(2, 10));
        
        // Prune old (all are "now" so none pruned)
        record.prune_old(10);
        assert_eq!(record.count_in_window(10), 3);
    }

    #[test]
    fn test_restart_strategy_predicates() {
        let perm = RestartStrategy::Permanent;
        assert!(perm.restarts_on_normal_exit());
        assert!(perm.restarts_on_abnormal_exit());
        assert!(perm.rate_limit().is_none());

        let transient = RestartStrategy::Transient;
        assert!(!transient.restarts_on_normal_exit());
        assert!(transient.restarts_on_abnormal_exit());

        let temp = RestartStrategy::Temporary;
        assert!(!temp.restarts_on_normal_exit());
        assert!(!temp.restarts_on_abnormal_exit());

        let one_for_one = RestartStrategy::OneForOne {
            max_restarts: 3,
            window_secs: 60,
        };
        assert!(!one_for_one.restarts_on_normal_exit());
        assert!(one_for_one.restarts_on_abnormal_exit());
        assert_eq!(one_for_one.rate_limit(), Some((3, 60)));
    }

    #[test]    fn test_supervisor_child_restart_logic() {
        use std::sync::Arc;
        
        let agent_id = AgentId::new();
        let factory: Arc<dyn Fn() -> Box<dyn AgentTask> + Send + Sync> = Arc::new(|| {
            unimplemented!()
        });
        let guard = CapabilityGuard::new(*agent_id.as_uuid(), CapabilitySet::empty());
        
        // Temporary: never restart
        let mut child = SupervisorChild::new(
            agent_id,
            RestartStrategy::Temporary,
            Arc::clone(&factory),
            1,
            guard.clone(),
        );
        assert!(!child.should_restart(true));
        assert!(!child.should_restart(false));
        
        // Transient: restart only on abnormal
        let mut child = SupervisorChild::new(
            agent_id,
            RestartStrategy::Transient,
            Arc::clone(&factory),
            1,
            guard.clone(),
        );
        assert!(!child.should_restart(true));   // normal exit: no restart
        assert!(child.should_restart(false));   // abnormal: restart
        
        // Permanent: always restart
        let mut child = SupervisorChild::new(
            agent_id,
            RestartStrategy::Permanent,
            Arc::clone(&factory),
            1,
            guard,
        );
        assert!(child.should_restart(true));
        assert!(child.should_restart(false));
    }

    #[tokio::test]
    async fn test_supervisor_tree_basic() {
        let tree = SupervisorTree::new();
        
        // Root should exist
        assert!(tree.get_supervisor("root").is_some());
        assert_eq!(tree.root().id(), "root");        
        // Add child supervisor
        let child_sup = tree.add_supervisor(
            "child",
            RestartStrategy::OneForAll {
                max_restarts: 5,
                window_secs: 30,
            },
            Some("root"),
        ).unwrap();
        
        assert!(tree.get_supervisor("child").is_some());
        assert_eq!(child_sup.child_count(), 0);
        
        // Remove should work
        assert!(tree.remove_supervisor("child").is_ok());
        assert!(tree.get_supervisor("child").is_none());
        
        // Can't remove root
        assert!(tree.remove_supervisor("root").is_err());
    }

    #[test]
    fn test_restart_record_pruning() {
        use std::thread::sleep;
        
        let mut record = RestartRecord::new();
        
        // Record a restart
        record.record();
        assert_eq!(record.count_in_window(1), 1);
        
        // Wait for it to expire (use short window for test)
        sleep(Duration::from_millis(1100));
        
        // Should be pruned
        assert_eq!(record.count_in_window(1), 0);
    }
}
