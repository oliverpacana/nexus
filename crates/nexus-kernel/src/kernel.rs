use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, watch, RwLock};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Instant};
use tracing::{debug, error, info, instrument, warn, Instrument};
use uuid::Uuid;

use nexus_proto::agent::{AgentCapabilities, AgentId, AgentKind, AgentMeta, AgentPriority, AgentStatus};
use nexus_proto::message::{Channel, Envelope, MessageKind};
use nexus_proto::NexusError;

use crate::agent::{AgentProcess, AgentTask, AgentContext};
use crate::capabilities::{Capability, CapabilityGuard, CapabilitySet};
use crate::error::{KernelError, Result};
use crate::registry::AgentRegistry;
use crate::scheduler::PriorityScheduler;
use crate::supervisor::{RestartStrategy, SupervisorTree};

// =============================================================================
// Kernel Configuration
// =============================================================================

/// Configuration for the Nexus kernel runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Maximum number of concurrent agents the kernel will schedule.
    #[serde(default = "default_max_agents")]
    pub max_agents: usize,

    /// Number of Tokio worker threads for the async runtime.
    /// Set to 0 to auto-detect based on CPU cores.
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,

    /// Stack size in kilobytes allocated per agent task.
    #[serde(default = "default_stack_size")]
    pub agent_stack_size_kb: usize,

    /// Grace period to wait for agents to finish on shutdown.
    #[serde(default = "default_grace_period", with = "serde_duration")]
    pub shutdown_grace_period: Duration,
    /// Default token bucket capacity for per-agent rate limiting.
    #[serde(default = "default_token_capacity")]
    pub default_token_bucket_capacity: u64,

    /// Default token refill rate (tokens/second) for rate limiting.
    #[serde(default = "default_token_refill")]
    pub default_token_refill_rate: f64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            max_agents: default_max_agents(),
            worker_threads: default_worker_threads(),
            agent_stack_size_kb: default_stack_size(),
            shutdown_grace_period: default_grace_period(),
            default_token_bucket_capacity: default_token_capacity(),
            default_token_refill_rate: default_token_refill(),
        }
    }
}

fn default_max_agents() -> usize { 256 }
fn default_worker_threads() -> usize { 0 }
fn default_stack_size() -> usize { 2048 }
fn default_grace_period() -> Duration { Duration::from_secs(30) }
fn default_token_capacity() -> u64 { 20 }
fn default_token_refill() -> f64 { 2.0 }

// Serde helper for Duration
mod serde_duration {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where D: Deserializer<'de> {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

// =============================================================================
// Kernel Events
// =============================================================================
/// Events emitted by the kernel for observability and external monitoring.
#[derive(Debug, Clone)]
pub enum KernelEvent {
    /// An agent was successfully spawned.
    AgentSpawned(AgentMeta),

    /// An agent terminated normally.
    AgentTerminated {
        id: AgentId,
        success: bool,
    },

    /// An agent terminated with an error or panic.
    AgentFailed {
        id: AgentId,
        error: String,
    },

    /// An agent was restarted by its supervisor.
    AgentRestarted(AgentId),

    /// An agent attempted to use a capability it doesn't possess.
    CapabilityViolation {
        agent_id: AgentId,
        capability: String,
    },

    /// Kernel is beginning shutdown sequence.
    ShutdownInitiated,

    /// Kernel shutdown completed.
    ShutdownComplete,
}

// =============================================================================
// Spawn Options
// =============================================================================

/// Options for spawning a new agent.
#[derive(Debug, Clone)]
pub struct SpawnOptions {
    /// Optional human-readable name for the agent (must be unique if provided).
    pub name: Option<String>,

    /// Scheduling priority for the agent.
    pub priority: AgentPriority,

    /// Declared capabilities the agent requires.
    pub capabilities: AgentCapabilities,
    /// Optional supervisor ID to place this agent under.
    pub supervisor_id: Option<String>,

    /// Optional parent agent ID for hierarchical supervision.
    pub parent_id: Option<AgentId>,

    /// Arbitrary tags for filtering and organization.
    pub tags: HashMap<String, String>,

    /// Optional override restart strategy (defaults to supervisor's strategy).
    pub restart_strategy: Option<RestartStrategy>,
}

impl Default for SpawnOptions {
    fn default() -> Self {
        Self {
            name: None,
            priority: AgentPriority::default(),
            capabilities: AgentCapabilities::new(),
            supervisor_id: None,
            parent_id: None,
            tags: HashMap::new(),
            restart_strategy: None,
        }
    }
}

// =============================================================================
// Kernel Handle — Cheap Cloneable Reference
// =============================================================================

/// A cheap, cloneable handle to the kernel for use across async tasks.
/// Internally wraps an `Arc<KernelInner>`.
#[derive(Clone)]
pub struct KernelHandle {
    inner: Arc<KernelInner>,
}

impl KernelHandle {
    /// Returns the kernel's unique identifier.
    pub fn id(&self) -> &str {
        &self.inner.id
    }

    /// Returns a snapshot of current registry statistics.
    pub fn stats(&self) -> crate::registry::RegistryStats {
        self.inner.registry.stats()
    }

    /// Subscribes to kernel event broadcasts.    pub fn subscribe_events(&self) -> broadcast::Receiver<KernelEvent> {
        self.inner.event_bus.subscribe()
    }
}

// =============================================================================
// Kernel — Top-Level Public API
// =============================================================================

/// Internal kernel state, wrapped in Arc for KernelHandle.
struct KernelInner {
    /// Unique identifier for this kernel instance.
    id: String,

    /// Runtime configuration.
    config: KernelConfig,

    /// Central index of all registered agents.
    registry: Arc<AgentRegistry>,

    /// Supervision tree for hierarchical fault tolerance.
    supervisor_tree: Arc<SupervisorTree>,

    /// Priority-based scheduler for resource allocation.
    scheduler: Arc<PriorityScheduler>,

    /// Channel for receiving messages from agents.
    outbox: mpsc::Sender<Envelope>,

    /// Shutdown signal broadcaster.
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,

    /// Event bus for kernel lifecycle events.
    event_bus: broadcast::Sender<KernelEvent>,

    /// Counter for currently active agents (for max_agents enforcement).
    active_count: AtomicUsize,
}

/// The Nexus kernel: top-level agent runtime and process manager.
pub struct Kernel {
    inner: Arc<KernelInner>,
    /// Receive end of the outbox channel (for external message processing).
    inbox_rx: mpsc::Receiver<Envelope>,
}

impl Kernel {
    /// Creates a new kernel instance with the given configuration.
    ///    /// # Returns
    /// A tuple of `(Kernel, mpsc::Receiver<Envelope>)` where the receiver
    /// yields messages sent by agents via their outbox.
    ///
    /// # Errors
    /// Returns `KernelError` if initialization fails (e.g., Tokio runtime issues).
    #[instrument(skip(config), fields(kernel_id = %uuid::Uuid::new_v4()))]
    pub async fn new(config: KernelConfig) -> Result<(Self, mpsc::Receiver<Envelope>)> {
        let kernel_id = format!("nexus-{}", uuid::Uuid::new_v4());
        info!(kernel_id = %kernel_id, ?config, "initializing kernel");

        // Create channels
        let (outbox_tx, outbox_rx) = mpsc::channel(1024);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (event_bus, _) = broadcast::channel(256);

        // Initialize subsystems
        let registry = Arc::new(AgentRegistry::new());
        let supervisor_tree = Arc::new(SupervisorTree::new());
        let scheduler = Arc::new(PriorityScheduler::new(config.max_agents));

        let inner = Arc::new(KernelInner {
            id: kernel_id,
            config,
            registry,
            supervisor_tree,
            scheduler,
            outbox: outbox_tx,
            shutdown_tx,
            shutdown_rx,
            event_bus,
            active_count: AtomicUsize::new(0),
        });

        let kernel = Kernel {
            inner: Arc::clone(&inner),
            inbox_rx: outbox_rx,
        };

        info!(kernel_id = %inner.id, "kernel initialized");
        Ok((kernel, outbox_rx))
    }

    /// Spawns a new agent with the given task and options.
    ///
    /// # Arguments
    /// * `task` - The `AgentTask` implementation to run
    /// * `opts` - Configuration options for the agent
    ///
    /// # Returns    /// The unique `AgentId` of the spawned agent.
    ///
    /// # Errors
    /// - `KernelError::SchedulerAtCapacity` if max_agents limit reached
    /// - `KernelError::AgentAlreadyExists` if name collision
    /// - `KernelError::CapabilityDenied` if invalid capability declaration
    #[instrument(skip(self, task), fields(agent_name = ?opts.name, kind = %task.kind()))]
    pub async fn spawn<T>(&self, task: T, opts: SpawnOptions) -> Result<AgentId>
    where
        T: AgentTask + 'static,
    {
        // 1. Check max_agents limit
        let current = self.inner.active_count.load(Ordering::Acquire);
        if current >= self.inner.config.max_agents {
            return Err(KernelError::SchedulerAtCapacity {
                max_agents: self.inner.config.max_agents,
            });
        }

        // 2. Generate agent ID and build metadata
        let agent_id = AgentId::new();
        let name = opts.name.unwrap_or_else(|| format!("agent-{}", agent_id));
        let kind = task.kind();
        let capabilities = task.capabilities();

        let meta = AgentMeta {
            id: agent_id,
            kind: kind.clone(),
            name: name.clone(),
            priority: opts.priority,
            status: AgentStatus::Pending {
                created_at: chrono::Utc::now(),
            },
            capabilities: capabilities.clone(),
            created_at: chrono::Utc::now(),
            parent_id: opts.parent_id,
            tags: opts.tags.clone(),
        };

        // 3. Build capability guard from declared capabilities
        let cap_set = CapabilitySet::from(capabilities);
        let cap_guard = CapabilityGuard::new(*agent_id.as_uuid(), cap_set);

        // 4. Create message channels for this agent
        let (inbox_tx, inbox_rx) = Channel::bounded(64);
        let (outbox_tx, outbox_rx) = Channel::bounded(64);

        // 5. Build agent context
        let ctx = AgentContext {
            agent_id,            capability_guard: Arc::new(cap_guard.clone()),
            inbox: Arc::new(tokio::sync::Mutex::new(inbox_rx)),
            outbox: outbox_tx.clone(),
            shutdown_signal: self.inner.shutdown_tx.subscribe(),
            metadata: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        };

        // 6. Create the AgentProcess wrapper
        let process = AgentProcess::new(
            meta.clone(),
            Box::new(task),
            cap_guard,
            inbox_rx,
            outbox_tx,
            ctx.shutdown_signal.clone(),
        );

        let process_arc = Arc::new(tokio::sync::RwLock::new(process));

        // 7. Register with scheduler
        self.inner
            .scheduler
            .register(
                agent_id,
                opts.priority,
                self.inner.config.default_token_bucket_capacity,
                self.inner.config.default_token_refill_rate,
            )
            .await?;

        // 8. Register with supervisor tree if specified
        if let Some(ref sup_id) = opts.supervisor_id {
            if let Some(supervisor) = self.inner.supervisor_tree.get_supervisor(sup_id) {
                use crate::supervisor::SupervisorChild;
                use std::sync::Arc as StdArc;

                // Create a factory closure that can recreate this agent type
                // Note: This requires the task to be clone-able or use a factory pattern.
                // For this prototype, we use a placeholder that would be replaced in production.
                let factory: StdArc<dyn Fn() -> Box<dyn AgentTask> + Send + Sync> = 
                    StdArc::new(|| {
                        // In production, this would use a registered factory or clone the task
                        // For now, we panic to indicate this needs proper implementation
                        unimplemented!("agent factory for supervisor restarts requires concrete task cloning")
                    });

                let child = SupervisorChild::new(
                    agent_id,
                    opts.restart_strategy.unwrap_or_else(|| 
                        RestartStrategy::OneForOne {                            max_restarts: 3,
                            window_secs: 60,
                        }
                    ),
                    factory,
                    0, // start_order: would be tracked by supervisor
                    cap_guard.clone(),
                );
                supervisor.add_child(child);
            } else {
                return Err(KernelError::SupervisorNotFound(sup_id.clone()));
            }
        }

        // 9. Register in the central registry
        self.inner.registry.register(Arc::clone(&process_arc))?;

        // 10. Increment active count and start the process
        self.inner.active_count.fetch_add(1, Ordering::Release);
        {
            let mut proc = process_arc.write().await;
            proc.start().await?;
        }

        // 11. Emit spawn event
        let _ = self.inner.event_bus.send(KernelEvent::AgentSpawned(meta.clone()));
        debug!(agent_id = %agent_id, name = %name, "agent spawned");

        // 12. Spawn the monitor task for lifecycle management
        let monitor_inner = Arc::clone(&self.inner);
        let monitor_process = Arc::clone(&process_arc);
        let monitor_meta = meta.clone();

        tokio::spawn(
            async move {
                Self::monitor_agent(
                    monitor_inner,
                    monitor_process,
                    monitor_meta,
                    agent_id,
                    opts,
                )
                .await
            }
            .instrument(tracing::info_span!("agent_monitor", agent_id = %agent_id, name = %name)),
        );

        // 13. Return the agent ID to the caller
        Ok(agent_id)
    }
    /// Monitor task: watches an agent's lifecycle and handles restart logic.
    #[instrument(skip(kernel, process, meta, opts))]
    async fn monitor_agent(
        kernel: Arc<KernelInner>,
        process: Arc<RwLock<AgentProcess>>,
        meta: AgentMeta,
        agent_id: AgentId,
        opts: SpawnOptions,
    ) {
        let agent_name = meta.name.clone();
        let agent_kind = meta.kind.clone();

        // Await the agent's completion
        let result = {
            let mut proc = process.write().await;
            proc.wait().await
        };

        // Determine exit type
        let (is_normal, error_msg) = match &result {
            Ok(_) => (true, None),
            Err(e) => (false, Some(e.to_string())),
        };

        // 1. Deregister from registry and scheduler
        kernel.registry.deregister(agent_id).ok();
        kernel.scheduler.deregister(agent_id);
        kernel.active_count.fetch_sub(1, Ordering::Release);

        // 2. Emit termination event
        let event = match &result {
            Ok(_) => KernelEvent::AgentTerminated {
                id: agent_id,
                success: true,
            },
            Err(_) => KernelEvent::AgentFailed {
                id: agent_id,
                error: error_msg.clone().unwrap_or_else(|| "unknown error".into()),
            },
        };
        let _ = kernel.event_bus.send(event);

        // 3. Handle restart logic via supervisor tree
        if !is_normal {
            match kernel
                .supervisor_tree
                .handle_agent_exit(agent_id, is_normal, &kernel.registry)
                .await
            {                Ok(to_restart) => {
                    if to_restart.contains(&agent_id) {
                        // Attempt to restart this agent
                        info!(
                            agent_id = %agent_id,
                            name = %agent_name,
                            kind = %agent_kind,
                            "restarting agent per supervisor strategy"
                        );

                        // Emit restart event
                        let _ = kernel.event_bus.send(KernelEvent::AgentRestarted(agent_id));

                        // Re-spawn with same factory
                        // Note: This requires the original task factory, which we don't have here.
                        // In production, the SupervisorChild would hold the factory.
                        // For this prototype, we log that restart would occur.
                        warn!(
                            agent_id = %agent_id,
                            "agent restart requested but factory not available in prototype"
                        );
                    }
                }
                Err(e) => {
                    error!(
                        agent_id = %agent_id,
                        error = %e,
                        "supervisor failed to handle agent exit"
                    );
                }
            }
        }

        debug!(
            agent_id = %agent_id,
            name = %agent_name,
            success = is_normal,
            "agent monitor completed"
        );
    }

    /// Terminates an agent immediately with the given reason.
    #[instrument(skip(self, reason), fields(agent_id = %id))]
    pub async fn kill(&self, id: AgentId, reason: &str) -> Result<()> {
        let proc = self
            .inner
            .registry
            .get(id)
            .ok_or_else(|| KernelError::AgentNotFound(*id.as_uuid()))?;
        {
            let mut p = proc.write().await;
            p.kill(reason.to_string()).await?;
        }

        info!(agent_id = %id, reason, "agent killed");
        Ok(())
    }

    /// Suspends an agent's execution.
    #[instrument(skip(self), fields(agent_id = %id))]
    pub async fn suspend(&self, id: AgentId) -> Result<()> {
        let proc = self
            .inner
            .registry
            .get(id)
            .ok_or_else(|| KernelError::AgentNotFound(*id.as_uuid()))?;

        {
            let p = proc.read().await;
            p.suspend("user requested".into()).await?;
        }

        info!(agent_id = %id, "agent suspended");
        Ok(())
    }

    /// Resumes a previously suspended agent.
    #[instrument(skip(self), fields(agent_id = %id))]
    pub async fn resume(&self, id: AgentId) -> Result<()> {
        let proc = self
            .inner
            .registry
            .get(id)
            .ok_or_else(|| KernelError::AgentNotFound(*id.as_uuid()))?;

        {
            let p = proc.read().await;
            p.resume().await?;
        }

        info!(agent_id = %id, "agent resumed");
        Ok(())
    }

    /// Sends a message envelope to an agent's inbox.
    #[instrument(skip(self), fields(to = %envelope.to))]
    pub async fn send(&self, envelope: Envelope) -> Result<()> {
        if envelope.is_expired() {
            return Err(KernelError::Internal("message expired".into()));        }

        self.inner
            .registry
            .send_to(envelope.to, envelope)
            .map_err(|e| match e {
                NexusError::AgentNotFound(id) => {
                    KernelError::AgentNotFound(Uuid::parse_str(&id).unwrap_or(Uuid::nil()))
                }
                _ => KernelError::Internal(format!("send failed: {}", e)),
            })
    }

    /// Retrieves an agent's current metadata snapshot.
    pub fn get_meta(&self, id: AgentId) -> Option<AgentMeta> {
        self.inner
            .registry
            .get(id)
            .and_then(|proc| {
                // Use blocking_read since this is a sync method
                let p = proc.blocking_read();
                futures::executor::block_on(p.meta())
            })
    }

    /// Lists metadata for all registered agents.
    pub fn list_agents(&self) -> Vec<AgentMeta> {
        let ids = self.inner.registry.list_all();
        ids.into_iter()
            .filter_map(|id| self.get_meta(id))
            .collect()
    }

    /// Returns a snapshot of registry statistics.
    pub fn stats(&self) -> crate::registry::RegistryStats {
        self.inner.registry.stats()
    }

    /// Subscribes to kernel event broadcasts.
    pub fn subscribe_events(&self) -> broadcast::Receiver<KernelEvent> {
        self.inner.event_bus.subscribe()
    }

    /// Initiates graceful shutdown of the kernel.
    ///
    /// # Arguments
    /// * `grace_period` - Optional override for config.shutdown_grace_period
    ///
    /// # Behavior
    /// 1. Signals all agents via shutdown watch channel    /// 2. Waits up to grace_period for agents to terminate naturally
    /// 3. Force-kills any remaining agents
    /// 4. Emits ShutdownComplete event
    #[instrument(skip(self))]
    pub async fn shutdown(&self, grace_period: Option<Duration>) -> Result<()> {
        let period = grace_period.unwrap_or(self.inner.config.shutdown_grace_period);
        info!(grace_period_secs = %period.as_secs(), "initiating kernel shutdown");

        // Emit shutdown initiated event
        let _ = self.inner.event_bus.send(KernelEvent::ShutdownInitiated);

        // Signal shutdown to all agents
        let _ = self.inner.shutdown_tx.send(true);

        // Collect all agent IDs before they're deregistered
        let agent_ids: Vec<AgentId> = self.inner.registry.list_all();

        // Wait for agents to finish, with timeout
        let shutdown_start = Instant::now();
        let mut remaining = agent_ids.clone();

        while !remaining.is_empty() && shutdown_start.elapsed() < period {
            tokio::time::sleep(Duration::from_millis(50)).await;
            remaining.retain(|&id| {
                self.inner
                    .registry
                    .get(id)
                    .and_then(|p| {
                        let proc = p.try_read().ok()?;
                        Some(!futures::executor::block_on(proc.is_terminal()))
                    })
                    .unwrap_or(false)
            });
        }

        // Force-kill any remaining agents
        for id in remaining {
            warn!(agent_id = %id, "force-killing agent after grace period");
            let _ = self.kill(id, "shutdown timeout").await;
        }

        // Clear the registry
        self.inner.registry.clear();

        // Emit completion event
        let _ = self.inner.event_bus.send(KernelEvent::ShutdownComplete);
        info!("kernel shutdown complete");

        Ok(())
    }
    /// Returns a cheap, cloneable handle to this kernel.
    pub fn handle(&self) -> KernelHandle {
        KernelHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Returns the receive end of the outbox channel for external message processing.
    ///
    /// This method consumes `self`, transferring ownership of the receiver.
    /// Typically called once after kernel construction to start the message loop.
    pub fn into_message_receiver(self) -> mpsc::Receiver<Envelope> {
        self.inbox_rx
    }

    /// Returns a reference to the kernel's unique identifier.
    pub fn id(&self) -> &str {
        &self.inner.id
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_proto::agent::AgentCapabilities;

    struct TestAgent;

    #[async_trait]
    impl AgentTask for TestAgent {
        async fn run(&mut self, _ctx: AgentContext) -> Result<serde_json::Value> {
            Ok(serde_json::Value::String("done".into()))
        }
        fn name(&self) -> &str { "test" }
        fn kind(&self) -> AgentKind { AgentKind::Custom("test".into()) }
        fn capabilities(&self) -> AgentCapabilities { AgentCapabilities::new() }
    }

    #[tokio::test]
    async fn test_kernel_spawn_basic() {
        let config = KernelConfig {
            max_agents: 10,
            ..Default::default()
        };
        let (kernel, _rx) = Kernel::new(config).await.unwrap();

        let opts = SpawnOptions {
            name: Some("test-agent".into()),
            ..Default::default()
        };

        let id = kernel.spawn(TestAgent, opts).await.unwrap();
        assert!(!id.as_uuid().is_nil());

        // Verify registration
        let meta = kernel.get_meta(id).unwrap();
        assert_eq!(meta.name, "test-agent");
        assert!(matches!(meta.status, AgentStatus::Running { .. }));

        // Cleanup
        let _ = kernel.shutdown(Some(Duration::from_millis(100))).await;
    }

    #[tokio::test]
    async fn test_kernel_max_agents() {
        let config = KernelConfig {
            max_agents: 2,
            ..Default::default()
        };

        let (kernel, _rx) = Kernel::new(config).await.unwrap();

        // Spawn up to limit
        for i in 0..2 {
            let opts = SpawnOptions {
                name: Some(format!("agent-{}", i)),
                ..Default::default()
            };
            assert!(kernel.spawn(TestAgent, opts).await.is_ok());
        }

        // Next should fail
        let opts = SpawnOptions {
            name: Some("agent-overflow".into()),
            ..Default::default()
        };
        let result = kernel.spawn(TestAgent, opts).await;
        assert!(matches!(result, Err(KernelError::SchedulerAtCapacity { .. })));

        let _ = kernel.shutdown(Some(Duration::from_millis(100))).await;
    }

    #[test]
    fn test_kernel_config_defaults() {        let config = KernelConfig::default();
        assert_eq!(config.max_agents, 256);
        assert_eq!(config.worker_threads, 0);
        assert_eq!(config.agent_stack_size_kb, 2048);
        assert_eq!(config.shutdown_grace_period, Duration::from_secs(30));
        assert_eq!(config.default_token_bucket_capacity, 20);
        assert_eq!(config.default_token_refill_rate, 2.0);
    }

    #[test]
    fn test_kernel_event_clone() {
        let id = AgentId::new();
        let event = KernelEvent::AgentSpawned(AgentMeta {
            id,
            kind: AgentKind::Research,
            name: "test".into(),
            priority: AgentPriority::Normal,
            status: AgentStatus::Pending { created_at: chrono::Utc::now() },
            capabilities: AgentCapabilities::new(),
            created_at: chrono::Utc::now(),
            parent_id: None,
            tags: HashMap::new(),
        });

        let _clone = event.clone(); // Should compile: KernelEvent is Clone
    }

    #[tokio::test]
    async fn test_kernel_handle_clone() {
        let (kernel, _rx) = Kernel::new(KernelConfig::default()).await.unwrap();
        let handle = kernel.handle();

        // Handle should be cheaply cloneable
        let handle2 = handle.clone();
        assert_eq!(handle.id(), handle2.id());

        let _ = kernel.shutdown(Some(Duration::from_millis(100))).await;
    }
}
