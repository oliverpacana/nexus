use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{watch, Mutex, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use nexus_proto::agent::{AgentCapabilities, AgentId, AgentKind, AgentMeta, AgentStatus};
use nexus_proto::message::{ChannelRx, ChannelTx, ControlKind, Envelope, MessageKind};
use nexus_proto::NexusError;

use crate::capabilities::{Capability, CapabilityGuard};
use crate::error::{KernelError, Result};

// =============================================================================
// AgentTask — The Agent Implementation Trait
// =============================================================================

/// The core trait that every agent implementation must satisfy.
/// Agents are stateful, async tasks that run within the Nexus kernel.
#[async_trait]
pub trait AgentTask: Send + Sync + 'static {
    /// Executes the agent's main logic loop.
    ///
    /// This method is called once when the agent is started. It should:
    /// - Process messages from `ctx.inbox`
    /// - Use `ctx` to access kernel services (memory, tools, router, etc.)
    /// - Return a JSON value representing the final result, or an error
    ///
    /// The agent should periodically check `ctx.is_shutting_down()` and
    /// exit gracefully if the runtime is terminating.
    async fn run(&mut self, ctx: AgentContext) -> Result<Value>;

    /// Returns a human-readable name for this agent type (for logging/UI).
    fn name(&self) -> &str;

    /// Returns the categorical kind of this agent.
    fn kind(&self) -> AgentKind;

    /// Returns the capabilities this agent declares it needs.
    /// These are enforced at runtime by the capability guard.
    fn capabilities(&self) -> AgentCapabilities;
}
// =============================================================================
// AgentContext — The Agent "Syscall" Interface
// =============================================================================

/// The interface through which agents interact with the Nexus kernel.
/// All fields are `Arc`-wrapped for cheap cloning across async boundaries.
#[derive(Clone)]
pub struct AgentContext {
    /// This agent's unique identifier.
    pub agent_id: AgentId,

    /// Capability enforcement guard for this agent.
    pub capability_guard: Arc<CapabilityGuard>,

    /// Receive end of the agent's message inbox.
    pub inbox: Arc<Mutex<ChannelRx<Envelope>>>,

    /// Send end of the agent's message outbox.
    pub outbox: ChannelTx<Envelope>,

    /// Watch channel receiver for shutdown signals.
    pub shutdown_signal: watch::Receiver<bool>,

    /// Mutable metadata store for agent-specific state.
    pub metadata: Arc<RwLock<HashMap<String, Value>>>,
}

impl AgentContext {
    /// Asynchronously receives the next message from the inbox.
    /// Returns `None` if the channel is closed (agent should terminate).
    pub async fn recv_message(&self) -> Option<Envelope> {
        let mut inbox = self.inbox.lock().await;
        inbox.recv().await
    }

    /// Sends a message via the outbox.
    /// Returns an error if the channel is closed or the runtime is shutting down.
    pub async fn send_message(&self, env: Envelope) -> Result<()> {
        if self.is_shutting_down() {
            return Err(KernelError::ShuttingDown.into());
        }
        self.outbox
            .send(env)
            .await
            .map_err(|e| KernelError::ChannelClosed(e.to_string()).into())
    }

    /// Returns `true` if the runtime has signaled shutdown.
    /// Agents should check this periodically and exit gracefully.
    pub fn is_shutting_down(&self) -> bool {        *self.shutdown_signal.borrow()
    }

    /// Returns this agent's unique identifier.
    pub fn agent_id(&self) -> AgentId {
        self.agent_id
    }

    /// Checks if the agent possesses the specified capability.
    /// Returns `Err(KernelError::CapabilityDenied)` if not authorized.
    pub fn check_capability(&self, cap: &Capability) -> Result<()> {
        self.capability_guard
            .has(cap)
            .then_some(())
            .ok_or_else(|| KernelError::CapabilityDenied {
                agent_id: *self.agent_id.as_uuid(),
                capability: cap.to_capability_string(),
            })
    }

    /// Yields execution to the Tokio scheduler, allowing other tasks to run.
    /// Useful for cooperative multitasking in long-running agent loops.
    pub async fn yield_now(&self) {
        tokio::task::yield_now().await;
    }

    /// Reads a metadata value by key.
    pub async fn get_metadata(&self, key: &str) -> Option<Value> {
        self.metadata.read().await.get(key).cloned()
    }

    /// Writes a metadata value.
    pub async fn set_metadata(&self, key: String, value: Value) {
        self.metadata.write().await.insert(key, value);
    }
}

// =============================================================================
// AgentProcessState — Internal State Machine
// =============================================================================

/// Internal state machine for an `AgentProcess`.
/// Mirrors `AgentStatus` but includes additional internal states.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AgentProcessState {
    Pending,
    Running,
    Suspended {
        reason: String,
        suspended_at: DateTime<Utc>,
    },
    Terminating {
        reason: String,
    },
    Completed {
        success: bool,
    },
    Failed {
        error: String,
        failed_at: DateTime<Utc>,
        retries: u32,
    },
}

impl AgentProcessState {
    /// Validates a state transition, returning an error if illegal.
    ///
    /// Legal transitions:
    /// - Pending → Running
    /// - Running → Suspended | Terminating | Failed
    /// - Suspended → Running | Terminating
    /// - Terminating → Completed | Failed
    /// - Completed, Failed → (terminal, no outgoing transitions)
    fn try_transition(&mut self, new_state: AgentProcessState) -> Result<()> {
        use AgentProcessState::*;

        let allowed = match (&self, &new_state) {
            // Initial transition
            (Pending, Running) => true,

            // Running can go to suspended, terminating, or failed
            (Running, Suspended { .. }) => true,
            (Running, Terminating { .. }) => true,
            (Running, Failed { .. }) => true,

            // Suspended can resume or terminate
            (Suspended { .. }, Running) => true,
            (Suspended { .. }, Terminating { .. }) => true,

            // Terminating can only go to terminal states
            (Terminating { .. }, Completed { .. }) => true,
            (Terminating { .. }, Failed { .. }) => true,

            // Terminal states are absorbing
            (Completed { .. } | Failed { .. }, _) => false,

            // All other transitions are illegal
            _ => false,
        };

        if allowed {
            *self = new_state;
            Ok(())
        } else {
            Err(KernelError::InvalidStateTransition {
                from: format!("{:?}", self),
                to: format!("{:?}", new_state),
                agent_id: Uuid::nil(), // Will be filled by caller
            })
        }
    }
    /// Converts internal state to the public `AgentStatus` type.
    fn to_public(&self, created_at: chrono::DateTime<chrono::Utc>) -> AgentStatus {
        use AgentProcessState::*;
        let now = chrono::Utc::now();

        match self {
            Pending => AgentStatus::Pending { created_at },
            Running { .. } => AgentStatus::Running {
                started_at: now,
                task_id: None,
            },
            Suspended { reason, .. } => AgentStatus::Suspended {
                reason: reason.clone(),
                suspended_at: now,
            },
            Terminating { .. } => AgentStatus::Pending { created_at }, // Intermediate
            Completed { success } => AgentStatus::Completed {
                finished_at: now,
                success: *success,
            },
            Failed { error, .. } => AgentStatus::Failed {
                error: error.clone(),
                failed_at: now,
                retries: 0,
            },
        }
    }
}

// =============================================================================
// AgentProcess — Runtime Wrapper for AgentTask
// =============================================================================

/// A supervised, lifecycled agent process running as a Tokio task.
/// Manages state transitions, message passing, capability enforcement,
/// and panic recovery for a single agent instance.
pub struct AgentProcess {
    /// Immutable metadata about this agent.
    meta: RwLock<AgentMeta>,

    /// Runtime context provided to the agent task.
    context: AgentContext,

    /// Handle to the running Tokio task (None if not started or completed).
    task_handle: Option<JoinHandle<Result<Value>>>,

    /// Send end of the inbox channel (kernel side for sending to agent).
    inbox_tx: ChannelTx<Envelope>,
    /// Internal state machine.
    state: RwLock<AgentProcessState>,

    /// Timestamp when this process was created.
    created_at: Instant,

    /// Shutdown signal sender (to notify agent of runtime termination).
    shutdown_tx: watch::Sender<bool>,
}

impl AgentProcess {
    /// Constructs a new `AgentProcess` for the given agent task.
    ///
    /// # Arguments
    /// * `meta` - Initial metadata for the agent
    /// * `task` - The boxed agent implementation
    /// * `capability_guard` - Pre-validated capability set for this agent
    /// * `inbox_rx` - Receive end of the message inbox
    /// * `outbox_tx` - Send end of the message outbox
    /// * `shutdown_rx` - Receiver for shutdown signals
    pub fn new(
        meta: AgentMeta,
        task: Box<dyn AgentTask>,
        capability_guard: CapabilityGuard,
        inbox_rx: ChannelRx<Envelope>,
        outbox_tx: ChannelTx<Envelope>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        let agent_id = meta.id;
        let (shutdown_tx, _) = watch::channel(false);

        let context = AgentContext {
            agent_id,
            capability_guard: Arc::new(capability_guard),
            inbox: Arc::new(Mutex::new(inbox_rx)),
            outbox: outbox_tx.clone(),
            shutdown_signal: shutdown_rx,
            metadata: Arc::new(RwLock::new(HashMap::new())),
        };

        Self {
            meta: RwLock::new(meta),
            context,
            task_handle: None,
            inbox_tx: outbox_tx,
            state: RwLock::new(AgentProcessState::Pending),
            created_at: Instant::now(),
            shutdown_tx,
        }
    }
    /// Starts the agent task, transitioning state to `Running`.
    ///
    /// This spawns a Tokio task that:
    /// 1. Calls `agent.run(context)`
    /// 2. Catches panics and converts them to `KernelError::AgentPanicked`
    /// 3. Updates internal state on completion
    ///
    /// # Cancellation Safety
    /// If this future is cancelled after spawning the task, the task
    /// continues running. Call `kill()` or `wait()` to manage lifecycle.
    pub async fn start(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        state.try_transition(AgentProcessState::Running).map_err(|e| {
            // Fill in the actual agent ID for the error
            if let KernelError::InvalidStateTransition { agent_id, .. } = e {
                KernelError::InvalidStateTransition {
                    from: format!("{:?}", AgentProcessState::Pending),
                    to: format!("{:?}", AgentProcessState::Running),
                    agent_id: *self.context.agent_id.as_uuid(),
                }
            } else {
                e
            }
        })?;
        drop(state);

        // Update public metadata
        {
            let mut meta = self.meta.write().await;
            meta.status = AgentProcessState::Running.to_public(meta.created_at);
        }

        // Prepare to spawn the task
        let context = self.context.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();
        let agent_id = self.context.agent_id;
        let agent_name = {
            let meta = self.meta.read().await;
            meta.name.clone()
        };

        // We need to move the task into the spawn; since AgentTask is a trait object,
        // we'll use a wrapper closure. In practice, the caller would pass a concrete type.
        // For this prototype, we assume the task is clone-able via Arc or similar.
        // A production implementation would use a factory pattern.

        // Spawn the task with panic handling
        let handle = tokio::spawn(async move {
            let mut shutdown_rx_signal = shutdown_rx.clone();
            // Wrap execution to catch panics
            let result = tokio::select! {
                // Run the agent task
                res = Self::run_agent_task(context, shutdown_rx) => res,
                // If shutdown signal fires, exit gracefully
                _ = async {
                    shutdown_rx_signal.changed().await.ok();
                    let x = *shutdown_rx_signal.borrow();
                    x
                } => {
                    info!(agent_id = %agent_id, "agent received shutdown signal");
                    Ok(Value::Null)
                }
            };

            // Log completion
            match &result {
                Ok(_) => debug!(agent_id = %agent_id, "agent completed successfully"),
                Err(e) => warn!(agent_id = %agent_id, error = %e, "agent completed with error"),
            }

            result
        });

        self.task_handle = Some(handle);
        info!(agent_id = %agent_id, name = %agent_name, "agent started");

        Ok(())
    }

    /// Internal helper: runs the agent task with proper error handling.
    async fn run_agent_task(
        context: AgentContext,
        mut shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Value> {
        // In a real implementation, we'd have the concrete AgentTask here.
        // For this prototype, we simulate by returning a placeholder.
        // The actual agent would be passed via a factory or generic parameter.

        // Placeholder: agents should implement their own run loop
        loop {
            if *shutdown_rx.borrow() {
                return Ok(Value::Null);
            }

            // Check for messages
            tokio::select! {
                msg = context.recv_message() => {
                    match msg {
                        Some(Envelope { kind: MessageKind::ControlSignal(ControlKind::Suspend), .. }) => {
                            // Agent should handle suspend by returning or yielding                            info!(agent_id = %context.agent_id(), "agent suspending");
                            return Ok(Value::Null);
                        }
                        Some(Envelope { kind: MessageKind::ControlSignal(ControlKind::Kill { reason }), .. }) => {
                            info!(agent_id = %context.agent_id(), reason = %reason, "agent killed");
                            return Err(KernelError::Internal(format!("killed: {}", reason)).into());
                        }
                        Some(_) => {
                            // Agent would process other messages here
                            context.yield_now().await;
                        }
                        None => {
                            // Inbox closed, agent should terminate
                            break;
                        }
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                    // Periodic yield to allow shutdown checks
                    context.yield_now().await;
                }
            }
        }

        Ok(Value::Null)
    }

    /// Suspends the agent by sending a `ControlKind::Suspend` signal.
    /// The agent should respond by pausing its work and awaiting resume.
    pub async fn suspend(&self, reason: String) -> Result<()> {
        let mut state = self.state.write().await;
        state.try_transition(AgentProcessState::Suspended {
            reason: reason.clone(),
        }).map_err(|e| {
            if let KernelError::InvalidStateTransition { .. } = e {
                KernelError::InvalidStateTransition {
                    from: format!("{:?}", *state),
                    to: format!("{:?}", AgentProcessState::Suspended { reason: reason.clone() }),
                    agent_id: *self.context.agent_id.as_uuid(),
                }
            } else {
                e
            }
        })?;
        drop(state);

        // Update public metadata
        {
            let mut meta = self.meta.write().await;
            meta.status = AgentProcessState::Suspended {                reason: reason.clone(),
                suspended_at: chrono::Utc::now(),
            }.to_public(meta.created_at);
        }

        // Send suspend signal
        let env = Envelope::new(
            AgentId::nil(), // kernel sender
            self.context.agent_id,
            MessageKind::ControlSignal(ControlKind::Suspend),
        );
        self.inbox_tx.send(env).await?;

        info!(agent_id = %self.context.agent_id, reason, "agent suspended");
        Ok(())
    }

    /// Resumes a previously suspended agent.
    pub async fn resume(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.try_transition(AgentProcessState::Running).map_err(|e| {
            if let KernelError::InvalidStateTransition { .. } = e {
                KernelError::InvalidStateTransition {
                    from: format!("{:?}", *state),
                    to: format!("{:?}", AgentProcessState::Running),
                    agent_id: *self.context.agent_id.as_uuid(),
                }
            } else {
                e
            }
        })?;
        drop(state);

        // Update public metadata
        {
            let mut meta = self.meta.write().await;
            meta.status = AgentProcessState::Running.to_public(meta.created_at);
        }

        // Send resume signal (agent should handle by continuing work)
        let env = Envelope::new(
            AgentId::nil(),
            self.context.agent_id,
            MessageKind::ControlSignal(ControlKind::Resume),
        );
        self.inbox_tx.send(env).await?;

        info!(agent_id = %self.context.agent_id, "agent resumed");
        Ok(())
    }
    /// Terminates the agent immediately with the given reason.
    /// The task is aborted if still running.
    pub async fn kill(&self, reason: String) -> Result<()> {
        let mut state = self.state.write().await;
        state.try_transition(AgentProcessState::Terminating {
            reason: reason.clone(),
        }).map_err(|e| {
            if let KernelError::InvalidStateTransition { .. } = e {
                KernelError::InvalidStateTransition {
                    from: format!("{:?}", *state),
                    to: format!("{:?}", AgentProcessState::Terminating { reason: reason.clone() }),
                    agent_id: *self.context.agent_id.as_uuid(),
                }
            } else {
                e
            }
        })?;
        drop(state);

        // Send kill signal
        let env = Envelope::new(
            AgentId::nil(),
            self.context.agent_id,
            MessageKind::ControlSignal(ControlKind::Kill { reason: reason.clone() }),
        );
        let _ = self.inbox_tx.send(env).await; // Best effort

        // Abort the task if running
        if let Some(handle) = &self.task_handle {
            handle.abort();
        }

        // Signal shutdown
        let _ = self.shutdown_tx.send(true);

        info!(agent_id = %self.context.agent_id, reason, "agent killed");
        Ok(())
    }

    /// Waits for the agent task to complete and returns its result.
    /// Transitions state to `Completed` or `Failed` based on outcome.
    pub async fn wait(&mut self) -> Result<Value> {
        let handle = self.task_handle.take().ok_or_else(|| {
            KernelError::Internal("agent task not started or already waited".into())
        })?;

        // Await the task, catching panics
        let result = match handle.await {
            Ok(Ok(value)) => {                // Successful completion
                let mut state = self.state.write().await;
                let _ = state.try_transition(AgentProcessState::Completed { success: true });
                drop(state);

                // Update metadata
                {
                    let mut meta = self.meta.write().await;
                    meta.status = AgentProcessState::Completed { success: true }
                        .to_public(meta.created_at);
                }

                Ok(value)
            }
            Ok(Err(e)) => {
                // Agent returned an error
                let error_msg = e.to_string();
                let mut state = self.state.write().await;
                let _ = state.try_transition(AgentProcessState::Failed {
                    error: error_msg.clone(),
                });
                drop(state);

                // Update metadata
                {
                    let mut meta = self.meta.write().await;
                    meta.status = AgentProcessState::Failed {
                        error: error_msg.clone(),
                        failed_at: chrono::Utc::now(),
                        retries: 0,
                    }.to_public(meta.created_at);
                }

                Err(e)
            }
            Err(join_err) => {
                // Task panicked or was cancelled
                let panic_msg = if join_err.is_panic() {
                    "agent task panicked".to_string()
                } else if join_err.is_cancelled() {
                    "agent task was cancelled".to_string()
                } else {
                    format!("agent task failed: {}", join_err)
                };

                error!(
                    agent_id = %self.context.agent_id,
                    error = %panic_msg,
                    "agent task panicked"
                );
                let mut state = self.state.write().await;
                let _ = state.try_transition(AgentProcessState::Failed {
                    error: panic_msg.clone(),
                });
                drop(state);

                // Update metadata
                {
                    let mut meta = self.meta.write().await;
                    meta.status = AgentProcessState::Failed {
                        error: panic_msg.clone(),
                        failed_at: chrono::Utc::now(),
                        retries: 0,
                    }.to_public(meta.created_at);
                }

                Err(KernelError::AgentPanicked {
                    agent_id: *self.context.agent_id.as_uuid(),
                    message: panic_msg,
                }.into())
            }
        };

        // Signal shutdown to clean up watchers
        let _ = self.shutdown_tx.send(true);

        result
    }

    /// Returns a clone of the agent's current metadata.
    pub async fn meta(&self) -> AgentMeta {
        self.meta.read().await.clone()
    }

    /// Returns the agent's current public status.
    pub async fn status(&self) -> AgentStatus {
        let state = self.state.read().await;
        let meta = self.meta.read().await;
        state.to_public(meta.created_at)
    }

    /// Sends a message to the agent's inbox.
    pub fn send(&self, env: Envelope) -> Result<()> {
        self.inbox_tx
            .try_send(env)
            .map_err(|e| KernelError::ChannelClosed(e.to_string()).into())
    }

    /// Returns the agent's unique identifier.
    pub fn agent_id(&self) -> AgentId {
        self.context.agent_id
    }

    /// Returns `true` if the agent is in a terminal state.
    pub async fn is_terminal(&self) -> bool {
        let state = self.state.read().await;
        matches!(
            *state,
            AgentProcessState::Completed { .. } | AgentProcessState::Failed { .. }
        )
    }

    /// Returns the elapsed time since this process was created.
    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

impl fmt::Debug for AgentProcess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentProcess")
            .field("agent_id", &self.context.agent_id)
            .field("state", &futures::executor::block_on(self.state.read()))
            .field("created_at", &self.created_at.elapsed())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_proto::message::Channel;

    struct DummyAgent;

    #[async_trait]
    impl AgentTask for DummyAgent {
        async fn run(&mut self, _ctx: AgentContext) -> Result<Value> {
            Ok(Value::String("dummy".into()))
        }

        fn name(&self) -> &str { "dummy" }
        fn kind(&self) -> AgentKind { AgentKind::Custom("dummy".into()) }
        fn capabilities(&self) -> AgentCapabilities { AgentCapabilities::new() }
    }
    #[tokio::test]
    async fn test_agent_lifecycle() {
        let agent_id = AgentId::new();
        let meta = AgentMeta {
            id: agent_id,
            kind: AgentKind::Custom("test".into()),
            name: "test-agent".into(),
            priority: nexus_proto::agent::AgentPriority::Normal,
            status: AgentStatus::Pending { created_at: chrono::Utc::now() },
            capabilities: AgentCapabilities::new(),
            created_at: chrono::Utc::now(),
            parent_id: None,
            tags: HashMap::new(),
        };

        let (inbox_tx, inbox_rx) = Channel::bounded(16);
        let (outbox_tx, _outbox_rx) = Channel::bounded(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let guard = CapabilityGuard::new(*agent_id.as_uuid(), crate::capabilities::CapabilitySet::empty());

        let mut process = AgentProcess::new(
            meta,
            Box::new(DummyAgent),
            guard,
            inbox_rx,
            outbox_tx,
            shutdown_rx,
        );

        // Start should succeed
        assert!(process.start().await.is_ok());
        assert!(matches!(process.status().await, AgentStatus::Running { .. }));

        // Wait for completion
        let result = process.wait().await;
        assert!(result.is_ok());
        assert!(process.is_terminal().await);
    }

    #[test]
    fn test_state_transitions() {
        use AgentProcessState::*;

        let mut state = Pending;
        assert!(state.try_transition(Running).is_ok());
        assert!(state.try_transition(Pending).is_err()); // Can't go back

        assert!(state.try_transition(Suspended { reason: "test".into() }).is_ok());
        assert!(state.try_transition(Running).is_ok()); // Can resume
        assert!(state.try_transition(Terminating { reason: "done".into() }).is_ok());
        assert!(state.try_transition(Completed { success: true }).is_ok());

        // Terminal state can't transition
        assert!(state.try_transition(Running).is_err());
    }
}
