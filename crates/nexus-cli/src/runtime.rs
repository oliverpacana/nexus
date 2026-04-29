use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::signal;
use tokio::sync::watch;
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};

use crate::config::NexusCliConfig;

/// The unified runtime handle for all Nexus subsystems.
pub struct NexusRuntime {
    pub kernel: Arc<nexus_kernel::KernelHandle>,
    pub memory: Arc<nexus_mem::MemoryStore>,
    pub tools: Arc<nexus_tools::ToolEngine>,
    pub router: Arc<nexus_router::ModelRouter>,
    pub mesh: Option<Arc<nexus_mesh::MeshNode>>,
    pub flow: Arc<nexus_flow::WorkflowEngine>,
    pub obs: Arc<nexus_obs::ObsHandle>,
    pub config: NexusCliConfig,
    pub shutdown_tx: watch::Sender<bool>,
}

impl NexusRuntime {
    /// Returns a reference to the agent kernel.
    pub fn kernel(&self) -> &Arc<nexus_kernel::KernelHandle> {
        &self.kernel
    }

    /// Returns a reference to the four-tier memory store.
    pub fn memory(&self) -> &Arc<nexus_mem::MemoryStore> {
        &self.memory
    }

    /// Returns a reference to the tool execution engine.
    pub fn tools(&self) -> &Arc<nexus_tools::ToolEngine> {
        &self.tools
    }

    /// Returns a reference to the model routing engine.
    pub fn router(&self) -> &Arc<nexus_router::ModelRouter> {
        &self.router
    }

    /// Returns a reference to the workflow execution engine.
    pub fn flow(&self) -> &Arc<nexus_flow::WorkflowEngine> {
        &self.flow
    }
    /// Returns a reference to the observability handle.
    pub fn obs(&self) -> &Arc<nexus_obs::ObsHandle> {
        &self.obs
    }

    /// Returns the mesh node, if enabled.
    pub fn mesh(&self) -> Option<&Arc<nexus_mesh::MeshNode>> {
        self.mesh.as_ref()
    }

    /// Gracefully shuts down all runtime subsystems.
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Initiating graceful runtime shutdown...");

        // Signal all background tasks and subsystems to stop
        self.shutdown_tx.send(true)
            .context("failed to send shutdown signal")?;

        // 1. Shutdown kernel (waits for agents to finish with grace period)
        let grace_period = Duration::from_secs(self.config.runtime.graceful_shutdown_timeout_secs);
        info!("Shutting down kernel (grace period: {}s)...", grace_period.as_secs());
        self.kernel.shutdown(Some(grace_period)).await
            .context("kernel shutdown failed")?;

        // 2. Flush memory tiers to disk
        info!("Flushing memory store to disk...");
        self.memory.flush().await
            .map_err(|e| anyhow::anyhow!("memory flush failed: {}", e))?;

        // 3. Finalize cost ledger
        info!("Finalizing cost ledger...");
        self.obs.ledger().memory_ledger().prune_expired_grants();

        info!("✅ Runtime shutdown complete.");
        Ok(())
    }
}

/// Starts all Nexus subsystems in the correct dependency order.
///
/// Returns a `NexusRuntime` handle with a spawned shutdown watcher.
pub async fn start(config: NexusCliConfig) -> Result<NexusRuntime> {
    info!("⚡ Bootstrapping Nexus Runtime...");

    // 1. Initialize tracing
    info!("1/12 Initializing observability and tracing...");
    nexus_obs::tracer::init_tracing(
        &config.observability.log_level,
        &config.observability.log_format,
        config.observability.otlp_endpoint.as_deref(),    )?;

    // 2. Create PersistentCostLedger
    info!("2/12 Initializing persistent cost ledger...");
    let ledger = Arc::new(
        nexus_obs::ledger::PersistentCostLedger::new(&config.observability.cost_ledger_db_path)
            .await
            .context("failed to initialize cost ledger")?
    );

    // 3 & 4. Create NexusTracer and build ObsHandle
    info!("3/12 Creating unified observability handle...");
    let tracer = nexus_obs::tracer::get_tracer()
        .context("global tracer not initialized after init_tracing()")?;
    let obs = Arc::new(nexus_obs::ObsHandle::new(tracer, Arc::clone(&ledger)));

    // 5. Build MemoryStore
    info!("4/12 Initializing four-tier memory subsystem...");
    let memory = Arc::new(
        nexus_mem::MemoryStore::new(config.memory.clone())
            .await
            .context("failed to initialize memory store")?
    );

    // 6. Build ToolEngine
    info!("5/12 Initializing tool engine...");
    // ToolEngine requires a CapabilityGuard for security enforcement.
    // For CLI bootstrap, we start with a permissive root guard.
    let root_guard = Arc::new(nexus_kernel::capabilities::CapabilityGuard::root());
    let tools = Arc::new(
        nexus_tools::ToolEngine::new(config.tools.clone(), root_guard)
            .await
            .context("failed to initialize tool engine")?
    );

    // 7. Build ModelRouter and register providers
    info!("6/12 Initializing model router...");
    let provider_registry = Arc::new(nexus_router::providers::ProviderRegistry::new());
    let cost_ledger_mem = Arc::clone(&ledger.memory_ledger());

    if let Some(cfg) = &config.providers.openai {
        let provider = Arc::new(nexus_router::providers::openai::OpenAIProvider::new(cfg.clone()));
        provider_registry.register(provider);
        debug!("registered OpenAI provider");
    }
    if let Some(cfg) = &config.providers.anthropic {
        let provider = Arc::new(nexus_router::providers::anthropic::AnthropicProvider::new(cfg.clone()));
        provider_registry.register(provider);
        debug!("registered Anthropic provider");
    }    if let Some(cfg) = &config.providers.groq {
        let provider = Arc::new(nexus_router::providers::groq::GroqProvider::new(cfg.clone()));
        provider_registry.register(provider);
        debug!("registered Groq provider");
    }
    if let Some(cfg) = &config.providers.local {
        let provider = Arc::new(nexus_router::providers::local::LocalProvider::new(cfg.clone()));
        provider_registry.register(provider);
        debug!("registered Local provider");
    }

    let router = Arc::new(
        nexus_router::ModelRouter::new(config.router.clone(), provider_registry, cost_ledger_mem)
            .await
            .context("failed to initialize model router")?
    );

    // 8. Build Kernel
    info!("7/12 Initializing agent kernel...");
    let kernel_cfg = nexus_kernel::KernelConfig {
        max_agents: config.scheduler.max_agents,
        default_token_bucket_capacity: config.scheduler.default_token_capacity,
        default_token_refill_rate: config.scheduler.default_refill_rate,
        ..Default::default()
    };
    let kernel = nexus_kernel::Kernel::new(kernel_cfg)
        .await
        .context("failed to initialize agent kernel")?;
    // Convert to KernelHandle for unified access
    let kernel_handle = Arc::new(kernel.handle());

    // 9. Build WorkflowEngine
    info!("8/12 Initializing workflow engine...");
    let flow_cfg = nexus_flow::executor::ExecutorConfig::default();
    let flow = Arc::new(
        nexus_flow::WorkflowEngine::new(
            flow_cfg,
            Arc::clone(&kernel_handle),
            Arc::clone(&router),
            Arc::clone(&tools),
            Arc::clone(&memory),
            Arc::clone(&ledger),
        )
        .context("failed to initialize workflow engine")?
    );

    // 10. Build MeshNode (if enabled)
    info!("9/12 Initializing P2P mesh...");
    let mesh = if config.mesh.mdns_enabled || !config.mesh.listen_addr.is_empty() {
        let mut mesh_node = nexus_mesh::MeshNode::new(config.mesh.clone())            .await
            .context("failed to create mesh node")?;
        mesh_node.start().await.context("failed to start mesh node")?;
        info!("mesh node started on {}", config.mesh.listen_addr);
        Some(Arc::new(mesh_node))
    } else {
        info!("mesh disabled in configuration");
        None
    };

    // 11. Spawn background maintenance tasks
    info!("10/12 Spawning background maintenance tasks...");
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    spawn_background_tasks(
        Arc::clone(&ledger),
        Arc::clone(&memory),
        Arc::clone(&router),
        mesh.clone(),
        shutdown_rx.clone(),
    );

    // 12. Register Ctrl+C shutdown handler
    info!("11/12 Registering system signal handler...");
    {
        let runtime_shutdown = shutdown_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = signal::ctrl_c().await {
                error!("failed to listen for Ctrl+C: {}", e);
                return;
            }
            info!("🛑 Received SIGINT (Ctrl+C)");
            let _ = runtime_shutdown.send(true);
        });
    }

    info!("12/12 Runtime bootstrap complete.");
    println!("\n{}", "⚡ Nexus Runtime is running. Press Ctrl+C to shutdown.".bold().green());

    Ok(NexusRuntime {
        kernel: kernel_handle,
        memory,
        tools,
        router,
        mesh,
        flow,
        obs,
        config,
        shutdown_tx,
    })}

/// Spawns periodic maintenance tasks that run until the shutdown signal is received.
fn spawn_background_tasks(
    ledger: Arc<nexus_obs::ledger::PersistentCostLedger>,
    memory: Arc<nexus_mem::MemoryStore>,
    router: Arc<nexus_router::ModelRouter>,
    mesh: Option<Arc<nexus_mesh::MeshNode>>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    // Cost ledger flush / cleanup every 30s
    tokio::spawn({
        let ledger = Arc::clone(&ledger);
        async move {
            let mut interval = interval(Duration::from_secs(30));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Prune expired grants and compact ledger state
                        ledger.memory_ledger().prune_expired_grants();
                        debug!("cost ledger maintenance tick");
                    }
                    _ = shutdown_rx.changed() => {
                        info!("cost ledger maintenance task stopped");
                        break;
                    }
                }
            }
        }
    });

    // Memory semantic index save every 60s
    tokio::spawn({
        let memory = Arc::clone(&memory);
        async move {
            let mut interval = interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = memory.flush().await {
                            warn!("memory flush failed: {}", e);
                        }
                        debug!("memory index persisted");
                    }
                    _ = shutdown_rx.changed() => {
                        info!("memory maintenance task stopped");
                        break;
                    }
                }
            }        }
    });

    // Router health check refresh every 30s
    tokio::spawn({
        let router = Arc::clone(&router);
        async move {
            let mut interval = interval(Duration::from_secs(30));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        router.refresh_health().await;
                        debug!("model router health refreshed");
                    }
                    _ = shutdown_rx.changed() => {
                        info!("router health task stopped");
                        break;
                    }
                }
            }
        }
    });

    // Mesh blackboard eviction every 60s (only if mesh is enabled)
    if let Some(mesh_node) = mesh {
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        mesh_node.blackboard().evict_expired().await;
                        debug!("mesh blackboard expired entries evicted");
                    }
                    _ = shutdown_rx.changed() => {
                        info!("mesh maintenance task stopped");
                        break;
                    }
                }
            }
        });
    }
}

// Re-export colored for usage in runtime if needed
use colored::Colorize;
