// crates/nexus-tools/src/engine.rs

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nexus_proto::tool::{ToolCall, ToolId, ToolManifest as ProtoManifest, ToolResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, instrument, warn};

use crate::error::{ToolError, Result};
use crate::registry::{ToolRegistry, ToolRegistration, ToolStats};
use crate::sandbox::WasmSandbox;

// CapabilityGuard is defined in nexus-kernel. In a full workspace, 
// nexus-tools would either depend on nexus-kernel or use a trait abstraction.
// For this implementation, we assume the type is available via workspace linkage.
#[allow(dead_code)]
use nexus_kernel::capabilities::CapabilityGuard;

/// Configuration for the ToolEngine runtime.
#[derive(Debug, Clone)]
pub struct ToolEngineConfig {
    /// Root directory where installed tools are stored and versioned.
    pub registry_path: std::path::PathBuf,

    /// Maximum WASM memory per tool invocation (megabytes).
    pub sandbox_max_memory_mb: u32,

    /// Maximum WASM fuel units per tool invocation.
    pub sandbox_max_fuel: u64,

    /// Hard timeout for any tool execution (milliseconds).
    pub sandbox_timeout_ms: u64,

    /// Whether tools are granted network access by default unless explicitly restricted.
    pub allow_network_by_default: bool,
}

/// The top-level public API for tool execution and management.
///
/// `ToolEngine` orchestrates capability checking, tool resolution, WASM sandboxing,
/// metrics collection, and hot-reloading. All agent tool invocations flow through
/// this single entry point.
pub struct ToolEngine {
    /// Registry managing installed tool versions and WASM compilation.
    pub registry: Arc<ToolRegistry>,

    /// Reference to the calling agent's capability guard for security enforcement.
    pub kernel_guard: Arc<CapabilityGuard>,
    /// Runtime configuration applied to all invocations.
    pub config: ToolEngineConfig,
}

impl ToolEngine {
    /// Creates a new `ToolEngine` instance.
    ///
    /// # Arguments
    /// * `config` - Engine configuration including registry path and sandbox limits
    /// * `kernel_guard` - Agent capability guard for enforcing tool permissions
    ///
    /// # Returns
    /// * `Ok(ToolEngine)` - If registry initialized and existing tools loaded
    /// * `Err(ToolError)` - If registry creation or directory scan fails
    #[instrument(skip(config, kernel_guard), fields(registry_path = ?config.registry_path))]
    pub async fn new(
        config: ToolEngineConfig,
        kernel_guard: Arc<CapabilityGuard>,
    ) -> Result<Self> {
        debug!("initializing tool engine");

        // Initialize registry with shared engine
        let registry = ToolRegistry::new(config.registry_path.clone())?;

        // Load existing tools from disk
        let count = registry.load_from_directory().await?;
        if count > 0 {
            info!(count, "loaded existing tools from registry directory");
        }

        Ok(Self {
            registry: Arc::new(registry),
            kernel_guard,
            config,
        })
    }

    /// Executes a tool call in an isolated WASM sandbox with capability enforcement.
    ///
    /// # Flow
    /// 1. Verifies calling agent has permission to invoke the requested tool
    /// 2. Resolves tool version (defaults to latest)
    /// 3. Executes tool via pre-compiled `WasmSandbox`
    /// 4. Records execution metrics (duration, success/failure)
    /// 5. Emits structured tracing events
    ///
    /// # Arguments
    /// * `call` - The `ToolCall` containing tool name, arguments, and tracing metadata
    ///    /// # Returns
    /// * `Ok(ToolResult)` - If tool executed successfully or failed gracefully
    /// * `Err(ToolError)` - If capability denied, tool not found, or sandbox execution failed
    #[instrument(skip(self, call), fields(tool = %call.tool_name, agent = %call.agent_id))]
    pub async fn call(&self, call: ToolCall) -> Result<ToolResult> {
        let start = Instant::now();

        // 1. Capability enforcement
        if let Err(e) = self.kernel_guard.check_tool(&call.tool_name) {
            warn!(
                tool = %call.tool_name,
                agent = %call.agent_id,
                error = %e,
                "capability denied for tool call"
            );
            return Err(ToolError::CapabilityDenied(e.to_string()));
        }

        // 2. Tool resolution
        let registration = self.registry.get(&call.tool_name, None)
            .ok_or_else(|| ToolError::NotFound(
                self.config.registry_path.join(&call.tool_name)
            ))?;

        // 3. Sandbox execution
        debug!(
            version = %registration.manifest.version,
            "dispatching to WASM sandbox"
        );

        let exec_result = registration.sandbox.invoke(&call).await;
        let duration_ms = start.elapsed().as_millis() as u64;

        // 4. Metrics recording & 5. Logging
        match exec_result {
            Ok(tool_result) => {
                let success = !tool_result.is_error;
                registration.record_call(duration_ms, success);

                if success {
                    info!(
                        tool = %call.tool_name,
                        agent = %call.agent_id,
                        duration_ms,
                        fuel_consumed = tool_result.fuel_consumed,
                        memory_mb = tool_result.memory_peak_mb,
                        "tool execution succeeded"
                    );
                } else {
                    warn!(                        tool = %call.tool_name,
                        agent = %call.agent_id,
                        duration_ms,
                        error = %tool_result.error_message.as_deref().unwrap_or("unknown"),
                        "tool execution completed with error"
                    );
                }

                Ok(tool_result)
            }
            Err(e) => {
                registration.record_call(duration_ms, false);
                error!(
                    tool = %call.tool_name,
                    agent = %call.agent_id,
                    duration_ms,
                    error = %e,
                    "tool execution failed"
                );
                Err(e)
            }
        }
    }

    /// Installs a new tool from a WASM binary into the registry.
    ///
    /// # Arguments
    /// * `wasm_path` - Filesystem path to the compiled `.wasm` file
    ///
    /// # Returns
    /// * `Ok(ToolId)` - Identifier for the newly installed tool
    /// * `Err(ToolError)` - If manifest missing, checksum mismatch, or compilation fails
    pub async fn install_tool(&self, wasm_path: &Path) -> Result<ToolId> {
        self.registry.install(wasm_path, None).await
    }

    /// Returns a list of usage statistics for all installed tools.
    ///
    /// Includes one entry per installed version. Sorted by tool name.
    pub fn list_tools(&self) -> Vec<ToolStats> {
        self.registry.list()
    }

    /// Retrieves the manifest metadata for a tool by name.
    ///
    /// Resolves to the latest version if no specific version is requested.
    pub fn get_tool_manifest(&self, name: &str) -> Option<ProtoManifest> {
        self.registry.get(name, None).map(|reg| reg.manifest.clone())
    }
    /// Hot-reloads a tool by re-reading from disk and recompiling the WASM module.
    ///
    /// Existing in-flight calls continue with the old sandbox; new calls use the
    /// recompiled version. Useful for updating tool logic without restarting agents.
    pub async fn reload_tool(&self, name: &str) -> Result<()> {
        self.registry.reload(name).await
    }

    /// Returns the underlying registry for advanced operations.
    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }

    /// Returns the engine configuration.
    pub fn config(&self) -> &ToolEngineConfig {
        &self.config
    }
}
