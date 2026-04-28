// crates/nexus-tools/src/lib.rs

//! # Nexus Tools
//!
//! WASM plugin sandbox and tool execution engine for the Nexus AI agent runtime.
//!
//! `nexus-tools` provides a secure, sandboxed execution environment for AI agent tools.
//! Every tool runs as an isolated WebAssembly module with:
//! - **Capability-based security**: Host functions only expose explicitly allowed resources
//! - **Resource limits**: Per-invocation memory, fuel (CPU), and timeout enforcement
//! - **Schema validation**: JSON Schema validation for tool inputs and outputs
//! - **Hot-reloading**: Update tool implementations at runtime without agent restart
//! - **Observability**: Structured metrics, tracing, and cost accounting
//!
//! ## Architecture
//!
//! ```text
//! Agent Request → ToolEngine.call()
//!                  ├─ CapabilityGuard.check_tool()
//!                  ├─ ToolRegistry.get_latest()
//!                  ├─ WasmSandbox.invoke()
//!                  │   ├─ Validate input schema
//!                  │   ├─ Link host functions (HTTP, FS, Log, etc.)
//!                  │   ├─ Set fuel & memory limits
//!                  │   ├─ Execute WASM _nexus_run()
//!                  │   └─ Validate output schema
//!                  └─ Record metrics & return ToolResult
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use nexus_tools::{ToolEngine, ToolEngineConfig};
//! use nexus_proto::tool::ToolCall;
//! use std::sync::Arc;
//! use std::path::PathBuf;
//!
//! async fn example(kernel_guard: Arc<nexus_kernel::capabilities::CapabilityGuard>) {
//!     let config = ToolEngineConfig {
//!         registry_path: PathBuf::from("./tools/registry"),
//!         sandbox_max_memory_mb: 64,
//!         sandbox_max_fuel: 10_000_000,
//!         sandbox_timeout_ms: 5000,
//!         allow_network_by_default: false,
//!     };
//!
//!     let engine = ToolEngine::new(config, kernel_guard).await.unwrap();
//!
//!     // Install a tool
//!     engine.install_tool(PathBuf::from("./tools/web-search.wasm").as_path()).await.unwrap();//!
//!     // Execute a tool call
//!     let call = ToolCall {
//!         id: uuid::Uuid::new_v4(),
//!         tool_name: "web-search".into(),
//!         arguments: serde_json::json!({"query": "Rust WASM sandbox"}),
//!         agent_id: nexus_proto::agent::AgentId::new(),
//!         trace_id: uuid::Uuid::new_v4(),
//!     };
//!
//!     let result = engine.call(call).await.unwrap();
//!     println!("Tool output: {:?}", result.output);
//! }
//! ```
//!
//! ## Security Model
//!
//! WASM tools run in a sandbox with no default access to:
//! - Network (requires explicit host allowlist)
//! - Filesystem (requires explicit path allowlist)
//! - Environment variables (requires explicit key allowlist)
//! - Time/Random (explicitly allowed via host functions)
//!
//! All host functions perform bounds-checked linear memory access and enforce
//! resource limits via wasmtime's fuel consumption and memory reservation.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

// =============================================================================
// Module Declarations
// =============================================================================

pub mod error;
pub mod manifest;
pub mod host_functions;
pub mod sandbox;
pub mod registry;
pub mod engine;

// =============================================================================
// Public API Re-Exports
// =============================================================================

// Core engine API
pub use engine::{ToolEngine, ToolEngineConfig};
// Registry & tool management
pub use registry::{ToolRegistry, ToolRegistration, ToolStats};

// WASM sandbox execution
pub use sandbox::WasmSandbox;

// Error types
pub use error::ToolError;

// Convenience prelude
pub mod prelude {
    pub use crate::engine::{ToolEngine, ToolEngineConfig};
    pub use crate::registry::{ToolRegistry, ToolRegistration, ToolStats};
    pub use crate::sandbox::WasmSandbox;
    pub use crate::error::ToolError;
    pub use crate::manifest::{load_manifest_from_file, load_manifest_alongside_wasm};
    pub use nexus_proto::tool::{ToolCall, ToolResult, ToolManifest, ToolId, ResourceLimits};
}
