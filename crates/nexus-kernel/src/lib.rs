//! # Nexus Kernel
//!
//! The agent process manager and supervision core for the Nexus AI agent runtime.
//!
//! `nexus-kernel` provides the foundational primitives for treating AI agents as
//! first-class OS processes: lifecycle management, capability-based security,
//! priority scheduling, hierarchical supervision, and concurrent registry.
//!
//! ## Key Abstractions
//!
//! - [`Kernel`]: Top-level runtime API for spawning and managing agents
//! - [`AgentTask`]: Trait implemented by all agent logic
//! - [`CapabilityGuard`]: Runtime enforcement of declared permissions
//! - [`PriorityScheduler`]: Token-bucket rate limiting with priority queues
//! - [`SupervisorTree`]: Erlang/OTP-style fault tolerance with restart strategies
//! - [`AgentRegistry`]: Lock-free concurrent index for agent lookup
//!
//! ## Usage
//!
//! ```rust
//! use nexus_kernel::{Kernel, KernelConfig, AgentTask, AgentContext, SpawnOptions};
//! use nexus_proto::agent::{AgentKind, AgentCapabilities};
//!
//! async fn example() -> anyhow::Result<()> {
//!     let config = KernelConfig::default();
//!     let (kernel, mut msg_rx) = Kernel::new(config).await?;
//!
//!     let opts = SpawnOptions {
//!         name: Some("researcher".into()),
//!         priority: nexus_proto::agent::AgentPriority::Normal,
//!         capabilities: AgentCapabilities::new().with_tool("web-search"),
//!         ..Default::default()
//!     };
//!
//!     let agent_id = kernel.spawn(MyAgent, opts).await?;
//!
//!     // Process messages from agents
//!     while let Some(msg) = msg_rx.recv().await {
//!         // handle message...
//!     }
//!
//!     kernel.shutdown(None).await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Thread Safety
//!
//! All public types are `Send + Sync`. The kernel uses `DashMap` for lock-free
//! reads and `tokio::sync::RwLock` for mutable state, ensuring safe concurrent//! access from many Tokio tasks.
//!
//! ## Observability
//!
//! The kernel emits [`KernelEvent`]s via a broadcast channel for monitoring,
//! logging, and the TUI dashboard. Subscribe via [`Kernel::subscribe_events`].

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

// =============================================================================
// Module Declarations
// =============================================================================

pub mod error;
pub mod capabilities;
pub mod scheduler;
pub mod agent;
pub mod supervisor;
pub mod registry;
pub mod kernel;

// =============================================================================
// Public API Re-Exports
// =============================================================================

// Kernel — top-level runtime API
pub use kernel::{
    Kernel, KernelHandle, KernelConfig, KernelEvent, SpawnOptions,
};

// Agent — process abstraction and task trait
pub use agent::{AgentTask, AgentContext, AgentProcess};

// Supervisor — hierarchical fault tolerance
pub use supervisor::{Supervisor, SupervisorTree, RestartStrategy};

// Registry — concurrent agent index
pub use registry::{AgentRegistry, RegistryStats};

// Scheduler — priority-based resource allocation
pub use scheduler::{PriorityScheduler, SchedulerStats};

// Capabilities — security enforcement
pub use capabilities::{CapabilityGuard, CapabilitySet, Capability};
// Error — kernel-specific error types
pub use error::{KernelError, Result};

// =============================================================================
// Prelude Module
// =============================================================================

/// Convenience module for importing common kernel types.
///
/// # Example
///
/// ```rust
/// use nexus_kernel::prelude::*;
///
/// async fn handle(kernel: &Kernel) -> Result<()> {
///     // ...
/// }
/// ```
pub mod prelude {
    pub use crate::kernel::{Kernel, KernelHandle, KernelConfig, KernelEvent, SpawnOptions};
    pub use crate::agent::{AgentTask, AgentContext, AgentProcess};
    pub use crate::supervisor::{Supervisor, SupervisorTree, RestartStrategy};
    pub use crate::registry::{AgentRegistry, RegistryStats};
    pub use crate::scheduler::{PriorityScheduler, SchedulerStats};
    pub use crate::capabilities::{CapabilityGuard, CapabilitySet, Capability};
    pub use crate::error::{KernelError, Result};
}
