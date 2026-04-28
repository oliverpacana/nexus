// crates/nexus-flow/src/lib.rs

//! # Nexus Flow
//!
//! DAG workflow engine for multi-step AI agent pipelines in the Nexus runtime.
//!
//! `nexus-flow` provides a robust, checkpoint-aware execution engine for orchestrating
//! agents, tools, conditionals, and parallel branches. It features:
//! - **Rust Builder DSL**: Fluent, type-safe workflow definition in code
//! - **TOML/YAML Loader**: Declarative workflow configuration for dynamic pipelines
//! - **DAG Validation**: Cycle detection, reachability checks, and terminal step verification
//! - **Checkpointing**: Crash-resilient execution with SQLite-backed state persistence
//! - **Retry & Timeout**: Configurable backoff policies and per-step deadlines
//! - **Structured Routing**: LLM-driven conditional branching with JSON schema validation
//!
//! ## Usage
//!
//! ### Programmatic (DSL)
//! ```rust
//! use nexus_flow::dsl::{StepBuilder, WorkflowBuilder};
//! use nexus_proto::agent::{AgentKind, AgentCapabilities};
//!
//! let wf = WorkflowBuilder::new("research-pipeline")
//!     .version("1.0")
//!     .description("Research and analyze")
//!     .step(
//!         StepBuilder::agent("research")
//!             .kind(AgentKind::Research)
//!             .prompt("Research: {{topic}}")
//!             .output("research_data")
//!             .then("analyze")
//!             .build()?
//!     )
//!     .step(StepBuilder::end("end", true))
//!     .build()?;
//! ```
//!
//! ### Declarative (TOML)
//! ```rust
//! use nexus_flow::loader::load_from_file;
//! let wf = load_from_file(std::path::Path::new("workflow.toml"))?;
//! ```
//!
//! ## Thread Safety
//!
//! All public types are `Send + Sync`. The executor uses `Arc`, `tokio::sync`, and
//! async primitives to ensure safe concurrent execution across agent tasks.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

// =============================================================================
// Module Declarations
// =============================================================================

pub mod error;
pub mod dag;
pub mod step;
pub mod checkpoint;
pub mod executor;
pub mod dsl;
pub mod loader;
pub mod condition;

// =============================================================================
// Public API Re-Exports
// =============================================================================

// Workflow definition & validation
pub use nexus_proto::workflow::{
    WorkflowDefinition, WorkflowRun, WorkflowRunStatus, WorkflowContext,
    StepDefinition, StepId, StepKind, StepStatus,
    RetryPolicy, TransformKind,
};

// DSL
pub use dsl::{
    WorkflowBuilder, StepBuilder,
    AgentStepBuilder, ToolStepBuilder, ConditionalStepBuilder,
    ParallelStepBuilder, TransformStepBuilder,
};

// Loader
pub use loader::{load_from_toml, load_from_yaml, load_from_file};

// Executor & DAG
pub use executor::{WorkflowExecutor, ExecutorConfig, StepExecutor};
pub use dag::WorkflowDag;

// Checkpoint
pub use checkpoint::CheckpointStore;

// Error
pub use error::FlowError;

// =============================================================================// Prelude
// =============================================================================

/// Convenience module for importing common workflow types.
///
/// # Example
///
/// ```rust
/// use nexus_flow::prelude::*;
///
/// async fn run_wf() -> Result<WorkflowRun, FlowError> {
///     // ...
/// }
/// ```
pub mod prelude {
    pub use crate::dsl::{WorkflowBuilder, StepBuilder};
    pub use crate::loader::{load_from_toml, load_from_file};
    pub use crate::executor::{WorkflowExecutor, ExecutorConfig};
    pub use crate::checkpoint::CheckpointStore;
    pub use crate::dag::WorkflowDag;
    pub use crate::error::FlowError;
    pub use nexus_proto::workflow::{
        WorkflowDefinition, WorkflowRun, StepDefinition, StepId, StepKind, RetryPolicy,
    };
}
