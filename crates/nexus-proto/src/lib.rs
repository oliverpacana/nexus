//! # Nexus Protocol Definitions
//!
//! `nexus-proto` is the shared foundation crate for the Nexus AI agent runtime.
//! It contains all types, traits, and protocol definitions used for inter-subsystem
//! communication, ensuring type safety and consistency across the entire codebase.
//!
//! ## Usage
//!
//! ```rust
//! use nexus_proto::{AgentId, ModelRequest, ToolCall, MemoryEntry, Envelope};
//! ```
//!
//! For convenience, all major types are re-exported at the crate root.
//! Use the `prelude` module to import everything at once:
//!
//! ```rust
//! use nexus_proto::prelude::*;
//! ```
//!
//! ## Modules
//!
//! - [`error`]: Unified error handling with `NexusError`
//! - [`agent`]: Agent identity, lifecycle, and capability types
//! - [`model`]: LLM request/response types and provider abstractions
//! - [`tool`]: WASM tool plugin manifests and invocation types
//! - [`memory`]: Four-tier memory hierarchy types and search queries
//! - [`message`]: Inter-agent messaging protocol and envelope types
//! - [`workflow`]: DAG workflow engine types and runtime context

#![deny(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]

// =============================================================================
// Module Declarations
// =============================================================================

pub mod error;
pub mod agent;
pub mod model;
pub mod tool;
pub mod memory;
pub mod message;
pub mod workflow;

// =============================================================================
// Public Re-Exports
// =============================================================================
// Error types
pub use error::{NexusError, Result};

// Agent types
pub use agent::{
    AgentId, AgentKind, AgentPriority, AgentStatus, AgentCapabilities, AgentMeta, MemoryAccess,
};

// Model/LLM types
pub use model::{
    ModelId, ModelRequest, ModelResponse, Message, MessageRole, ContentBlock, RoutingPolicy, Token,
    ModelUsage, ToolSpec, ProviderId, FinishReason, ModelRequestBuilder,
};

// Tool types
pub use tool::{
    ToolId, ToolCall, ToolResult, ToolManifest, ResourceLimits, ToolCapabilityRequirement,
    ToolRegistryEntry,
};

// Memory types
pub use memory::{
    MemoryTier, MemoryScope, MemoryKey, MemoryEntry, EmbeddingVector, SemanticSearchQuery,
    SemanticSearchResult, EpisodicEvent, EpisodicEventType,
};

// Message types
pub use message::{MessageId, Envelope, MessageKind, ControlKind, Channel, ChannelTx, ChannelRx};

// Workflow types
pub use workflow::{
    WorkflowId, StepId, WorkflowDefinition, WorkflowRun, WorkflowContext, StepDefinition,
    StepKind, StepStatus, WorkflowRunStatus, RetryPolicy, TransformKind,
};

// =============================================================================
// Prelude Module
// =============================================================================

/// A convenience module for importing all major `nexus-proto` types at once.
///
/// # Example
///
/// ```rust
/// use nexus_proto::prelude::*;
///
/// fn handle_agent(id: AgentId, request: ModelRequest) -> Result<ToolResult> {
///     // ...
/// }/// ```
pub mod prelude {
    pub use crate::error::{NexusError, Result};
    pub use crate::agent::{
        AgentId, AgentKind, AgentPriority, AgentStatus, AgentCapabilities, AgentMeta, MemoryAccess,
    };
    pub use crate::model::{
        ModelId, ModelRequest, ModelResponse, Message, MessageRole, ContentBlock, RoutingPolicy,
        Token, ModelUsage, ToolSpec, ProviderId, FinishReason, ModelRequestBuilder,
    };
    pub use crate::tool::{
        ToolId, ToolCall, ToolResult, ToolManifest, ResourceLimits, ToolCapabilityRequirement,
        ToolRegistryEntry,
    };
    pub use crate::memory::{
        MemoryTier, MemoryScope, MemoryKey, MemoryEntry, EmbeddingVector, SemanticSearchQuery,
        SemanticSearchResult, EpisodicEvent, EpisodicEventType,
    };
    pub use crate::message::{MessageId, Envelope, MessageKind, ControlKind, Channel, ChannelTx, ChannelRx};
    pub use crate::workflow::{
        WorkflowId, StepId, WorkflowDefinition, WorkflowRun, WorkflowContext, StepDefinition,
        StepKind, StepStatus, WorkflowRunStatus, RetryPolicy, TransformKind,
    };
}
