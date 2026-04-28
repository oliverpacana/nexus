// crates/nexus-router/src/lib.rs

//! # Nexus Router
//!
//! Universal model gateway and intelligent routing engine for the Nexus AI agent runtime.
//!
//! `nexus-router` provides a unified interface to multiple LLM providers with:
//! - **Intelligent routing**: Cost, latency, and capability-based provider selection
//! - **Budget enforcement**: Per-agent spending limits with real-time tracking
//! - **Retry logic**: Exponential backoff with jitter for transient failures
//! - **Health monitoring**: Automatic provider health checks and failover
//! - **Streaming support**: Server-Sent Events (SSE) handling for token streaming
//! - **Cost tracking**: Token counting and USD cost estimation for all requests
//!
//! ## Architecture
//!
//! ```text
//! Agent Request → ModelRouter.complete()/stream()
//!                  ├─ select_provider() by RoutingPolicy
//!                  │  ├─ CostOptimized: cheapest within latency budget
//!                  │  ├─ LatencyOptimized: fastest within cost budget
//!                  │  ├─ CapabilityFirst: context/vision requirements
//!                  │  ├─ LocalFirst: prefer local, fallback to cloud
//!                  │  └─ Pinned: specific provider/model
//!                  ├─ check_budget() if enforcement enabled
//!                  ├─ execute with timeout + retry
//!                  ├─ record_cost() to ledger
//!                  └─ return ModelResponse or Token stream
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use nexus_router::{ModelRouter, RouterConfig};
//! use nexus_router::providers::{ProviderRegistry, openai::OpenAIProvider, OpenAIConfig};
//! use nexus_router::cost::CostLedger;
//! use nexus_proto::model::{ModelRequest, RoutingPolicy, ProviderId};
//! use std::sync::Arc;
//!
//! async fn example() {
//!     // Setup providers
//!     let registry = Arc::new(ProviderRegistry::new());
//!     registry.register(Arc::new(OpenAIProvider::new(OpenAIConfig::default())));
//!
//!     // Setup cost ledger
//!     let ledger = Arc::new(CostLedger::new());
//!
//!     // Configure router
//!     let config = RouterConfig {
//!         default_policy: RoutingPolicy::CostOptimized { max_latency_ms: 3000 },//!         request_timeout_secs: 60,
//!         max_retries: 2,
//!         budget_enforcement: true,
//!         ..Default::default()
//!     };
//!
//!     // Create router
//!     let router = ModelRouter::new(config, registry, ledger).await;
//!
//!     // Execute a request
//!     let request = ModelRequest::builder()
//!         .messages(vec![nexus_proto::model::Message::user("Hello!")])
//!         .routing_policy(RoutingPolicy::LatencyOptimized { max_cost_per_1k_tokens: 0.01 })
//!         .build()
//!         .unwrap();
//!
//!     let response = router.complete(request, nexus_proto::agent::AgentId::new()).await.unwrap();
//!     println!("Response: {}", response.message.text_content());
//! }
//! ```
//!
//! ## Routing Policies
//!
//! | Policy | Use Case | Selection Criteria |
//! |--------|----------|-------------------|
//! | `CostOptimized` | Budget-conscious workloads | Cheapest provider within latency budget |
//! | `LatencyOptimized` | Real-time/interactive use | Fastest provider within cost budget |
//! | `CapabilityFirst` | Specialized requirements | Context window, vision, tool support |
//! | `LocalFirst` | Privacy/offline preference | Local provider first, cloud fallback |
//! | `Pinned` | Testing/debugging | Specific provider/model only |
//!
//! ## Thread Safety
//!
//! All public types are `Send + Sync`. The router uses `Arc`, `DashMap`, and `tokio::sync`
//! primitives to ensure safe concurrent access from many agent tasks.

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
pub mod stream;
pub mod cost;pub mod policy;
pub mod providers;
pub mod router;

// =============================================================================
// Public API Re-Exports
// =============================================================================

// Core router API
pub use router::{ModelRouter, RouterConfig};

// Provider system
pub use providers::{ModelProvider, ProviderRegistry, ProviderHealth};

// Cost tracking
pub use cost::{CostLedger, CostRecord, CostBudget, BudgetPeriod, CostSummary};

// Routing policies
pub use policy::RoutingPolicy;

// Error types
pub use error::RouterError;

// Convenience prelude
pub mod prelude {
    pub use crate::router::{ModelRouter, RouterConfig};
    pub use crate::providers::{ModelProvider, ProviderRegistry};
    pub use crate::cost::{CostLedger, CostRecord, CostBudget, BudgetPeriod};
    pub use crate::policy::RoutingPolicy;
    pub use crate::error::RouterError;
    pub use nexus_proto::model::{ModelRequest, ModelResponse, Token, ProviderId, RoutingPolicy};
}
