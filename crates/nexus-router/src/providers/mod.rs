use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};

use nexus_proto::model::{ModelRequest, ModelResponse, ProviderId, Token};

use crate::error::RouterError;

pub mod openai;
pub mod anthropic;
pub mod groq;
pub mod mistral;
pub mod local;

// =============================================================================
// ModelProvider Trait
// =============================================================================

/// Core trait for interacting with LLM providers.
/// All providers must implement this to be registered with the Nexus router.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Executes a complete, non-streaming generation request.
    ///
    /// Returns the final `ModelResponse` containing the full assistant message,
    /// token usage, and latency metrics.
    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, RouterError>;

    /// Streams a generation request, yielding tokens as they are produced.
    ///
    /// Returns a pinned, boxed async stream of `Token` items. Each token
    /// contains the text fragment, a final flag, and an optional finish reason.
    async fn stream(
        &self,
        request: &ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, RouterError>> + Send>>, RouterError>;

    /// Pings the provider to verify connectivity and readiness.
    ///
    /// Used by the router's health monitor to mark providers as Healthy,
    /// Degraded, or Down. Should complete quickly (< 5 seconds).
    async fn health_check(&self) -> Result<(), RouterError>;

    /// Returns the canonical identifier for this provider.
    fn provider_id(&self) -> ProviderId;
    /// Lists all model identifiers supported by this provider.
    fn available_models(&self) -> Vec<String>;

    /// Returns the maximum context window size for the specified model.
    fn max_context_tokens(&self, model: &str) -> usize;

    /// Estimates the USD cost for a given token count on the specified model.
    ///
    /// Costs are typically calculated as:
    /// `(input_tokens * input_rate + output_tokens * output_rate) / 1_000_000`
    fn estimate_cost(&self, model: &str, input_tokens: u32, output_tokens: u32) -> f64;

    /// Returns `true` if the provider supports Server-Sent Events (SSE) streaming.
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Returns `true` if the provider supports function/tool calling in the API.
    fn supports_tool_calls(&self) -> bool {
        false
    }
}

// =============================================================================
// ProviderHealth
// =============================================================================

/// Health status of a model provider, tracked by the router's monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderHealth {
    /// Provider is responding normally within acceptable latency thresholds.
    Healthy,

    /// Provider is responding but with elevated latency.
    Degraded {
        /// Last measured round-trip latency in milliseconds.
        latency_ms: u64,
    },

    /// Provider is unreachable or returning errors.
    Down {
        /// Timestamp when the provider was last marked as down.
        since: DateTime<Utc>,
        /// The last error message encountered.
        error: String,
    },
}

impl ProviderHealth {    /// Returns `true` if the provider can safely accept requests.
    pub fn is_operational(&self) -> bool {
        matches!(self, ProviderHealth::Healthy | ProviderHealth::Degraded { .. })
    }
}

// =============================================================================
// ProviderRegistry
// =============================================================================

/// Runtime collection of registered `ModelProvider` instances.
///
/// Thread-safe and optimized for concurrent reads via `DashMap`.
/// Used by the router to discover, select, and invoke providers dynamically.
pub struct ProviderRegistry {
    providers: DashMap<ProviderId, Arc<dyn ModelProvider + Send + Sync>>,
}

impl ProviderRegistry {
    /// Creates a new empty provider registry.
    pub fn new() -> Self {
        Self {
            providers: DashMap::new(),
        }
    }

    /// Registers a provider in the registry.
    ///
    /// Overwrites any existing provider with the same `ProviderId`.
    pub fn register(&self, provider: Arc<dyn ModelProvider + Send + Sync>) {
        let id = provider.provider_id();
        self.providers.insert(id, provider);
    }

    /// Retrieves a provider by its identifier.
    pub fn get(&self, id: &ProviderId) -> Option<Arc<dyn ModelProvider + Send + Sync>> {
        self.providers.get(id).map(|entry| entry.value().clone())
    }

    /// Returns a vector of all registered providers.
    pub fn all(&self) -> Vec<Arc<dyn ModelProvider + Send + Sync>> {
        self.providers.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Returns the identifiers of all registered providers.
    pub fn available_ids(&self) -> Vec<ProviderId> {
        self.providers.iter().map(|entry| entry.key().clone()).collect()
    }
}
