// crates/nexus-router/src/providers/local.rs

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use nexus_proto::model::{ModelRequest, ModelResponse, ProviderId, Token};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::error::RouterError;
use crate::providers::{openai::OpenAIProvider, ModelProvider};

// =============================================================================
// Local Configuration
// =============================================================================

/// Configuration for local/OpenAI-compatible inference servers.
///
/// Supports:
/// - Ollama: http://localhost:11434/v1
/// - llama.cpp server: http://localhost:8080
/// - Any OpenAI-compatible endpoint (LM Studio, Text Generation WebUI, etc.)
#[derive(Debug, Clone)]
pub struct LocalConfig {
    /// Base URL for the local inference server.
    /// Examples:
    /// - Ollama: "http://localhost:11434/v1"
    /// - llama.cpp: "http://localhost:8080"
    pub base_url: String,

    /// Request timeout in seconds.
    pub timeout_secs: u64,

    /// Default model to use when request doesn't specify one.
    pub default_model: String,

    /// Default max context tokens for cost estimation (local models vary).
    pub max_context_tokens: usize,

    /// Whether the server supports tool/function calling.
    /// Some local models support this; others don't.
    pub supports_tool_calls: bool,

    /// Optional API key (some local servers support auth).
    pub api_key: Option<String>,}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434/v1".to_string(),
            timeout_secs: 120, // Local can be slower; longer timeout
            default_model: "llama3.2".to_string(),
            max_context_tokens: 8192, // Conservative default
            supports_tool_calls: false, // Most local models don't support this well yet
            api_key: None,
        }
    }
}

// =============================================================================
// Local Provider Implementation
// =============================================================================

/// Provider for local/OpenAI-compatible inference servers.
///
/// Wraps an `OpenAIProvider` pointed at a custom base_url, with adjustments for:
/// - Zero-cost pricing (local = free)
/// - Dynamic model discovery via /models endpoint
/// - Configurable tool call support
/// - Health check against local endpoint
pub struct LocalProvider {
    // Reuse OpenAI provider for core request/response logic
    inner: OpenAIProvider,
    // Local-specific configuration
    config: LocalConfig,
    // Cached model list from /models endpoint
    cached_models: std::sync::RwLock<Vec<String>>,
    // Cached context sizes (populated from model metadata if available)
    context_sizes: std::sync::RwLock<HashMap<String, usize>>,
}

impl LocalProvider {
    /// Creates a new local provider with the given configuration.
    pub fn new(config: LocalConfig) -> Self {
        // Build OpenAI config pointed at local endpoint
        let openai_config = crate::providers::openai::OpenAIConfig {
            api_key: config.api_key.clone().unwrap_or_default(),
            base_url: config.base_url.clone(),
            organization: None,
            timeout_secs: config.timeout_secs,
            max_retries: 1, // Local: fewer retries; fail fast
            retry_backoff_ms: 100,
        };
        Self {
            inner: OpenAIProvider::new(openai_config),
            config,
            cached_models: std::sync::RwLock::new(Vec::new()),
            context_sizes: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Refreshes the cached model list by querying the /models endpoint.
    /// Should be called periodically or on cache miss.
    pub async fn refresh_models(&self) -> Result<(), RouterError> {
        let base = self.config.base_url.trim_end_matches("/chat/completions").trim_end_matches('/');
        let url = format!("{}/models", base);

        let response = self
            .inner
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| RouterError::ProviderError(format!("failed to fetch models: {}", e)))?;

        if !response.status().is_success() {
            return Err(RouterError::ProviderError(format!(
                "models endpoint returned {}",
                response.status()
            )));
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelInfo>,
        }
        #[derive(Deserialize)]
        struct ModelInfo {
            id: String,
            #[serde(default)]
            context_length: Option<usize>,
        }

        let models_resp: ModelsResponse = response
            .json()
            .await
            .map_err(|e| RouterError::ProviderError(format!("failed to parse models: {}", e)))?;

        let mut models = Vec::new();
        let mut sizes = HashMap::new();

        for model in models_resp.data {
            models.push(model.id.clone());            if let Some(ctx) = model.context_length {
                sizes.insert(model.id, ctx);
            }
        }

        // Update caches
        *self.cached_models.write().unwrap() = models;
        *self.context_sizes.write().unwrap() = sizes;

        debug!(count = self.cached_models.read().unwrap().len(), "refreshed local model list");
        Ok(())
    }

    /// Returns the configured default model.
    pub fn default_model(&self) -> &str {
        &self.config.default_model
    }
}

#[async_trait]
impl ModelProvider for LocalProvider {
    #[instrument(skip(self, request), fields(model = ?request.model, base_url = %self.config.base_url))]
    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, RouterError> {
        // Ensure we have a model; use default if not specified
        let mut req = request.clone();
        if req.model.is_none() {
            req.model = Some(nexus_proto::model::ModelId::new(
                ProviderId::Local,
                &self.config.default_model,
            ));
        }

        self.inner.complete(&req).await
    }

    #[instrument(skip(self, request), fields(model = ?request.model))]
    async fn stream(
        &self,
        request: &ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, RouterError>> + Send>>, RouterError> {
        let mut req = request.clone();
        if req.model.is_none() {
            req.model = Some(nexus_proto::model::ModelId::new(
                ProviderId::Local,
                &self.config.default_model,
            ));
        }

        self.inner.stream(&req).await
    }
    /// Health check: ping the local server's /models endpoint.
    /// Returns Down if the server is unreachable.
    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<(), RouterError> {
        let base = self.config.base_url.trim_end_matches("/chat/completions").trim_end_matches('/');
        let url = format!("{}/models", base);

        match self
            .inner
            .client
            .get(&url)
            .timeout(Duration::from_secs(5)) // Short timeout for health check
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                // Optionally refresh model cache on successful health check
                let _ = self.refresh_models().await;
                debug!("local provider health check passed");
                Ok(())
            }
            Ok(resp) => {
                let status = resp.status();
                warn!(status = %status, "local provider health check returned non-success");
                Err(RouterError::ProviderUnavailable(format!(
                    "health check failed with status {}",
                    status
                )))
            }
            Err(e) => {
                warn!(error = %e, "local provider health check request failed");
                Err(RouterError::ProviderUnavailable(format!(
                    "health check request failed: {}",
                    e
                )))
            }
        }
    }

    fn provider_id(&self) -> ProviderId {
        ProviderId::Local
    }

    fn available_models(&self) -> Vec<String> {
        let cached = self.cached_models.read().unwrap();
        if cached.is_empty() {
            // Return default model if cache is empty
            vec![self.config.default_model.clone()]
        } else {            cached.clone()
        }
    }

    fn max_context_tokens(&self, model: &str) -> usize {
        // Check cached context sizes first
        let cached = self.context_sizes.read().unwrap();
        if let Some(&size) = cached.get(model) {
            return size;
        }
        drop(cached);

        // Fall back to configured default
        self.config.max_context_tokens
    }

    fn estimate_cost(&self, _model: &str, _input_tokens: u32, _output_tokens: u32) -> f64 {
        // Local inference is free (excluding hardware/electricity)
        0.0
    }

    fn supports_streaming(&self) -> bool {
        // Most OpenAI-compatible local servers support streaming
        true
    }

    fn supports_tool_calls(&self) -> bool {
        self.config.supports_tool_calls
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = LocalConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434/v1");
        assert_eq!(config.default_model, "llama3.2");
        assert_eq!(config.max_context_tokens, 8192);
        assert!(!config.supports_tool_calls);
        assert!(config.api_key.is_none());
    }

    #[test]    fn test_zero_cost() {
        let provider = LocalProvider::new(LocalConfig::default());
        assert_eq!(provider.estimate_cost("any-model", 100_000, 50_000), 0.0);
    }

    #[test]
    fn test_default_model_fallback() {
        let provider = LocalProvider::new(LocalConfig {
            default_model: "mistral".to_string(),
            ..Default::default()
        });

        // Empty cache should return default model
        let models = provider.available_models();
        assert_eq!(models, vec!["mistral"]);
    }

    #[test]
    fn test_context_size_cache() {
        let provider = LocalProvider::new(LocalConfig {
            max_context_tokens: 4096,
            ..Default::default()
        });

        // Unknown model should return configured default
        assert_eq!(provider.max_context_tokens("unknown"), 4096);

        // Populate cache
        {
            let mut sizes = provider.context_sizes.write().unwrap();
            sizes.insert("custom-model".to_string(), 32768);
        }

        // Cached model should return cached size
        assert_eq!(provider.max_context_tokens("custom-model"), 32768);
    }

    #[test]
    fn test_provider_id() {
        let provider = LocalProvider::new(LocalConfig::default());
        assert_eq!(provider.provider_id(), ProviderId::Local);
    }

    #[test]
    fn test_tool_call_support() {
        let provider_no_tools = LocalProvider::new(LocalConfig {
            supports_tool_calls: false,
            ..Default::default()
        });
        assert!(!provider_no_tools.supports_tool_calls());
        let provider_with_tools = LocalProvider::new(LocalConfig {
            supports_tool_calls: true,
            ..Default::default()
        });
        assert!(provider_with_tools.supports_tool_calls());
    }
}
