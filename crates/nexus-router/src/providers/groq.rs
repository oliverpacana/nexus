// crates/nexus-router/src/providers/groq.rs

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use nexus_proto::model::{ModelRequest, ModelResponse, ProviderId, Token};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use tracing::{debug, instrument, warn};

use crate::error::RouterError;
use crate::providers::{openai::OpenAIProvider, ModelProvider};

// =============================================================================
// Groq Configuration
// =============================================================================

/// Configuration for the Groq provider.
#[derive(Debug, Clone)]
pub struct GroqConfig {
    /// API key for authentication.
    pub api_key: String,

    /// Base URL for Groq's OpenAI-compatible API.
    /// Default: "https://api.groq.com/openai/v1"
    pub base_url: String,

    /// Request timeout in seconds.
    pub timeout_secs: u64,

    /// Maximum retry attempts for transient failures.
    pub max_retries: u32,

    /// Initial backoff delay in milliseconds for exponential retry.
    pub retry_backoff_ms: u64,
}

impl Default for GroqConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("GROQ_API_KEY").unwrap_or_default(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            timeout_secs: 30, // Groq is fast; shorter timeout is reasonable
            max_retries: 3,
            retry_backoff_ms: 250,        }
    }
}

// =============================================================================
// Groq Provider Implementation
// =============================================================================

/// Provider implementation for Groq's ultra-low-latency inference API.
///
/// Groq uses an OpenAI-compatible API endpoint, so we can reuse most of the
/// OpenAI provider logic while customizing:
/// - Base URL and headers
/// - Model-specific context sizes and pricing
/// - Health check that measures actual latency (Groq's key differentiator)
///
/// Supported models (as of 2025):
/// - llama-3.1-70b-versatile: 128K context, $0.59/$0.79 per 1M tokens
/// - llama-3.1-8b-instant: 128K, $0.05/$0.08
/// - llama-3.2-90b-vision-preview: 8K, $0.90/$0.90
/// - mixtral-8x7b-32768: 32K, $0.24/$0.24
/// - gemma2-9b-it: 8K, $0.20/$0.20
pub struct GroqProvider {
    // Reuse OpenAI provider for core logic
    inner: OpenAIProvider,
    // Groq-specific model metadata
    model_context_sizes: HashMap<String, usize>,
    model_pricing: HashMap<String, (f64, f64)>, // (input_per_1M, output_per_1M) in USD
    // Cached latency from last health check
    last_latency_ms: std::sync::atomic::AtomicU64,
}

impl GroqProvider {
    /// Creates a new Groq provider with the given configuration.
    pub fn new(config: GroqConfig) -> Self {
        // Build OpenAI config pointed at Groq's endpoint
        let openai_config = crate::providers::openai::OpenAIConfig {
            api_key: config.api_key,
            base_url: config.base_url,
            organization: None,
            timeout_secs: config.timeout_secs,
            max_retries: config.max_retries,
            retry_backoff_ms: config.retry_backoff_ms,
        };

        // Groq model context sizes (real values)
        let mut model_context_sizes = HashMap::new();
        model_context_sizes.insert("llama-3.1-70b-versatile".to_string(), 128_000);
        model_context_sizes.insert("llama-3.1-8b-instant".to_string(), 128_000);
        model_context_sizes.insert("llama-3.2-90b-vision-preview".to_string(), 8_192);        model_context_sizes.insert("mixtral-8x7b-32768".to_string(), 32_768);
        model_context_sizes.insert("gemma2-9b-it".to_string(), 8_192);

        // Groq pricing per 1M tokens in USD (real public pricing)
        let mut model_pricing = HashMap::new();
        model_pricing.insert("llama-3.1-70b-versatile".to_string(), (0.59, 0.79));
        model_pricing.insert("llama-3.1-8b-instant".to_string(), (0.05, 0.08));
        model_pricing.insert("llama-3.2-90b-vision-preview".to_string(), (0.90, 0.90));
        model_pricing.insert("mixtral-8x7b-32768".to_string(), (0.24, 0.24));
        model_pricing.insert("gemma2-9b-it".to_string(), (0.20, 0.20));

        Self {
            inner: OpenAIProvider::new(openai_config),
            model_context_sizes,
            model_pricing,
            last_latency_ms: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Returns the last measured latency from health checks.
    pub fn last_measured_latency_ms(&self) -> u64 {
        self.last_latency_ms.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl ModelProvider for GroqProvider {
    #[instrument(skip(self, request), fields(model = ?request.model))]
    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, RouterError> {
        // Delegate to inner OpenAI provider
        self.inner.complete(request).await
    }

    #[instrument(skip(self, request), fields(model = ?request.model))]
    async fn stream(
        &self,
        request: &ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, RouterError>> + Send>>, RouterError> {
        // Delegate to inner OpenAI provider
        self.inner.stream(request).await
    }

    /// Health check that measures actual round-trip latency to Groq.
    /// Groq's key differentiator is ultra-low latency; we track this for routing decisions.
    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<(), RouterError> {
        let start = Instant::now();

        // Use a minimal request to the /models endpoint for health check
        let url = format!("{}/models", self.inner.provider_id().to_string().replace("openai", &self.inner.config.base_url.replace("https://", "").replace("/v1", "")));        
        // Actually, Groq's /models endpoint works like OpenAI's
        let base = self.inner.config.base_url.trim_end_matches("/chat/completions").trim_end_matches('/');
        let url = format!("{}/models", base);

        let response = self
            .inner
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.inner.config.api_key))
            .send()
            .await;

        let latency_ms = start.elapsed().as_millis() as u64;
        self.last_latency_ms
            .store(latency_ms, std::sync::atomic::Ordering::Relaxed);

        match response {
            Ok(resp) if resp.status().is_success() => {
                debug!(latency_ms, "Groq health check passed");
                Ok(())
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                warn!(status = %status, body = %body, "Groq health check failed");
                Err(RouterError::ProviderUnavailable(format!(
                    "health check failed with status {}: {}",
                    status, body
                )))
            }
            Err(e) => {
                warn!(error = %e, "Groq health check request failed");
                Err(RouterError::ProviderUnavailable(format!(
                    "health check request failed: {}",
                    e
                )))
            }
        }
    }

    fn provider_id(&self) -> ProviderId {
        ProviderId::Groq
    }

    fn available_models(&self) -> Vec<String> {
        self.model_context_sizes.keys().cloned().collect()
    }

    fn max_context_tokens(&self, model: &str) -> usize {        self.model_context_sizes
            .get(model)
            .copied()
            .unwrap_or(128_000) // Default to 70B size
    }

    fn estimate_cost(&self, model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
        let (input_rate, output_rate) = self
            .model_pricing
            .get(model)
            .copied()
            .unwrap_or((0.59, 0.79)); // Default to 70B pricing

        // Rates are per 1M tokens
        (input_tokens as f64 * input_rate + output_tokens as f64 * output_rate) / 1_000_000.0
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }

    fn supports_tool_calls(&self) -> bool {
        // Groq supports tool calls for most models
        true
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
        let config = GroqConfig::default();
        assert_eq!(config.base_url, "https://api.groq.com/openai/v1");
        assert_eq!(config.timeout_secs, 30); // Shorter than OpenAI default
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_model_context_sizes() {
        let provider = GroqProvider::new(GroqConfig::default());
        assert_eq!(provider.max_context_tokens("llama-3.1-70b-versatile"), 128_000);
        assert_eq!(provider.max_context_tokens("llama-3.1-8b-instant"), 128_000);
        assert_eq!(provider.max_context_tokens("mixtral-8x7b-32768"), 32_768);
        assert_eq!(provider.max_context_tokens("gemma2-9b-it"), 8_192);        assert_eq!(provider.max_context_tokens("unknown"), 128_000); // Default
    }

    #[test]
    fn test_model_pricing() {
        let provider = GroqProvider::new(GroqConfig::default());
        
        // llama-3.1-70b: $0.59 input / $0.79 output per 1M
        let cost = provider.estimate_cost("llama-3.1-70b-versatile", 100_000, 50_000);
        assert!((cost - 0.0985).abs() < 0.0001); // (100k*0.59 + 50k*0.79)/1M

        // llama-3.1-8b: $0.05 input / $0.08 output per 1M
        let cost = provider.estimate_cost("llama-3.1-8b-instant", 100_000, 50_000);
        assert!((cost - 0.009).abs() < 0.00001); // (100k*0.05 + 50k*0.08)/1M
    }

    #[test]
    fn test_provider_id() {
        let provider = GroqProvider::new(GroqConfig::default());
        assert_eq!(provider.provider_id(), ProviderId::Groq);
    }

    #[test]
    fn test_available_models() {
        let provider = GroqProvider::new(GroqConfig::default());
        let models = provider.available_models();
        assert!(models.contains(&"llama-3.1-70b-versatile".to_string()));
        assert!(models.contains(&"mixtral-8x7b-32768".to_string()));
        assert!(!models.contains(&"gpt-4o".to_string())); // OpenAI model, not Groq
    }
}
