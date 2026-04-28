use std::collections::HashMap;
use std::f32::consts::SQRT_2;
use std::hash::Hasher;
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use nexus_proto::memory::EmbeddingVector;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::{sleep, Duration};
use tracing::{debug, error, instrument, warn};

// =============================================================================
// MemoryError — Embedding-Specific Errors
// =============================================================================

/// Errors that can occur during embedding operations.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("embedding provider error: {0}")]
    ProviderError(String),

    #[error("invalid embedding dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("rate limit exceeded: retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },

    #[error("authentication failed: {0}")]
    AuthFailed(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;

// =============================================================================
// EmbeddingProvider Trait// =============================================================================

/// Abstraction over embedding model providers.
/// All implementations produce L2-normalized float32 vectors.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync + std::fmt::Debug {
    /// Embeds a single text string into a vector.
    async fn embed(&self, text: &str) -> Result<EmbeddingVector>;

    /// Embeds multiple texts in a single batch request (more efficient).
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>>;

    /// Returns the dimensionality of embeddings produced by this provider.
    fn dimensions(&self) -> usize;

    /// Returns the model identifier string (for logging/observability).
    fn model_name(&self) -> &str;
}

// =============================================================================
// FNV-1a Hash — Stable, Fast, Inline Implementation
// =============================================================================

/// FNV-1a 64-bit hash for deterministic feature hashing.
/// Implemented inline to avoid external dependencies.
#[inline]
fn fnv1a_hash(data: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in data.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// =============================================================================
// LocalEmbeddingProvider — Deterministic N-Gram Hash Embedder
// =============================================================================

/// A local, deterministic embedding provider using character n-gram hashing.
///
/// # Algorithm
/// 1. Tokenize input text (split on whitespace/punctuation, lowercase)
/// 2. Extract character bigrams and trigrams from each token
/// 3. Hash each n-gram via FNV-1a into [0, dimensions) bucket
/// 4. Increment bucket count (bag-of-n-grams)
/// 5. L2-normalize the resulting vector///
/// # Properties
/// - Deterministic: same input → same output across runs/machines
/// - No external dependencies or API calls
/// - Reasonable locality: similar texts produce similar vectors
/// - NOT a neural embedding: don't expect semantic understanding
///
/// # Use Cases
/// - Development/testing without API keys
/// - Offline/embedded deployments
/// - Fallback when external providers are unavailable
#[derive(Debug, Clone)]
pub struct LocalEmbeddingProvider {
    dimensions: usize,
}

impl LocalEmbeddingProvider {
    /// Creates a new local embedder with the specified output dimension.
    /// Recommended: 768 or 1536 for compatibility with common models.
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    /// Default constructor with 768 dimensions.
    pub fn default() -> Self {
        Self::new(768)
    }

    /// Tokenizes text into lowercase alphanumeric tokens.
    fn tokenize(text: &str) -> Vec<String> {
        text.chars()
            .map(|c| {
                if c.is_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    ' '
                }
            })
            .collect::<String>()
            .split_whitespace()
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect()
    }

    /// Extracts character bigrams and trigrams from a token.
    fn extract_ngrams(token: &str) -> Vec<String> {
        let mut ngrams = Vec::new();
        let chars: Vec<char> = token.chars().collect();
        if chars.len() >= 2 {
            for i in 0..chars.len() - 1 {
                ngrams.push(format!("{}{}", chars[i], chars[i + 1]));
            }
        }
        if chars.len() >= 3 {
            for i in 0..chars.len() - 2 {
                ngrams.push(format!("{}{}{}", chars[i], chars[i + 1], chars[i + 2]));
            }
        }
        ngrams
    }

    /// L2-normalizes a vector in-place.
    fn l2_normalize(vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    #[instrument(skip(self, text), fields(model = "local-ngram", dims = self.dimensions))]
    async fn embed(&self, text: &str) -> Result<EmbeddingVector> {
        if text.is_empty() {
            return Ok(EmbeddingVector::new(self.dimensions));
        }

        let mut buckets = vec![0u32; self.dimensions];

        // Tokenize and extract n-grams
        for token in Self::tokenize(text) {
            for ngram in Self::extract_ngrams(&token) {
                let hash = fnv1a_hash(&ngram);
                let idx = (hash % self.dimensions as u64) as usize;
                buckets[idx] += 1;
            }
        }

        // Convert counts to f32 and normalize
        let mut vec: Vec<f32> = buckets.iter().map(|&c| c as f32).collect();
        Self::l2_normalize(&mut vec);

        Ok(EmbeddingVector::from_vec(vec))
    }
    #[instrument(skip(self, texts), fields(model = "local-ngram", batch_size = texts.len()))]
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        // Process sequentially to avoid contention; parallelize if needed
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        "local-ngram"
    }
}

// =============================================================================
// OpenAIEmbeddingProvider — OpenAI Embeddings API Client
// =============================================================================

/// Embedding provider using OpenAI's embeddings API.
///
/// # Models
/// - `text-embedding-3-small`: 1536 dims, fast, cost-effective
/// - `text-embedding-3-large`: 3072 dims, higher quality
/// - `text-embedding-ada-002`: 1536 dims, legacy
///
/// # Rate Limits
/// Implements exponential backoff retry on 429 responses.
#[derive(Debug)]
pub struct OpenAIEmbeddingProvider {
    client: Client,
    api_key: String,
    model: String,
    dimensions: usize,
    base_url: String,
    max_retries: u32,
    base_backoff_ms: u64,
}

#[derive(Serialize)]
struct OpenAIEmbedRequest {
    model: String,
    input: Vec<String>,
    encoding_format: String,
}
#[derive(Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<OpenAIEmbedData>,
    model: String,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize)]
struct OpenAIEmbedData {
    embedding: Vec<f64>,
    index: usize,
    object: String,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Deserialize)]
struct OpenAIError {
    error: OpenAIErrorDetail,
}

#[derive(Deserialize)]
struct OpenAIErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

impl OpenAIEmbeddingProvider {
    /// Creates a new OpenAI embedding provider.
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key (never log this)
    /// * `model` - Optional model name; defaults to "text-embedding-3-small"
    pub fn new(api_key: String, model: Option<String>) -> Self {
        let model = model.unwrap_or_else(|| "text-embedding-3-small".to_string());
        let dimensions = match model.as_str() {
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // text-embedding-3-small default
        };

        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))                .build()
                .expect("failed to build HTTP client"),
            api_key,
            model,
            dimensions,
            base_url: "https://api.openai.com/v1".to_string(),
            max_retries: 3,
            base_backoff_ms: 500,
        }
    }

    /// Sets a custom base URL (for Azure OpenAI or compatible endpoints).
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    /// Configures retry behavior for rate limits.
    pub fn with_retry_config(mut self, max_retries: u32, base_backoff_ms: u64) -> Self {
        self.max_retries = max_retries;
        self.base_backoff_ms = base_backoff_ms;
        self
    }

    /// Computes exponential backoff duration for a given attempt.
    fn backoff_duration(&self, attempt: u32) -> Duration {
        let multiplier = 2u64.saturating_pow(attempt);
        let backoff = self.base_backoff_ms.saturating_mul(multiplier);
        // Add jitter: ±25% to avoid thundering herd
        let jitter = (backoff as f64 * 0.25 * rand::random::<f64>()) as u64;
        Duration::from_millis(backoff.saturating_add(jitter).saturating_sub(jitter / 2))
    }

    /// Converts OpenAI's f64 embeddings to our f32 EmbeddingVector with L2 normalization.
    fn convert_embedding(data: &[f64]) -> Result<EmbeddingVector> {
        if data.len() != 1536 && data.len() != 3072 {
            warn!(dims = data.len(), "unexpected embedding dimension from OpenAI");
        }

        let mut vec: Vec<f32> = data.iter().map(|&x| x as f32).collect();

        // L2-normalize for consistency with other providers
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }

        Ok(EmbeddingVector::from_vec(vec))    }

    /// Handles OpenAI API error responses.
    fn handle_api_error(status: StatusCode, body: &str) -> MemoryError {
        match serde_json::from_str::<OpenAIError>(body) {
            Ok(err) => {
                let msg = err.error.message;
                match err.error.error_type.as_str() {
                    "invalid_request_error" => MemoryError::ProviderError(msg),
                    "authentication_error" => MemoryError::AuthFailed(msg),
                    "insufficient_quota" => MemoryError::AuthFailed("account quota exceeded".into()),
                    "rate_limit_exceeded" => {
                        // Try to parse retry-after from headers (caller should provide)
                        MemoryError::RateLimited { retry_after_secs: 60 }
                    }
                    "model_not_found" => MemoryError::ModelNotFound(err.error.code.unwrap_or_default()),
                    _ => MemoryError::ProviderError(format!("{} ({})", msg, err.error.error_type)),
                }
            }
            Err(_) => MemoryError::ProviderError(format!("API error {}: {}", status, body)),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    #[instrument(skip(self, text), fields(model = %self.model))]
    async fn embed(&self, text: &str) -> Result<EmbeddingVector> {
        let batch = vec![text.to_string()];
        let mut results = self.embed_batch(&batch).await?;
        results.pop().ok_or_else(|| MemoryError::Internal("empty batch result".into()))
    }

    #[instrument(skip(self, texts), fields(model = %self.model, batch_size = texts.len()))]
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.base_url);
        let request = OpenAIEmbedRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
            encoding_format: "float".to_string(),
        };

        let mut last_error: Option<MemoryError> = None;

        for attempt in 0..=self.max_retries {
            let response = self.client                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.map_err(|e| {
                        MemoryError::NetworkError(reqwest::Error::new(e.into()))
                    })?;

                    if status.is_success() {
                        let parsed: OpenAIEmbedResponse = serde_json::from_str(&body)
                            .map_err(MemoryError::SerializationError)?;

                        // Convert and order results by original index
                        let mut embeddings = vec![None; texts.len()];
                        for item in parsed.data {
                            if item.index < texts.len() {
                                embeddings[item.index] = Some(Self::convert_embedding(&item.embedding)?);
                            }
                        }

                        // Check for missing results
                        if embeddings.iter().any(|e| e.is_none()) {
                            return Err(MemoryError::Internal("incomplete batch response".into()));
                        }

                        return Ok(embeddings.into_iter().flatten().collect());
                    } else if status == StatusCode::TOO_MANY_REQUESTS {
                        // Rate limit: parse retry-after or use exponential backoff
                        let retry_after = resp
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.parse::<u64>().ok())
                            .unwrap_or(60);

                        warn!(attempt, retry_after, "rate limited by OpenAI");

                        if attempt < self.max_retries {
                            sleep(self.backoff_duration(attempt)).await;
                            continue;
                        } else {
                            return Err(MemoryError::RateLimited { retry_after_secs: retry_after });
                        }
                    } else {                        // Other error: don't retry
                        return Err(Self::handle_api_error(status, &body));
                    }
                }
                Err(e) => {
                    last_error = Some(MemoryError::NetworkError(e));
                    if attempt < self.max_retries {
                        sleep(self.backoff_duration(attempt)).await;
                        continue;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| MemoryError::Internal("embedding request failed".into())))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// =============================================================================
// OllamaEmbeddingProvider — Local Ollama-Compatible API
// =============================================================================

/// Embedding provider using Ollama's `/api/embed` endpoint.
/// Works with any Ollama-compatible server (Ollama, LM Studio, etc.).
#[derive(Debug)]
pub struct OllamaEmbeddingProvider {
    client: Client,
    base_url: String,
    model: String,
    dimensions: usize,
    max_retries: u32,
    base_backoff_ms: u64,
}

#[derive(Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {    embeddings: Vec<Vec<f64>>,
    model: String,
}

impl OllamaEmbeddingProvider {
    /// Creates a new Ollama embedding provider.
    ///
    /// # Arguments
    /// * `base_url` - Ollama server URL (e.g., "http://localhost:11434")
    /// * `model` - Model name (e.g., "nomic-embed-text", "mxbai-embed-large")
    pub fn new(base_url: String, model: String) -> Self {
        // Ollama doesn't expose dimensions via API; common models:
        let dimensions = match model.as_str() {
            "nomic-embed-text" => 768,
            "mxbai-embed-large" => 1024,
            "all-minilm" => 384,
            _ => 768, // default guess
        };

        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60)) // Local models can be slow
                .build()
                .expect("failed to build HTTP client"),
            base_url,
            model,
            dimensions,
            max_retries: 3,
            base_backoff_ms: 250,
        }
    }

    /// Overrides the detected dimensions (use if you know the model's output size).
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    fn backoff_duration(&self, attempt: u32) -> Duration {
        let multiplier = 2u64.saturating_pow(attempt);
        let backoff = self.base_backoff_ms.saturating_mul(multiplier);
        Duration::from_millis(backoff.min(5000)) // Cap at 5s
    }

    fn convert_embedding(data: &[f64]) -> Result<EmbeddingVector> {
        let mut vec: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        // L2-normalize for consistency
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in vec.iter_mut() {                *x /= norm;
            }
        }
        Ok(EmbeddingVector::from_vec(vec))
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaEmbeddingProvider {
    #[instrument(skip(self, text), fields(model = %self.model, url = %self.base_url))]
    async fn embed(&self, text: &str) -> Result<EmbeddingVector> {
        let batch = vec![text.to_string()];
        let mut results = self.embed_batch(&batch).await?;
        results.pop().ok_or_else(|| MemoryError::Internal("empty batch result".into()))
    }

    #[instrument(skip(self, texts), fields(model = %self.model, batch_size = texts.len()))]
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let request = OllamaEmbedRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };

        let mut last_error: Option<MemoryError> = None;

        for attempt in 0..=self.max_retries {
            match self.client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
            {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let parsed: OllamaEmbedResponse = resp.json().await
                            .map_err(MemoryError::SerializationError)?;

                        if parsed.embeddings.len() != texts.len() {
                            return Err(MemoryError::Internal(format!(
                                "embedding count mismatch: expected {}, got {}",
                                texts.len(),
                                parsed.embeddings.len()
                            )));
                        }
                        return parsed.embeddings
                            .into_iter()
                            .map(|emb| Self::convert_embedding(&emb))
                            .collect();
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        return Err(MemoryError::ProviderError(format!(
                            "Ollama API error {}: {}",
                            resp.status(),
                            body
                        )));
                    }
                }
                Err(e) => {
                    last_error = Some(MemoryError::NetworkError(e));
                    if attempt < self.max_retries {
                        sleep(self.backoff_duration(attempt)).await;
                        continue;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| MemoryError::Internal("embedding request failed".into())))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// =============================================================================
// EmbeddingConfig — Provider Selection Configuration
// =============================================================================

/// Configuration for selecting and constructing an embedding provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum EmbeddingConfig {
    /// Use the local n-gram hash embedder (no API key required).
    Local {
        #[serde(default = "default_local_dims")]
        dimensions: usize,
    },
    /// Use OpenAI's embeddings API.
    OpenAI {
        /// Environment variable name containing the API key.
        api_key_env: String,
        /// Model name (default: "text-embedding-3-small").
        #[serde(default = "default_openai_model")]
        model: String,
    },

    /// Use an Ollama-compatible local embedding server.
    Ollama {
        /// Base URL of the Ollama server (e.g., "http://localhost:11434").
        #[serde(default = "default_ollama_url")]
        base_url: String,
        /// Model name to use for embeddings.
        model: String,
    },
}

fn default_local_dims() -> usize { 768 }
fn default_openai_model() -> String { "text-embedding-3-small".to_string() }
fn default_ollama_url() -> String { "http://localhost:11434".to_string() }

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig::Local {
            dimensions: default_local_dims(),
        }
    }
}

impl EmbeddingConfig {
    /// Constructs the configured embedding provider.
    ///
    /// # Errors
    /// - `MemoryError::AuthFailed` if API key env var not found for OpenAI
    /// - `MemoryError::ProviderError` if configuration is invalid
    pub fn build(&self) -> Result<Box<dyn EmbeddingProvider + Send + Sync>> {
        match self {
            EmbeddingConfig::Local { dimensions } => {
                Ok(Box::new(LocalEmbeddingProvider::new(*dimensions)))
            }

            EmbeddingConfig::OpenAI { api_key_env, model } => {
                let api_key = std::env::var(api_key_env)
                    .map_err(|_| MemoryError::AuthFailed(
                        format!("environment variable '{}' not set", api_key_env)
                    ))?;

                Ok(Box::new(OpenAIEmbeddingProvider::new(api_key, Some(model.clone()))))            }

            EmbeddingConfig::Ollama { base_url, model } => {
                Ok(Box::new(OllamaEmbeddingProvider::new(base_url.clone(), model.clone())))
            }
        }
    }

    /// Returns the expected embedding dimensions for this configuration.
    pub fn expected_dimensions(&self) -> usize {
        match self {
            EmbeddingConfig::Local { dimensions } => *dimensions,
            EmbeddingConfig::OpenAI { model, .. } => match model.as_str() {
                "text-embedding-3-large" => 3072,
                _ => 1536,
            },
            EmbeddingConfig::Ollama { model, .. } => match model.as_str() {
                "nomic-embed-text" => 768,
                "mxbai-embed-large" => 1024,
                "all-minilm" => 384,
                _ => 768, // best guess
            },
        }
    }
}

// =============================================================================
// Utility: Cosine Similarity Helper
// =============================================================================

/// Computes cosine similarity between two embedding vectors.
/// Assumes vectors are L2-normalized (dot product = cosine similarity).
#[inline]
pub fn cosine_similarity(a: &EmbeddingVector, b: &EmbeddingVector) -> f32 {
    a.cosine_similarity(b)
}

/// Finds the most similar embedding from a list using cosine similarity.
/// Returns `(index, similarity)` or `None` if list is empty.
pub fn find_most_similar(
    query: &EmbeddingVector,
    candidates: &[EmbeddingVector],
) -> Option<(usize, f32)> {
    candidates
        .iter()
        .enumerate()
        .map(|(i, cand)| (i, cosine_similarity(query, cand)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}
// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_hash_stability() {
        // FNV-1a should produce same hash for same input across runs
        let h1 = fnv1a_hash("hello world");
        let h2 = fnv1a_hash("hello world");
        assert_eq!(h1, h2);

        // Different inputs should produce different hashes (usually)
        let h3 = fnv1a_hash("hello worlD");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_local_tokenize() {
        let tokens = LocalEmbeddingProvider::tokenize("Hello, World! 123");
        assert_eq!(tokens, vec!["hello", "world", "123"]);
    }

    #[test]
    fn test_local_ngrams() {
        let ngrams = LocalEmbeddingProvider::extract_ngrams("test");
        // bigrams: te, es, st; trigrams: tes, est
        assert!(ngrams.contains(&"te".to_string()));
        assert!(ngrams.contains(&"tes".to_string()));
        assert_eq!(ngrams.len(), 5);
    }

    #[tokio::test]
    async fn test_local_embed_basic() {
        let provider = LocalEmbeddingProvider::default();
        let embedding = provider.embed("hello world").await.unwrap();

        assert_eq!(embedding.dims(), 768);

        // Same input should produce same output (deterministic)
        let embedding2 = provider.embed("hello world").await.unwrap();
        assert_eq!(embedding, embedding2);

        // Different input should produce different output (usually)
        let embedding3 = provider.embed("goodbye world").await.unwrap();
        // Cosine similarity should be < 1.0 for different texts
        assert!(cosine_similarity(&embedding, &embedding3) < 1.0);    }

    #[tokio::test]
    async fn test_local_embed_batch() {
        let provider = LocalEmbeddingProvider::new(256);
        let texts = vec!["apple".into(), "banana".into(), "cherry".into()];
        let embeddings = provider.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.dims(), 256);
        }
    }

    #[test]
    fn test_l2_normalization() {
        let mut vec = vec![3.0, 4.0, 0.0];
        LocalEmbeddingProvider::l2_normalize(&mut vec);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_config_build_local() {
        let config = EmbeddingConfig::Local { dimensions: 512 };
        let provider = config.build().unwrap();
        assert_eq!(provider.dimensions(), 512);
        assert_eq!(provider.model_name(), "local-ngram");
    }

    #[test]
    fn test_embedding_config_dimensions() {
        let openai_small = EmbeddingConfig::OpenAI {
            api_key_env: "OPENAI_API_KEY".into(),
            model: "text-embedding-3-small".into(),
        };
        assert_eq!(openai_small.expected_dimensions(), 1536);

        let ollama_nomic = EmbeddingConfig::Ollama {
            base_url: "http://localhost:11434".into(),
            model: "nomic-embed-text".into(),
        };
        assert_eq!(ollama_nomic.expected_dimensions(), 768);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        // Identical normalized vectors: similarity = 1.0
        let v1 = EmbeddingVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v2 = EmbeddingVector::from_vec(vec![1.0, 0.0, 0.0]);        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);

        // Orthogonal vectors: similarity = 0.0
        let v3 = EmbeddingVector::from_vec(vec![0.0, 1.0, 0.0]);
        assert!((cosine_similarity(&v1, &v3) - 0.0).abs() < 1e-5);

        // Opposite vectors: similarity = -1.0
        let v4 = EmbeddingVector::from_vec(vec![-1.0, 0.0, 0.0]);
        assert!((cosine_similarity(&v1, &v4) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_find_most_similar() {
        let query = EmbeddingVector::from_vec(vec![1.0, 0.0, 0.0]);
        let candidates = vec![
            EmbeddingVector::from_vec(vec![0.0, 1.0, 0.0]), // ortho
            EmbeddingVector::from_vec(vec![1.0, 0.0, 0.0]), // identical
            EmbeddingVector::from_vec(vec![0.0, 0.0, 1.0]), // ortho
        ];

        let (idx, sim) = find_most_similar(&query, &candidates).unwrap();
        assert_eq!(idx, 1);
        assert!((sim - 1.0).abs() < 1e-5);
    }
}
