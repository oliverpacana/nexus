use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use bytes::Bytes;
use chrono::Utc;
use eventsource_stream::Eventsource;
use futures::stream::{Stream, StreamExt};
use nexus_proto::model::{
    ContentBlock, FinishReason, Message, MessageRole, ModelId, ModelRequest, ModelResponse,
    ModelUsage, ProviderId, Token, ToolSpec,
};
use reqwest::{Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, instrument, warn};

use crate::error::RouterError;
use crate::providers::ModelProvider;

// =============================================================================
// OpenAI Configuration
// =============================================================================

/// Configuration for the OpenAI provider.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API key for authentication (never log this).
    pub api_key: String,

    /// Base URL for API requests.
    /// Default: "https://api.openai.com/v1"
    /// Override for Azure OpenAI: "https://{resource}.openai.azure.com/openai/deployments/{deployment}"
    pub base_url: String,

    /// Optional organization header for OpenAI API.
    pub organization: Option<String>,

    /// Request timeout in seconds.
    pub timeout_secs: u64,

    /// Maximum retry attempts for transient failures.
    pub max_retries: u32,

    /// Initial backoff delay in milliseconds for exponential retry.    pub retry_backoff_ms: u64,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
            timeout_secs: 120,
            max_retries: 3,
            retry_backoff_ms: 500,
        }
    }
}

// =============================================================================
// OpenAI Provider Implementation
// =============================================================================

/// Provider implementation for OpenAI and OpenAI-compatible APIs.
///
/// Supports:
/// - gpt-4o, gpt-4o-mini, o1, o1-mini, o3-mini
/// - text-embedding-3-small, text-embedding-3-large
/// - Azure OpenAI via base_url override
/// - Any OpenAI-compatible endpoint (LM Studio, Ollama, etc.)
pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: Client,
    model_context_sizes: HashMap<String, usize>,
    model_pricing: HashMap<String, (f64, f64)>, // (input_per_1k, output_per_1k) in USD
}

impl OpenAIProvider {
    /// Creates a new OpenAI provider with the given configuration.
    pub fn new(config: OpenAIConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("failed to build HTTP client");

        // Real context window sizes (as of 2025)
        let mut model_context_sizes = HashMap::new();
        model_context_sizes.insert("gpt-4o".to_string(), 128_000);
        model_context_sizes.insert("gpt-4o-2024-08-06".to_string(), 128_000);
        model_context_sizes.insert("gpt-4o-2024-05-13".to_string(), 128_000);
        model_context_sizes.insert("gpt-4o-mini".to_string(), 128_000);
        model_context_sizes.insert("gpt-4o-mini-2024-07-18".to_string(), 128_000);
        model_context_sizes.insert("o1".to_string(), 128_000);        model_context_sizes.insert("o1-preview".to_string(), 128_000);
        model_context_sizes.insert("o1-mini".to_string(), 128_000);
        model_context_sizes.insert("o3-mini".to_string(), 200_000);
        model_context_sizes.insert("gpt-4-turbo".to_string(), 128_000);
        model_context_sizes.insert("gpt-4".to_string(), 8192);
        model_context_sizes.insert("gpt-3.5-turbo".to_string(), 16385);

        // Real pricing per 1K tokens in USD (as of 2025, OpenAI public pricing)
        let mut model_pricing = HashMap::new();
        // gpt-4o: $5.00 input / $15.00 output per 1M tokens = $0.005 / $0.015 per 1K
        model_pricing.insert("gpt-4o".to_string(), (0.005, 0.015));
        model_pricing.insert("gpt-4o-2024-08-06".to_string(), (0.005, 0.015));
        model_pricing.insert("gpt-4o-2024-05-13".to_string(), (0.005, 0.015));
        // gpt-4o-mini: $0.15 input / $0.60 output per 1M = $0.00015 / $0.0006 per 1K
        model_pricing.insert("gpt-4o-mini".to_string(), (0.00015, 0.0006));
        model_pricing.insert("gpt-4o-mini-2024-07-18".to_string(), (0.00015, 0.0006));
        // o1 series: $15 input / $60 output per 1M = $0.015 / $0.06 per 1K
        model_pricing.insert("o1".to_string(), (0.015, 0.06));
        model_pricing.insert("o1-preview".to_string(), (0.015, 0.06));
        model_pricing.insert("o1-mini".to_string(), (0.003, 0.012));
        model_pricing.insert("o3-mini".to_string(), (0.0015, 0.006));
        // Legacy models
        model_pricing.insert("gpt-4-turbo".to_string(), (0.01, 0.03));
        model_pricing.insert("gpt-4".to_string(), (0.03, 0.06));
        model_pricing.insert("gpt-3.5-turbo".to_string(), (0.0005, 0.0015));

        Self {
            config,
            client,
            model_context_sizes,
            model_pricing,
        }
    }

    /// Computes exponential backoff duration for a given attempt.
    fn backoff_duration(&self, attempt: u32) -> Duration {
        let multiplier = 2u64.saturating_pow(attempt);
        let backoff = self.config.retry_backoff_ms.saturating_mul(multiplier);
        // Add jitter: ±25% to avoid thundering herd
        let jitter = (backoff as f64 * 0.25 * rand::random::<f64>()) as u64;
        Duration::from_millis(backoff.saturating_add(jitter).saturating_sub(jitter / 2))
    }

    /// Converts a nexus_proto Message to OpenAI API format.
    fn convert_message(msg: &Message) -> OpenAIMessage {
        let role = match msg.role {
            MessageRole::System => "system".to_string(),
            MessageRole::User => "user".to_string(),
            MessageRole::Assistant => "assistant".to_string(),
            MessageRole::Tool => "tool".to_string(),        };

        // Handle multi-part content
        let content = if msg.content.len() == 1 {
            // Single block: use string if Text, otherwise serialize
            match &msg.content[0] {
                ContentBlock::Text(s) => Value::String(s.clone()),
                block => serde_json::to_value(block).unwrap_or(Value::Null),
            }
        } else {
            // Multiple blocks: serialize as array
            serde_json::to_value(&msg.content).unwrap_or(Value::Null)
        };

        OpenAIMessage { role, content }
    }

    /// Converts OpenAI API response to nexus_proto ModelResponse.
    fn convert_response(
        openai_resp: &OpenAIResponse,
        model: &str,
        latency_ms: u64,
    ) -> Result<ModelResponse, RouterError> {
        let choice = openai_resp
            .choices
            .first()
            .ok_or_else(|| RouterError::ProviderError("empty choices in response".into()))?;

        let message = Message {
            role: match choice.message.role.as_str() {
                "system" => MessageRole::System,
                "user" => MessageRole::User,
                "assistant" => MessageRole::Assistant,
                "tool" => MessageRole::Tool,
                _ => MessageRole::Assistant,
            },
            content: parse_openai_content(&choice.message.content)?,
        };

        let usage = ModelUsage {
            prompt_tokens: openai_resp.usage.prompt_tokens,
            completion_tokens: openai_resp.usage.completion_tokens,
            total_tokens: openai_resp.usage.total_tokens,
            estimated_cost_usd: 0.0, // Filled by router cost calculator
        };

        Ok(ModelResponse {
            id: uuid::Uuid::parse_str(&openai_resp.id).unwrap_or_else(|_| uuid::Uuid::new_v4()),
            model: ModelId::new(crate::providers::ProviderId::OpenAI, model),
            message,            usage,
            latency_ms,
        })
    }

    /// Parses OpenAI content field (string or array) into nexus_proto ContentBlock vec.
    fn parse_content_field(content: &Value) -> Result<Vec<ContentBlock>, RouterError> {
        if let Some(s) = content.as_str() {
            return Ok(vec![ContentBlock::Text(s.to_string())]);
        }

        if let Some(arr) = content.as_array() {
            let mut blocks = Vec::new();
            for item in arr {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    blocks.push(ContentBlock::Text(text.to_string()));
                } else if let Some(tc) = item.get("tool_call").or(item.get("tool_calls")) {
                    // Parse tool call
                    if let Some(name) = tc.get("name").and_then(|v| v.as_str()) {
                        let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let args = tc.get("arguments")
                            .cloned()
                            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                        blocks.push(ContentBlock::ToolCall {
                            id,
                            name: name.to_string(),
                            arguments: args,
                        });
                    }
                }
            }
            if !blocks.is_empty() {
                return Ok(blocks);
            }
        }

        // Fallback: serialize whatever we have as text
        Ok(vec![ContentBlock::Text(content.to_string())])
    }

    /// Converts nexus_proto ToolSpec to OpenAI function definition.
    fn convert_tool(tool: &ToolSpec) -> OpenAITool {
        OpenAITool {
            r#type: "function".to_string(),
            function: OpenAIFunction {
                name: tool.name.clone(),
                description: Some(tool.description.clone()),
                parameters: tool.parameters.clone(),
            },
        }    }

    /// Builds the full API URL for chat completions.
    fn chat_completions_url(&self) -> String {
        if self.config.base_url.contains("/deployments/") {
            // Azure OpenAI format: base_url already includes deployment
            format!("{}/chat/completions?api-version=2024-06-01", self.config.base_url)
        } else {
            format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'))
        }
    }
}

#[async_trait]
impl ModelProvider for OpenAIProvider {
    #[instrument(skip(self, request), fields(model = ?request.model))]
    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, RouterError> {
        let model = request
            .model
            .as_ref()
            .map(|m| m.model.clone())
            .unwrap_or_else(|| "gpt-4o-mini".to_string());

        let url = self.chat_completions_url();
        let openai_req = self.build_openai_request(request, &model, false)?;

        let mut last_error: Option<RouterError> = None;

        for attempt in 0..=self.config.max_retries {
            match self.execute_request(&url, &openai_req, false).await {
                Ok(response) => {
                    let openai_resp: OpenAIResponse = response
                        .json()
                        .await
                        .map_err(|e| RouterError::ProviderError(format!("response parse failed: {}", e)))?;

                    let latency_ms = openai_resp.usage.total_tokens as u64; // Placeholder; real latency from headers or timing

                    return Self::convert_response(&openai_resp, &model, latency_ms);
                }
                Err(e) => {
                    last_error = Some(e.clone());
                    if should_retry(&e) && attempt < self.config.max_retries {
                        sleep(self.backoff_duration(attempt)).await;
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| RouterError::ProviderError("request failed after retries".into())))
    }

    #[instrument(skip(self, request), fields(model = ?request.model))]
    async fn stream(
        &self,
        request: &ModelRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, RouterError>> + Send>>, RouterError> {
        let model = request
            .model
            .as_ref()
            .map(|m| m.model.clone())
            .unwrap_or_else(|| "gpt-4o-mini".to_string());

        let url = self.chat_completions_url();
        let openai_req = self.build_openai_request(request, &model, true)?;

        let response = self
            .execute_request(&url, &openai_req, true)
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(handle_api_error(status, &body));
        }

        // Parse SSE stream
        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(|event| async {
                match event {
                    Ok(ev) if ev.data == "[DONE]" => None,
                    Ok(ev) if ev.data.starts_with('{') => {
                        match serde_json::from_str::<OpenAIStreamChunk>(&ev.data) {
                            Ok(chunk) => Some(Ok(chunk)),
                            Err(e) => {
                                warn!(error = %e, "failed to parse stream chunk");
                                None
                            }
                        }
                    }
                    Ok(_) => None, // Skip non-JSON events
                    Err(e) => {
                        warn!(error = %e, "SSE stream error");
                        Some(Err(RouterError::ProviderError(format!("stream error: {}", e))))
                    }
                }            })
            .filter_map(move |chunk_result| {
                let model_clone = model.clone();
                async move {
                    match chunk_result {
                        Ok(chunk) => convert_stream_chunk(chunk, &model_clone),
                        Err(e) => Some(Err(e)),
                    }
                }
            });

        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> Result<(), RouterError> {
        // Simple health check: call /models endpoint
        let url = format!("{}/models", self.config.base_url.trim_end_matches('/'));
        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await
            .map_err(|e| RouterError::ProviderError(format!("health check failed: {}", e)))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(handle_api_error(status, &body))
        }
    }

    fn provider_id(&self) -> ProviderId {
        ProviderId::OpenAI
    }

    fn available_models(&self) -> Vec<String> {
        self.model_context_sizes.keys().cloned().collect()
    }

    fn max_context_tokens(&self, model: &str) -> usize {
        self.model_context_sizes
            .get(model)
            .copied()
            .unwrap_or(128_000) // Default to gpt-4o size
    }

    fn estimate_cost(&self, model: &str, input_tokens: u32, output_tokens: u32) -> f64 {        let (input_rate, output_rate) = self
            .model_pricing
            .get(model)
            .copied()
            .unwrap_or((0.005, 0.015)); // Default to gpt-4o pricing

        // Rates are per 1K tokens
        (input_tokens as f64 * input_rate + output_tokens as f64 * output_rate) / 1000.0
    }
}

// =============================================================================
// Internal OpenAI API Types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: Value,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIChoice {
    index: usize,    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIStreamChunk {
    id: Option<String>,
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIStreamDelta {
    role: Option<String>,
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAIStreamToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIStreamToolCall {
    index: usize,
    id: Option<String>,
    #[serde(rename = "type")]
    tool_type: Option<String>,
    function: Option<OpenAIStreamFunction>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAITool {
    r#type: String,
    function: OpenAIFunction,}

#[derive(Debug, Clone, Serialize)]
struct OpenAIFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

// =============================================================================
// Helper Functions
// =============================================================================

impl OpenAIProvider {
    /// Builds an OpenAI API request from a nexus_proto ModelRequest.
    fn build_openai_request(
        &self,
        request: &ModelRequest,
        model: &str,
        stream: bool,
    ) -> Result<OpenAIRequest, RouterError> {
        let messages = request
            .messages
            .iter()
            .map(Self::convert_message)
            .collect();

        let tools = if !request.tools.is_empty() {
            Some(request.tools.iter().map(Self::convert_tool).collect())
        } else {
            None
        };

        Ok(OpenAIRequest {
            model: model.to_string(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream,
            tools,
            tool_choice: None, // Could be configurable
        })
    }

    /// Executes an HTTP request to the OpenAI API with proper headers and error handling.
    async fn execute_request(
        &self,
        url: &str,        body: &OpenAIRequest,
        stream: bool,
    ) -> Result<Response, RouterError> {
        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        if let Some(org) = &self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

        let response = req
            .json(body)
            .send()
            .await
            .map_err(|e| RouterError::ProviderError(format!("request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(handle_api_error(status, &body));
        }

        Ok(response)
    }
}

/// Parses OpenAI stream chunk into nexus_proto Token.
fn convert_stream_chunk(
    chunk: OpenAIStreamChunk,
    _model: &str,
) -> Option<Result<Token, RouterError>> {
    let choice = chunk.choices.first()?;
    let delta = &choice.delta;

    // Handle tool calls in streaming
    for tc in &delta.tool_calls {
        if let Some(func) = &tc.function {
            if let (Some(name), Some(args)) = (&func.name, &func.arguments) {
                // Parse arguments as JSON
                let args_value: Value = serde_json::from_str(args).unwrap_or_else(|_| {
                    json!({"_parse_error": args})
                });
                return Some(Ok(Token {
                    text: String::new(), // Tool calls don't emit text tokens
                    is_final: false,
                    finish_reason: None,
                }));            }
        }
    }

    // Handle text content
    if let Some(content) = &delta.content {
        if !content.is_empty() {
            return Some(Ok(Token {
                text: content.clone(),
                is_final: false,
                finish_reason: None,
            }));
        }
    }

    // Handle finish reason
    if let Some(reason) = &choice.finish_reason {
        let finish = match reason.as_str() {
            "stop" => Some(FinishReason::Stop),
            "length" => Some(FinishReason::MaxTokens),
            "tool_calls" => Some(FinishReason::ToolCall),
            "content_filter" => Some(FinishReason::ContentFilter),
            _ => Some(FinishReason::Error(reason.clone())),
        };
        return Some(Ok(Token {
            text: String::new(),
            is_final: true,
            finish_reason: finish,
        }));
    }

    None
}

/// Parses OpenAI content field into nexus_proto ContentBlock vec.
fn parse_openai_content(content: &Value) -> Result<Vec<ContentBlock>, RouterError> {
    if let Some(s) = content.as_str() {
        return Ok(vec![ContentBlock::Text(s.to_string())]);
    }

    if let Some(arr) = content.as_array() {
        let mut blocks = Vec::new();
        for item in arr {
            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                blocks.push(ContentBlock::Text(text.to_string()));
            } else if let Some(tc) = item.get("tool_call").or(item.get("tool_calls")) {
                if let Some(name) = tc.get("name").and_then(|v| v.as_str()) {
                    let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let args = tc.get("arguments")
                        .cloned()                        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                    blocks.push(ContentBlock::ToolCall {
                        id,
                        name: name.to_string(),
                        arguments: args,
                    });
                }
            }
        }
        if !blocks.is_empty() {
            return Ok(blocks);
        }
    }

    Ok(vec![ContentBlock::Text(content.to_string())])
}

/// Determines if an error is retryable.
fn should_retry(err: &RouterError) -> bool {
    matches!(
        err,
        RouterError::ProviderUnavailable(_)
            | RouterError::Timeout { .. }
            | RouterError::ProviderError(msg) if msg.contains("5") || msg.contains("429")
    )
}

/// Maps HTTP status and body to RouterError.
fn handle_api_error(status: StatusCode, body: &str) -> RouterError {
    let message = parse_openai_error(body).unwrap_or_else(|| body.to_string());

    match status {
        StatusCode::UNAUTHORIZED => RouterError::ProviderError(format!("authentication failed: {}", message)),
        StatusCode::FORBIDDEN => RouterError::ProviderError(format!("access denied: {}", message)),
        StatusCode::NOT_FOUND => RouterError::ProviderError(format!("endpoint not found: {}", message)),
        StatusCode::TOO_MANY_REQUESTS => {
            // Try to parse retry-after from body or headers (caller should handle)
            RouterError::ProviderError(format!("rate limit exceeded: {}", message))
        }
        StatusCode::INTERNAL_SERVER_ERROR
        | StatusCode::BAD_GATEWAY
        | StatusCode::SERVICE_UNAVAILABLE => {
            RouterError::ProviderUnavailable(format!("provider error {}: {}", status, message))
        }
        _ => RouterError::ProviderError(format!("API error {}: {}", status, message)),
    }
}

/// Parses OpenAI error response body.
fn parse_openai_error(body: &str) -> Option<String> {    #[derive(Deserialize)]
    struct OpenAIError {
        error: OpenAIErrorDetail,
    }
    #[derive(Deserialize)]
    struct OpenAIErrorDetail {
        message: String,
        #[serde(rename = "type")]
        error_type: String,
    }

    serde_json::from_str::<OpenAIError>(body)
        .ok()
        .map(|e| format!("{} ({})", e.error.message, e.error.error_type))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_proto::model::{AgentId, RoutingPolicy};

    #[test]
    fn test_config_defaults() {
        let config = OpenAIConfig::default();
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_model_context_sizes() {
        let provider = OpenAIProvider::new(OpenAIConfig::default());
        assert_eq!(provider.max_context_tokens("gpt-4o"), 128_000);
        assert_eq!(provider.max_context_tokens("gpt-4o-mini"), 128_000);
        assert_eq!(provider.max_context_tokens("o1"), 128_000);
        assert_eq!(provider.max_context_tokens("unknown-model"), 128_000); // Default fallback
    }

    #[test]
    fn test_model_pricing() {
        let provider = OpenAIProvider::new(OpenAIConfig::default());
        // gpt-4o: $0.005 input, $0.015 output per 1K tokens
        let cost = provider.estimate_cost("gpt-4o", 1000, 500);
        assert!((cost - 0.0125).abs() < 0.0001); // (1000*0.005 + 500*0.015)/1000

        // gpt-4o-mini: $0.00015 input, $0.0006 output per 1K        let cost = provider.estimate_cost("gpt-4o-mini", 1000, 500);
        assert!((cost - 0.00045).abs() < 0.00001);
    }

    #[test]
    fn test_convert_message() {
        let msg = Message {
            role: MessageRole::User,
            content: vec![ContentBlock::Text("Hello".to_string())],
        };
        let converted = OpenAIProvider::convert_message(&msg);
        assert_eq!(converted.role, "user");
        assert_eq!(converted.content, Value::String("Hello".to_string()));
    }

    #[test]
    fn test_backoff_duration() {
        let provider = OpenAIProvider::new(OpenAIConfig {
            retry_backoff_ms: 100,
            ..Default::default()
        });

        let d0 = provider.backoff_duration(0);
        let d1 = provider.backoff_duration(1);
        let d2 = provider.backoff_duration(2);

        assert!(d0.as_millis() >= 75 && d0.as_millis() <= 125); // 100ms ±25%
        assert!(d1.as_millis() >= 150 && d1.as_millis() <= 250); // 200ms ±25%
        assert!(d2.as_millis() >= 300 && d2.as_millis() <= 500); // 400ms ±25%
    }

    #[test]
    fn test_should_retry() {
        assert!(should_retry(&RouterError::ProviderUnavailable("502".into())));
        assert!(should_retry(&RouterError::Timeout {
            operation: "test".into(),
            duration_ms: 5000
        }));
        assert!(!should_retry(&RouterError::ProviderError("invalid request".into())));
    }

    #[test]
    fn test_parse_openai_error() {
        let body = r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error"}}"#;
        let parsed = parse_openai_error(body);
        assert!(parsed.unwrap().contains("Incorrect API key"));

        let bad_body = "not json";
        assert!(parse_openai_error(bad_body).is_none());
    }
    #[tokio::test]
    async fn test_health_check_mock() {
        // This would require a real API key and network access
        // For unit testing, we just verify the method exists and compiles
        let provider = OpenAIProvider::new(OpenAIConfig {
            api_key: "sk-test".to_string(),
            ..Default::default()
        });
        // Skip actual network call in unit tests
        // let result = provider.health_check().await;
        // assert!(result.is_err()); // Expected to fail with invalid key
    }
}
