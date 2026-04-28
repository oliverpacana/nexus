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
use tracing::{debug, error, instrument, warn};

use crate::error::RouterError;
use crate::providers::ModelProvider;

// =============================================================================
// Anthropic Configuration
// =============================================================================

/// Configuration for the Anthropic provider.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for authentication (never log this).
    pub api_key: String,

    /// Base URL for API requests.
    /// Default: "https://api.anthropic.com"
    pub base_url: String,

    /// Anthropic API version header value.
    /// Default: "2023-06-01"
    pub anthropic_version: String,

    /// Request timeout in seconds.
    pub timeout_secs: u64,

    /// Maximum retry attempts for transient failures.
    pub max_retries: u32,

    /// Initial backoff delay in milliseconds for exponential retry.
    pub retry_backoff_ms: u64,}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            base_url: "https://api.anthropic.com".to_string(),
            anthropic_version: "2023-06-01".to_string(),
            timeout_secs: 120,
            max_retries: 3,
            retry_backoff_ms: 500,
        }
    }
}

// =============================================================================
// Anthropic Provider Implementation
// =============================================================================

/// Provider implementation for Anthropic's Claude API.
///
/// Supports:
/// - claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
/// - claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
///
/// Key differences from OpenAI:
/// - `max_tokens` is required in every request
/// - System prompt goes in top-level `system` field, not messages array
/// - Different SSE event structure for streaming
/// - Tool use format uses `tool_use`/`tool_result` content blocks
pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: Client,
    model_context_sizes: HashMap<String, usize>,
    model_pricing: HashMap<String, (f64, f64)>, // (input_per_1M, output_per_1M) in USD
}

impl AnthropicProvider {
    /// Creates a new Anthropic provider with the given configuration.
    pub fn new(config: AnthropicConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("failed to build HTTP client");

        // Real context window sizes (all Claude 3 models: 200K tokens)
        let mut model_context_sizes = HashMap::new();
        model_context_sizes.insert("claude-3-5-sonnet-20241022".to_string(), 200_000);
        model_context_sizes.insert("claude-3-5-haiku-20241022".to_string(), 200_000);
        model_context_sizes.insert("claude-3-opus-20240229".to_string(), 200_000);        model_context_sizes.insert("claude-3-sonnet-20240229".to_string(), 200_000);
        model_context_sizes.insert("claude-3-haiku-20240307".to_string(), 200_000);

        // Real pricing per 1M tokens in USD (Anthropic public pricing as of 2025)
        let mut model_pricing = HashMap::new();
        // claude-3-5-sonnet: $3.00 input / $15.00 output per 1M
        model_pricing.insert("claude-3-5-sonnet-20241022".to_string(), (3.0, 15.0));
        // claude-3-5-haiku: $0.80 input / $4.00 output per 1M
        model_pricing.insert("claude-3-5-haiku-20241022".to_string(), (0.80, 4.0));
        // claude-3-opus: $15.00 input / $75.00 output per 1M
        model_pricing.insert("claude-3-opus-20240229".to_string(), (15.0, 75.0));
        // claude-3-sonnet: $3.00 input / $15.00 output per 1M
        model_pricing.insert("claude-3-sonnet-20240229".to_string(), (3.0, 15.0));
        // claude-3-haiku: $0.25 input / $1.25 output per 1M
        model_pricing.insert("claude-3-haiku-20240307".to_string(), (0.25, 1.25));

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

    /// Extracts system prompt from messages and converts rest to Anthropic format.
    fn convert_messages(
        messages: &[Message],
    ) -> (Option<String>, Vec<AnthropicMessage>) {
        let mut system_prompt: Option<String> = None;
        let mut anthropic_messages = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    // Anthropic: system prompt goes in top-level field
                    if system_prompt.is_none() {
                        system_prompt = Some(msg.text_content());
                    } else {
                        // Append to existing system prompt
                        if let Some(ref mut sys) = system_prompt {
                            sys.push_str("\n\n");                            sys.push_str(&msg.text_content());
                        }
                    }
                }
                MessageRole::User | MessageRole::Assistant => {
                    let role = match msg.role {
                        MessageRole::User => "user".to_string(),
                        MessageRole::Assistant => "assistant".to_string(),
                        _ => unreachable!(),
                    };

                    let content = msg
                        .content
                        .iter()
                        .filter_map(convert_content_block)
                        .collect();

                    if !content.is_empty() {
                        anthropic_messages.push(AnthropicMessage { role, content });
                    }
                }
                MessageRole::Tool => {
                    // Tool results become user messages with tool_result content
                    let content = msg
                        .content
                        .iter()
                        .filter_map(|block| {
                            if let ContentBlock::ToolResult { tool_call_id, content, is_error } = block {
                                Some(AnthropicContent::ToolResult {
                                    tool_use_id: tool_call_id.clone(),
                                    content: content.clone(),
                                    is_error: *is_error,
                                })
                            } else {
                                None
                            }
                        })
                        .collect();

                    if !content.is_empty() {
                        anthropic_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content,
                        });
                    }
                }
            }
        }

        (system_prompt, anthropic_messages)    }

    /// Converts nexus_proto ToolSpec to Anthropic tool definition format.
    fn convert_tool(tool: &ToolSpec) -> AnthropicTool {
        AnthropicTool {
            name: tool.name.clone(),
            description: Some(tool.description.clone()),
            input_schema: tool.parameters.clone(),
        }
    }

    /// Builds the full API URL for message completions.
    fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.config.base_url.trim_end_matches('/'))
    }

    /// Parses Anthropic SSE event into streaming tokens.
    fn parse_stream_event(
        event_type: &str,
        data: &str,
    ) -> Option<Result<Token, RouterError>> {
        match event_type {
            "content_block_delta" => {
                if let Ok(delta) = serde_json::from_str::<AnthropicContentBlockDelta>(data) {
                    if delta.delta.delta_type == "text_delta" {
                        return Some(Ok(Token {
                            text: delta.delta.text.unwrap_or_default(),
                            is_final: false,
                            finish_reason: None,
                        }));
                    }
                }
                None
            }
            "message_delta" => {
                if let Ok(msg_delta) = serde_json::from_str::<AnthropicMessageDelta>(data) {
                    let finish = msg_delta.delta.stop_reason.as_deref().map(|r| match r {
                        "end_turn" => FinishReason::Stop,
                        "max_tokens" => FinishReason::MaxTokens,
                        "tool_use" => FinishReason::ToolCall,
                        "stop_sequence" => FinishReason::Stop,
                        _ => FinishReason::Error(r.to_string()),
                    });
                    return Some(Ok(Token {
                        text: String::new(),
                        is_final: true,
                        finish_reason: finish,
                    }));
                }
                None            }
            "error" => {
                if let Ok(err) = serde_json::from_str::<AnthropicStreamError>(data) {
                    return Some(Err(RouterError::ProviderError(format!(
                        "stream error: {} ({})",
                        err.error.message, err.error.r#type
                    ))));
                }
                None
            }
            _ => None, // Skip other event types (message_start, content_block_start, etc.)
        }
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    #[instrument(skip(self, request), fields(model = ?request.model))]
    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, RouterError> {
        let model = request
            .model
            .as_ref()
            .map(|m| m.model.clone())
            .unwrap_or_else(|| "claude-3-5-sonnet-20241022".to_string());

        let url = self.messages_url();
        let (system, messages) = Self::convert_messages(&request.messages);

        // Anthropic requires max_tokens; derive from request or use default
        let max_tokens = request.max_tokens.unwrap_or(4096);

        let anthropic_req = AnthropicRequest {
            model: model.clone(),
            max_tokens,
            messages,
            system,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: false,
            tools: if !request.tools.is_empty() {
                Some(request.tools.iter().map(Self::convert_tool).collect())
            } else {
                None
            },
            tool_choice: None,
        };

        let mut last_error: Option<RouterError> = None;

        for attempt in 0..=self.config.max_retries {            match self.execute_request(&url, &anthropic_req, false).await {
                Ok(response) => {
                    let anthropic_resp: AnthropicResponse = response
                        .json()
                        .await
                        .map_err(|e| RouterError::ProviderError(format!("response parse failed: {}", e)))?;

                    let latency_ms = 0; // Would extract from response headers in production

                    return Self::convert_response(&anthropic_resp, &model, latency_ms);
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
            .unwrap_or_else(|| "claude-3-5-sonnet-20241022".to_string());

        let url = self.messages_url();
        let (system, messages) = Self::convert_messages(&request.messages);
        let max_tokens = request.max_tokens.unwrap_or(4096);

        let anthropic_req = AnthropicRequest {
            model: model.clone(),
            max_tokens,
            messages,
            system,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: true,
            tools: if !request.tools.is_empty() {
                Some(request.tools.iter().map(Self::convert_tool).collect())            } else {
                None
            },
            tool_choice: None,
        };

        let response = self
            .execute_request(&url, &anthropic_req, true)
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(handle_api_error(status, &body));
        }

        // Parse Anthropic's SSE stream
        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(|event| async {
                match event {
                    Ok(ev) => {
                        // Anthropic uses event type field to distinguish message types
                        Self::parse_stream_event(&ev.event, &ev.data)
                    }
                    Err(e) => {
                        warn!(error = %e, "SSE stream error");
                        Some(Err(RouterError::ProviderError(format!("stream error: {}", e))))
                    }
                }
            });

        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> Result<(), RouterError> {
        // Health check: call /v1/models endpoint or simple message with minimal tokens
        let url = format!("{}/v1/messages", self.config.base_url.trim_end_matches('/'));

        let health_req = AnthropicRequest {
            model: "claude-3-haiku-20240307".to_string(),
            max_tokens: 1,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: vec![AnthropicContent::Text { text: ".".to_string() }],
            }],
            system: None,
            temperature: None,
            top_p: None,            stream: false,
            tools: None,
            tool_choice: None,
        };

        let response = self
            .execute_request(&url, &health_req, false)
            .await;

        match response {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn provider_id(&self) -> ProviderId {
        ProviderId::Anthropic
    }

    fn available_models(&self) -> Vec<String> {
        self.model_context_sizes.keys().cloned().collect()
    }

    fn max_context_tokens(&self, model: &str) -> usize {
        self.model_context_sizes
            .get(model)
            .copied()
            .unwrap_or(200_000) // Default to Claude 3 context size
    }

    fn estimate_cost(&self, model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
        let (input_rate, output_rate) = self
            .model_pricing
            .get(model)
            .copied()
            .unwrap_or((3.0, 15.0)); // Default to sonnet pricing

        // Rates are per 1M tokens
        (input_tokens as f64 * input_rate + output_tokens as f64 * output_rate) / 1_000_000.0
    }
}

// =============================================================================
// Internal Anthropic API Types
// =============================================================================

#[derive(Debug, Clone, Serialize)]
struct AnthropicRequest {
    model: String,
    /// Required by Anthropic API: maximum tokens to generate.    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String, // "user" or "assistant"
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    Text { text: String },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    #[serde(rename = "image")]
    Image {
        source: AnthropicImageSource,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicImageSource {
    r#type: String, // "base64"
    media_type: String, // "image/jpeg", etc.
    data: String,
}

#[derive(Debug, Clone, Serialize)]struct AnthropicTool {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<AnthropicContent>,
    model: String,
    #[serde(rename = "stop_reason")]
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// Streaming event types

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamEvent {
    MessageStart { message: AnthropicResponse },
    ContentBlockStart { index: usize, content_block: AnthropicContent },
    ContentBlockDelta(AnthropicContentBlockDelta),
    ContentBlockStop { index: usize },
    MessageDelta(AnthropicMessageDelta),
    MessageStop,
    Error(AnthropicStreamError),
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicContentBlockDelta {
    index: usize,
    delta: AnthropicDelta,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: String,    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicMessageDelta {
    delta: AnthropicMessageDeltaInner,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicMessageDeltaInner {
    #[serde(rename = "stop_reason")]
    stop_reason: Option<String>,
    #[serde(rename = "stop_sequence")]
    stop_sequence: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicStreamError {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    r#type: String,
    message: String,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Converts nexus_proto ContentBlock to Anthropic content format.
fn convert_content_block(block: &ContentBlock) -> Option<AnthropicContent> {
    match block {
        ContentBlock::Text(s) => Some(AnthropicContent::Text { text: s.clone() }),
        ContentBlock::ToolCall { id, name, arguments } => Some(AnthropicContent::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: arguments.clone(),
        }),
        ContentBlock::ToolResult { tool_call_id, content, is_error } => {
            Some(AnthropicContent::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: content.clone(),
                is_error: if *is_error { Some(true) } else { None },            })
        }
    }
}

/// Converts Anthropic response to nexus_proto ModelResponse.
fn convert_response(
    anthropic_resp: &AnthropicResponse,
    model: &str,
    latency_ms: u64,
) -> Result<ModelResponse, RouterError> {
    let content: Vec<ContentBlock> = anthropic_resp
        .content
        .iter()
        .filter_map(convert_anthropic_content)
        .collect();

    let message = Message {
        role: match anthropic_resp.role.as_str() {
            "assistant" => MessageRole::Assistant,
            "user" => MessageRole::User,
            _ => MessageRole::Assistant,
        },
        content,
    };

    let usage = ModelUsage {
        prompt_tokens: anthropic_resp.usage.input_tokens,
        completion_tokens: anthropic_resp.usage.output_tokens,
        total_tokens: anthropic_resp.usage.input_tokens + anthropic_resp.usage.output_tokens,
        estimated_cost_usd: 0.0, // Filled by router cost calculator
    };

    Ok(ModelResponse {
        id: uuid::Uuid::parse_str(&anthropic_resp.id).unwrap_or_else(|_| uuid::Uuid::new_v4()),
        model: ModelId::new(crate::providers::ProviderId::Anthropic, model),
        message,
        usage,
        latency_ms,
    })
}

/// Converts Anthropic content to nexus_proto ContentBlock.
fn convert_anthropic_content(content: &AnthropicContent) -> Option<ContentBlock> {
    match content {
        AnthropicContent::Text { text } => Some(ContentBlock::Text(text.clone())),
        AnthropicContent::ToolUse { id, name, input } => Some(ContentBlock::ToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: input.clone(),        }),
        AnthropicContent::ToolResult { tool_use_id, content, is_error } => {
            Some(ContentBlock::ToolResult {
                tool_call_id: tool_use_id.clone(),
                content: content.clone(),
                is_error: is_error.unwrap_or(false),
            })
        }
        AnthropicContent::Image { .. } => None, // Images not yet supported in nexus_proto
    }
}

impl AnthropicProvider {
    /// Executes an HTTP request to the Anthropic API with proper headers.
    async fn execute_request(
        &self,
        url: &str,
        body: &AnthropicRequest,
        stream: bool,
    ) -> Result<Response, RouterError> {
        let mut req = self
            .client
            .post(url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.anthropic_version)
            .header("Content-Type", "application/json");

        if stream {
            req = req.header("Accept", "text/event-stream");
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

/// Determines if an error is retryable.
fn should_retry(err: &RouterError) -> bool {
    matches!(        err,
        RouterError::ProviderUnavailable(_)
            | RouterError::Timeout { .. }
            | RouterError::ProviderError(msg) if msg.contains("5") || msg.contains("429")
    )
}

/// Maps HTTP status and body to RouterError.
fn handle_api_error(status: StatusCode, body: &str) -> RouterError {
    let message = parse_anthropic_error(body).unwrap_or_else(|| body.to_string());

    match status {
        StatusCode::UNAUTHORIZED => RouterError::ProviderError(format!("authentication failed: {}", message)),
        StatusCode::FORBIDDEN => RouterError::ProviderError(format!("access denied: {}", message)),
        StatusCode::NOT_FOUND => RouterError::ProviderError(format!("endpoint not found: {}", message)),
        StatusCode::TOO_MANY_REQUESTS => {
            RouterError::ProviderError(format!("rate limit exceeded: {}", message))
        }
        StatusCode::BAD_REQUEST => RouterError::ProviderError(format!("invalid request: {}", message)),
        StatusCode::INTERNAL_SERVER_ERROR
        | StatusCode::BAD_GATEWAY
        | StatusCode::SERVICE_UNAVAILABLE => {
            RouterError::ProviderUnavailable(format!("provider error {}: {}", status, message))
        }
        _ => RouterError::ProviderError(format!("API error {}: {}", status, message)),
    }
}

/// Parses Anthropic error response body.
fn parse_anthropic_error(body: &str) -> Option<String> {
    #[derive(Deserialize)]
    struct AnthropicError {
        error: AnthropicErrorDetail,
    }

    serde_json::from_str::<AnthropicError>(body)
        .ok()
        .map(|e| format!("{} ({})", e.error.message, e.error.r#type))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_proto::model::{AgentId, RoutingPolicy};

    #[test]    fn test_config_defaults() {
        let config = AnthropicConfig::default();
        assert_eq!(config.base_url, "https://api.anthropic.com");
        assert_eq!(config.anthropic_version, "2023-06-01");
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_model_context_sizes() {
        let provider = AnthropicProvider::new(AnthropicConfig::default());
        assert_eq!(provider.max_context_tokens("claude-3-5-sonnet-20241022"), 200_000);
        assert_eq!(provider.max_context_tokens("claude-3-opus-20240229"), 200_000);
        assert_eq!(provider.max_context_tokens("unknown-model"), 200_000); // Default fallback
    }

    #[test]
    fn test_model_pricing() {
        let provider = AnthropicProvider::new(AnthropicConfig::default());
        
        // claude-3-5-sonnet: $3 input / $15 output per 1M tokens
        let cost = provider.estimate_cost("claude-3-5-sonnet-20241022", 100_000, 50_000);
        assert!((cost - 1.05).abs() < 0.001); // (100k*3 + 50k*15)/1M = 1.05

        // claude-3-haiku: $0.25 input / $1.25 output per 1M
        let cost = provider.estimate_cost("claude-3-haiku-20240307", 100_000, 50_000);
        assert!((cost - 0.0875).abs() < 0.0001); // (100k*0.25 + 50k*1.25)/1M = 0.0875
    }

    #[test]
    fn test_convert_messages_with_system() {
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: vec![ContentBlock::Text("You are a helpful assistant.".to_string())],
            },
            Message {
                role: MessageRole::User,
                content: vec![ContentBlock::Text("Hello".to_string())],
            },
        ];

        let (system, anthropic_msgs) = AnthropicProvider::convert_messages(&messages);
        
        assert_eq!(system, Some("You are a helpful assistant.".to_string()));
        assert_eq!(anthropic_msgs.len(), 1);
        assert_eq!(anthropic_msgs[0].role, "user");
    }

    #[test]    fn test_convert_content_block() {
        let text_block = ContentBlock::Text("Hello".to_string());
        let converted = convert_content_block(&text_block);
        assert!(matches!(converted, Some(AnthropicContent::Text { text }) if text == "Hello"));

        let tool_block = ContentBlock::ToolCall {
            id: "call_123".to_string(),
            name: "search".to_string(),
            arguments: json!({"query": "rust"}),
        };
        let converted = convert_content_block(&tool_block);
        assert!(matches!(converted, Some(AnthropicContent::ToolUse { name, .. }) if name == "search"));
    }

    #[test]
    fn test_backoff_duration() {
        let provider = AnthropicProvider::new(AnthropicConfig {
            retry_backoff_ms: 100,
            ..Default::default()
        });

        let d0 = provider.backoff_duration(0);
        let d1 = provider.backoff_duration(1);
        let d2 = provider.backoff_duration(2);

        assert!(d0.as_millis() >= 75 && d0.as_millis() <= 125);
        assert!(d1.as_millis() >= 150 && d1.as_millis() <= 250);
        assert!(d2.as_millis() >= 300 && d2.as_millis() <= 500);
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
    fn test_parse_anthropic_error() {
        let body = r#"{"error":{"type":"invalid_request_error","message":"Invalid model specified"}}"#;
        let parsed = parse_anthropic_error(body);
        assert!(parsed.unwrap().contains("Invalid model specified"));

        let bad_body = "not json";
        assert!(parse_anthropic_error(bad_body).is_none());
    }
    #[test]
    fn test_stream_event_parsing() {
        // Test content_block_delta with text
        let delta_data = r#"{"index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let result = AnthropicProvider::parse_stream_event("content_block_delta", delta_data);
        assert!(result.is_some());
        if let Some(Ok(token)) = result {
            assert_eq!(token.text, "Hello");
            assert!(!token.is_final);
        }

        // Test message_delta with stop_reason
        let msg_delta = r#"{"delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":5}}"#;
        let result = AnthropicProvider::parse_stream_event("message_delta", msg_delta);
        assert!(result.is_some());
        if let Some(Ok(token)) = result {
            assert!(token.is_final);
            assert!(matches!(token.finish_reason, Some(FinishReason::Stop)));
        }

        // Test error event
        let err_data = r#"{"error":{"type":"overloaded_error","message":"Service overloaded"}}"#;
        let result = AnthropicProvider::parse_stream_event("error", err_data);
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }
}
