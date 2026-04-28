use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

use crate::error::NexusError;

// =============================================================================
// Provider & Model Identification
// =============================================================================

/// Identifies the upstream model provider.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum ProviderId {
    OpenAI,
    Anthropic,
    Groq,
    Mistral,
    Local,
    Custom(String),
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderId::OpenAI => write!(f, "openai"),
            ProviderId::Anthropic => write!(f, "anthropic"),
            ProviderId::Groq => write!(f, "groq"),
            ProviderId::Mistral => write!(f, "mistral"),
            ProviderId::Local => write!(f, "local"),
            ProviderId::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Unique identifier for a specific model on a specific provider.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId {
    pub provider: ProviderId,
    pub model: String,
}

impl ModelId {
    /// Constructs a `ModelId` from a provider and model name.
    pub fn new(provider: ProviderId, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),        }
    }

    /// Convenience constructor for OpenAI models.
    pub fn openai(model: &str) -> Self {
        Self::new(ProviderId::OpenAI, model)
    }

    /// Convenience constructor for Anthropic models.
    pub fn anthropic(model: &str) -> Self {
        Self::new(ProviderId::Anthropic, model)
    }

    /// Convenience constructor for Groq models.
    pub fn groq(model: &str) -> Self {
        Self::new(ProviderId::Groq, model)
    }

    /// Convenience constructor for local/Ollama models.
    pub fn local(model: &str) -> Self {
        Self::new(ProviderId::Local, model)
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.provider, self.model)
    }
}

impl FromStr for ModelId {
    type Err = NexusError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.splitn(2, '/');
        let provider_str = parts.next().ok_or_else(|| {
            NexusError::RouterError("invalid model id format: missing provider".into())
        })?;
        let model_str = parts.next().ok_or_else(|| {
            NexusError::RouterError("invalid model id format: missing model name".into())
        })?;

        let provider = match provider_str {
            "openai" => ProviderId::OpenAI,
            "anthropic" => ProviderId::Anthropic,
            "groq" => ProviderId::Groq,
            "mistral" => ProviderId::Mistral,
            "local" => ProviderId::Local,
            _ => ProviderId::Custom(provider_str.to_string()),
        };
        Ok(Self::new(provider, model_str))
    }
}

// =============================================================================
// Message & Content Types
// =============================================================================

/// Role of a message in a conversation turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A discrete unit of content within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content.
    Text(String),
    /// A function/tool call requested by the model.
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// The result of executing a tool call.
    ToolResult {
        tool_call_id: String,
        content: String,
        is_error: bool,
    },
}

/// A single message in a conversation context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Creates a user message containing plain text.
    pub fn user(text: impl Into<String>) -> Self {
        Self {            role: MessageRole::User,
            content: vec![ContentBlock::Text(text.into())],
        }
    }

    /// Creates a system message containing plain text.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![ContentBlock::Text(text.into())],
        }
    }

    /// Creates an assistant message containing plain text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text(text.into())],
        }
    }

    /// Concatenates all `Text` content blocks in this message into a single string.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

// =============================================================================
// Tool & Routing Definitions
// =============================================================================

/// Specification of a tool/function available to the model during inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl ToolSpec {
    /// Constructs a tool specification from its JSON Schema definition.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Strategy for selecting a model/provider for a given request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum RoutingPolicy {
    /// Prefer cheapest models that can respond within the latency budget.
    CostOptimized { max_latency_ms: u64 },
    /// Prefer fastest models that stay under the cost ceiling per 1K tokens.
    LatencyOptimized { max_cost_per_1k_tokens: f64 },
    /// Route based on strict capability requirements (context size, vision, etc.).
    CapabilityFirst {
        required_context_tokens: Option<u32>,
        requires_vision: bool,
    },
    /// Attempt local inference first; fallback to cloud on failure or OOM.
    LocalFirst { cloud_fallback: bool },
    /// Force routing to a specific model/provider.
    Pinned(ModelId),
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        RoutingPolicy::CostOptimized { max_latency_ms: 5000 }
    }
}

// =============================================================================
// Streaming & Response Types
// =============================================================================

/// Reason why model generation terminated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    MaxTokens,
    ToolCall,
    ContentFilter,
    Error(String),
}

/// A single token yielded from a streaming model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {    pub text: String,
    pub is_final: bool,
    pub finish_reason: Option<FinishReason>,
}

/// Token usage accounting for a completed generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub estimated_cost_usd: f64,
}

/// A complete, non-streaming response from a model provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub id: Uuid,
    pub model: ModelId,
    pub message: Message,
    pub usage: ModelUsage,
    pub latency_ms: u64,
}

// =============================================================================
// Request Types & Builder
// =============================================================================

/// A complete request payload to be routed and executed against a model provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    pub id: Uuid,
    pub model: Option<ModelId>,
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub tools: Vec<ToolSpec>,
    pub routing_policy: RoutingPolicy,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Builder for constructing `ModelRequest` instances with validation.
pub struct ModelRequestBuilder {
    model: Option<ModelId>,
    messages: Vec<Message>,
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,    top_p: Option<f32>,
    tools: Vec<ToolSpec>,
    routing_policy: RoutingPolicy,
    metadata: HashMap<String, serde_json::Value>,
}

impl ModelRequest {
    /// Creates a new builder for `ModelRequest`.
    pub fn builder() -> ModelRequestBuilder {
        ModelRequestBuilder::default()
    }
}

impl Default for ModelRequestBuilder {
    fn default() -> Self {
        Self {
            model: None,
            messages: Vec::new(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: Vec::new(),
            routing_policy: RoutingPolicy::default(),
            metadata: HashMap::new(),
        }
    }
}

impl ModelRequestBuilder {
    /// Sets the target model (optional; routing policy selects one if None).
    pub fn model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the conversation message history.
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Sets an override system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max: u32) -> Self {        self.max_tokens = Some(max);
        self
    }

    /// Sets the sampling temperature (0.0 to 2.0).
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets nucleus sampling threshold (0.0 to 1.0).
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Sets the list of available tools for function calling.
    pub fn tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools = tools;
        self
    }

    /// Sets the routing strategy for this request.
    pub fn routing_policy(mut self, policy: RoutingPolicy) -> Self {
        self.routing_policy = policy;
        self
    }

    /// Sets arbitrary metadata for tracing/cost ledger correlation.
    pub fn metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Validates parameters and constructs the final `ModelRequest`.
    pub fn build(self) -> Result<ModelRequest, NexusError> {
        if self.messages.is_empty() {
            return Err(NexusError::RouterError(
                "model request must contain at least one message".into(),
            ));
        }

        if let Some(temp) = self.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(NexusError::RouterError(
                    "temperature must be between 0.0 and 2.0".into(),
                ));
            }
        }
        if let Some(tp) = self.top_p {
            if !(0.0..=1.0).contains(&tp) {
                return Err(NexusError::RouterError(
                    "top_p must be between 0.0 and 1.0".into(),
                ));
            }
        }

        Ok(ModelRequest {
            id: Uuid::new_v4(),
            model: self.model,
            messages: self.messages,
            system_prompt: self.system_prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            tools: self.tools,
            routing_policy: self.routing_policy,
            metadata: self.metadata,
        })
    }
}
