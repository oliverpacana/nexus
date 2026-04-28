// examples/multi-agent-pipeline/src/main.rs
//! Multi-Agent Content Production Pipeline Example
//!
//! This example demonstrates a supervised pipeline of specialized AI agents
//! working together to produce a complete content package:
//! 1. ResearchAgent: Gathers facts and sources
//! 2. OutlineAgent: Structures the content
//! 3. WriterAgent: Writes the full draft
//! 4. EditorAgent: Reviews, improves, and fact-checks
//! 5. SEOAgent: Adds metadata, keywords, and summary
//!
//! Agents communicate via shared L3 semantic memory and run under a
//! supervisor tree with fault tolerance.
//!
//! Usage:
//!   cargo run --example multi-agent-pipeline -- --topic "quantum computing"

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use chrono::Utc;
use clap::Parser;
use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use nexus_kernel::{
    capabilities::{CapabilityGuard, CapabilitySet},
    supervisor::RestartStrategy,
    AgentContext, AgentPriority, AgentTask, Kernel, KernelConfig, KernelHandle, SpawnOptions,
};
use nexus_mem::{
    embeddings::LocalEmbeddingProvider, MemoryConfig, MemoryEntry, MemoryKey, MemoryScope,
    MemoryStore, MemoryTier, SemanticSearchQuery,
};
use nexus_proto::agent::{AgentCapabilities, AgentId, AgentKind};
use nexus_proto::memory::{EpisodicEvent, EpisodicEventType};
use nexus_proto::model::{Message, MessageRole, ModelId, ModelRequest, ProviderId, RoutingPolicy};
use nexus_proto::tool::ToolCall;
use nexus_router::{
    providers::{anthropic::AnthropicConfig, openai::OpenAIConfig, ProviderRegistry},
    ModelRouter, RouterConfig,
};
use nexus_tools::ToolEngineConfig;
use serde_json::{json, Value};
use tokio::sync::watch;use tokio::time::{interval, sleep};
use tracing::{debug, info, instrument};
use uuid::Uuid;

// =============================================================================
// Constants and Memory Schema
// =============================================================================

const KEY_RESEARCH: &str = "pipeline::research::results";
const KEY_OUTLINE: &str = "pipeline::outline::structure";
const KEY_DRAFT: &str = "pipeline::draft::content";
const KEY_EDITED: &str = "pipeline::edited::content";
const KEY_FINAL: &str = "pipeline::final::package";
const KEY_META: &str = "pipeline::meta::run_info";

const DEFAULT_MODEL: &str = "anthropic/claude-3-5-sonnet-20241022";
const MAX_CONTENT_CHARS: usize = 4000;
const MAX_REPORT_TOKENS: u32 = 4000;

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The content topic to research and write about
    #[arg(short, long)]
    topic: String,

    /// Target audience for the content
    #[arg(short, long, default_value = "general", value_parser = ["technical", "general", "executive", "beginner"])]
    audience: String,

    /// Type of content to produce
    #[arg(long, default_value = "article", value_parser = ["article", "blog-post", "report", "tutorial"])]
    content_type: String,

    /// Target word count for the final content
    #[arg(long, default_value = "1500")]
    word_count: usize,

    /// Model to use for generation (format: provider/model)
    #[arg(short, long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Run research and outline agents in parallel
    #[arg(long)]
    parallel: bool,
    /// Stream each agent's output as it runs
    #[arg(long)]
    live_output: bool,

    /// Directory to save all pipeline outputs
    #[arg(long)]
    save_dir: Option<PathBuf>,
}

// =============================================================================
// Pipeline Configuration and Result
// =============================================================================

#[derive(Debug, Clone)]
struct PipelineConfig {
    topic: String,
    audience: String,
    content_type: String,
    word_count: usize,
    model: String,
    parallel: bool,
    live_output: bool,
    save_dir: Option<PathBuf>,
}

#[derive(Debug)]
struct PipelineResult {
    topic: String,
    content_type: String,
    audience: String,
    final_package: Value,
    agent_results: HashMap<String, Value>,
    total_duration: Duration,
    cost_summary: CostSummary,
}

#[derive(Debug, Default)]
struct CostSummary {
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_cost_usd: f64,
    by_agent: HashMap<String, AgentCost>,
}

#[derive(Debug)]
struct AgentCost {
    provider: String,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,}

// =============================================================================
// Agent Implementations
// =============================================================================

// Helper function to parse model spec
fn parse_model_spec(spec: &str) -> (ProviderId, String) {
    if let Some((provider, model)) = spec.split_once('/') {
        let provider_id = match provider.to_lowercase().as_str() {
            "openai" => ProviderId::OpenAI,
            "anthropic" => ProviderId::Anthropic,
            "groq" => ProviderId::Groq,
            "local" => ProviderId::Local,
            _ => ProviderId::Custom(provider.to_string()),
        };
        (provider_id, model.to_string())
    } else {
        (ProviderId::OpenAI, spec.to_string())
    }
}

// Helper function to wait for a memory key to appear
async fn wait_for_memory_key(
    memory: &MemoryStore,
    agent_id: AgentId,
    key: &MemoryKey,
    timeout: Duration,
) -> Result<Value> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Ok(Some(entry)) = memory.semantic().get(key, agent_id).await {
            return Ok(entry.value);
        }
        sleep(Duration::from_millis(500)).await;
    }
    bail!("Timeout waiting for memory key: {}", key)
}

// =============================================================================
// ResearchAgent (Simplified from research-agent example)
// =============================================================================

struct ResearchAgent {
    topic: String,
    audience: String,
    model_id: ModelId,
}

impl ResearchAgent {    fn new(topic: String, audience: String, model: String) -> Self {
        let (provider_id, model_name) = parse_model_spec(&model);
        Self {
            topic,
            audience,
            model_id: ModelId::new(provider_id, &model_name),
        }
    }

    async fn generate_queries(
        &self,
        router: &ModelRouter,
        agent_id: AgentId,
    ) -> Result<Vec<String>> {
        let prompt = format!(
            "You are a research assistant. Generate 3 distinct search queries to \
             thoroughly research the topic: '{}' for a {} audience.\n\
             Return ONLY a JSON array of strings, nothing else.\n\
             Make queries specific, varied, and covering different aspects.",
            self.topic, self.audience
        );

        let request = ModelRequest::builder()
            .messages(vec![Message::user(prompt)])
            .model(Some(self.model_id.clone()))
            .routing_policy(RoutingPolicy::default())
            .build()?;

        let response = router.complete(request, agent_id).await?;
        let response_text = response.message.text_content();
        
        // Parse JSON array with fallback
        serde_json::from_str(&response_text).or_else(|_| {
            if let Some(start) = response_text.find('[') {
                if let Some(end) = response_text[start..].find(']').map(|i| start + i + 1) {
                    serde_json::from_str(&response_text[start..end])
                } else {
                    Err(serde_json::Error::custom("No JSON array found"))
                }
            } else {
                Err(serde_json::Error::custom("No JSON array found"))
            }
        })
    }

    async fn synthesize_brief(
        &self,
        router: &ModelRouter,
        agent_id: AgentId,
        findings: Vec<(String, String, Option<String>)>, // (query, content, title)    ) -> Result<String> {
        let mut findings_text = String::new();
        for (i, (query, content, title)) in findings.iter().enumerate() {
            findings_text.push_str(&format!(
                "{}. Query: {}\n   Title: {}\n   Content: {}\n\n",
                i + 1,
                query,
                title.as_deref().unwrap_or("N/A"),
                &content[..content.len().min(500)]
            ));
        }

        let prompt = format!(
            "You are a research synthesizer. Based on these search findings about '{}', \
             create a concise research brief for a {} audience.\n\n\
             FINDINGS:\n{}\n\n\
             Write a brief that includes:\n\
             - Key facts and statistics\n\
             - Expert opinions or quotes\n\
             - Important sources (URLs)\n\
             - Controversies or differing viewpoints\n\
             - Current state of knowledge\n\n\
             Keep it under 500 words. Be specific and cite findings where relevant.",
            self.topic, self.audience, findings_text
        );

        let request = ModelRequest::builder()
            .messages(vec![Message::user(prompt)])
            .model(Some(self.model_id.clone()))
            .max_tokens(1000)
            .routing_policy(RoutingPolicy::default())
            .build()?;

        let response = router.complete(request, agent_id).await?;
        Ok(response.message.text_content())
    }
}

impl AgentTask for ResearchAgent {
    fn name(&self) -> &str { "research-agent" }
    fn kind(&self) -> AgentKind { AgentKind::Research }
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::new()
            .with_tool("web-search")
            .with_tool("http-fetch")
            .with_memory(nexus_proto::memory::MemoryAccess::ReadWrite)
            .with_model("anthropic/*")
            .with_model("openai/*")
    }
    #[instrument(skip(self, ctx), fields(topic = %self.topic))]
    async fn run(&mut self, ctx: AgentContext) -> Result<Value> {
        let agent_id = ctx.agent_id();
        let memory = ctx.memory();
        let tools = ctx.tools();
        let router = ctx.router();

        // Generate search queries
        let queries = self.generate_queries(router, agent_id).await?;
        info!("Generated {} research queries", queries.len());

        // Perform searches and fetch content
        let mut findings = Vec::new();
        for query in &queries {
            let tool_call = ToolCall {
                id: Uuid::new_v4(),
                tool_name: "web-search".to_string(),
                arguments: json!({ "query": query, "max_results": 3 }),
                agent_id,
                trace_id: Uuid::new_v4(),
            };

            if let Ok(result) = tools.call(tool_call).await {
                if !result.is_error {
                    if let Some(results) = result.output.get("results").and_then(|v| v.as_array()) {
                        for result in results {
                            if let Some(url) = result.get("url").and_then(|v| v.as_str()) {
                                let fetch_call = ToolCall {
                                    id: Uuid::new_v4(),
                                    tool_name: "http-fetch".to_string(),
                                    arguments: json!({
                                        "url": url,
                                        "extract_text": true,
                                        "max_chars": MAX_CONTENT_CHARS
                                    }),
                                    agent_id,
                                    trace_id: Uuid::new_v4(),
                                };

                                if let Ok(fetch_result) = tools.call(fetch_call).await {
                                    if !fetch_result.is_error {
                                        let content = fetch_result.output.get("content")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let title = fetch_result.output.get("title")
                                            .and_then(|v| v.as_str())
                                            .map(String::from);
                                        findings.push((query.clone(), content, title));
                                    }                                }
                            }
                        }
                    }
                }
            }
        }

        // Synthesize research brief
        let brief = self.synthesize_brief(router, agent_id, findings).await?;

        // Store in semantic memory
        let entry = MemoryEntry {
            key: MemoryKey::new("pipeline", "research::results"),
            tier: MemoryTier::Semantic,
            scope: MemoryScope::Global,
            owner_id: agent_id,
            value: json!({
                "topic": self.topic,
                "audience": self.audience,
                "brief": brief,
                "queries": queries,
                "source_count": findings.len()
            }),
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
            version: 1,
            tags: vec!["research".to_string(), self.topic.clone()],
        };

        memory.semantic_write(agent_id, entry).await?;

        Ok(json!({
            "brief": brief,
            "source_count": findings.len(),
            "word_count": brief.split_whitespace().count()
        }))
    }
}

// =============================================================================
// OutlineAgent
// =============================================================================

struct OutlineAgent {
    topic: String,
    audience: String,
    content_type: String,    word_count: usize,
    model_id: ModelId,
}

impl OutlineAgent {
    fn new(topic: String, audience: String, content_type: String, word_count: usize, model: String) -> Self {
        let (provider_id, model_name) = parse_model_spec(&model);
        Self {
            topic,
            audience,
            content_type,
            word_count,
            model_id: ModelId::new(provider_id, &model_name),
        }
    }
}

impl AgentTask for OutlineAgent {
    fn name(&self) -> &str { "outline-agent" }
    fn kind(&self) -> AgentKind { AgentKind::Planning }
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::new()
            .with_memory(nexus_proto::memory::MemoryAccess::ReadWrite)
            .with_model("anthropic/*")
            .with_model("openai/*")
    }

    #[instrument(skip(self, ctx))]
    async fn run(&mut self, ctx: AgentContext) -> Result<Value> {
        let agent_id = ctx.agent_id();
        let memory = ctx.memory();
        let router = ctx.router();

        // Wait for research results
        let research_key = MemoryKey::new("pipeline", "research::results");
        let research_data = wait_for_memory_key(memory, agent_id, &research_key, Duration::from_secs(300)).await?;
        
        let brief = research_data.get("brief").and_then(|v| v.as_str()).unwrap_or("");
        let audience_desc = match self.audience.as_str() {
            "technical" => "technical experts with domain knowledge",
            "executive" => "busy executives who need key insights quickly",
            "beginner" => "newcomers who need clear explanations",
            _ => "general readers with moderate background knowledge",
        };

        let prompt = format!(
            "You are a content strategist. Given this research brief and these parameters:\n\
             Topic: {}\n\
             Audience: {} ({})\n\
             Content type: {}\n\             Target word count: {}\n\n\
             Research Brief:\n{}\n\n\
             Create a detailed content outline with:\n\
             - A compelling title (3 options, ranked)\n\
             - Introduction approach\n\
             - 4-6 main sections with:\n\
               * Section heading\n\
               * Key points to cover (3-5 per section)\n\
               * Estimated word count per section\n\
               * Which research findings to incorporate\n\
             - Conclusion approach\n\
             - Call to action (if appropriate)\n\n\
             Return as structured JSON matching this schema:\n\
             {{\n\
               \"title_options\": [{{\"title\": str, \"rationale\": str}}],\n\
               \"chosen_title\": str,\n\
               \"intro\": {{\"approach\": str, \"hook\": str, \"estimated_words\": int}},\n\
               \"sections\": [{{\"heading\": str, \"points\": [str], \"estimated_words\": int, \"research_refs\": [str]}}],\n\
               \"conclusion\": {{\"approach\": str, \"estimated_words\": int}},\n\
               \"cta\": str | null,\n\
               \"total_estimated_words\": int\n\
             }}",
            self.topic, self.audience, audience_desc, self.content_type, self.word_count, brief
        );

        let request = ModelRequest::builder()
            .messages(vec![Message::user(prompt)])
            .model(Some(self.model_id.clone()))
            .max_tokens(2000)
            .routing_policy(RoutingPolicy::default())
            .build()?;

        let response = router.complete(request, agent_id).await?;
        let response_text = response.message.text_content();
        
        // Parse JSON with fallback
        let outline: Value = serde_json::from_str(&response_text).or_else(|_| {
            // Try to extract JSON from text
            if let Some(start) = response_text.find('{') {
                if let Some(end) = response_text[start..].rfind('}').map(|i| start + i + 1) {
                    serde_json::from_str(&response_text[start..end])
                } else {
                    Err(serde_json::Error::custom("No JSON object found"))
                }
            } else {
                Err(serde_json::Error::custom("No JSON object found"))
            }
        })?;

        // Store outline in semantic memory        let entry = MemoryEntry {
            key: MemoryKey::new("pipeline", "outline::structure"),
            tier: MemoryTier::Semantic,
            scope: MemoryScope::Global,
            owner_id: agent_id,
            value: outline.clone(),
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
            version: 1,
            tags: vec!["outline".to_string(), self.topic.clone()],
        };

        memory.semantic_write(agent_id, entry).await?;

        Ok(outline)
    }
}

// =============================================================================
// WriterAgent
// =============================================================================

struct WriterAgent {
    topic: String,
    audience: String,
    content_type: String,
    word_count: usize,
    model_id: ModelId,
}

impl WriterAgent {
    fn new(topic: String, audience: String, content_type: String, word_count: usize, model: String) -> Self {
        let (provider_id, model_name) = parse_model_spec(&model);
        Self {
            topic,
            audience,
            content_type,
            word_count,
            model_id: ModelId::new(provider_id, &model_name),
        }
    }
}

impl AgentTask for WriterAgent {
    fn name(&self) -> &str { "writer-agent" }
    fn kind(&self) -> AgentKind { AgentKind::Writing }
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::new()            .with_memory(nexus_proto::memory::MemoryAccess::ReadWrite)
            .with_model("anthropic/*")
            .with_model("openai/*")
    }

    #[instrument(skip(self, ctx))]
    async fn run(&mut self, ctx: AgentContext) -> Result<Value> {
        let agent_id = ctx.agent_id();
        let memory = ctx.memory();
        let router = ctx.router();

        // Wait for research and outline
        let research_key = MemoryKey::new("pipeline", "research::results");
        let outline_key = MemoryKey::new("pipeline", "outline::structure");
        
        let research_data = wait_for_memory_key(memory, agent_id, &research_key, Duration::from_secs(300)).await?;
        let outline_data = wait_for_memory_key(memory, agent_id, &outline_key, Duration::from_secs(300)).await?;
        
        let brief = research_data.get("brief").and_then(|v| v.as_str()).unwrap_or("");
        let sections = outline_data.get("sections").and_then(|v| v.as_array()).unwrap_or(&[]);
        let intro = outline_data.get("intro").and_then(|v| v.as_object()).unwrap_or(&serde_json::Map::new());
        let conclusion = outline_data.get("conclusion").and_then(|v| v.as_object()).unwrap_or(&serde_json::Map::new());

        let mut full_draft = String::new();
        let mut previous_sections = String::new();

        // Write introduction
        if let Some(approach) = intro.get("approach").and_then(|v| v.as_str()) {
            let intro_prompt = format!(
                "Write the introduction for a {} about '{}'.\n\
                 Target audience: {}\n\
                 Approach: {}\n\
                 Hook: {}\n\
                 Target length: {} words\n\
                 Research brief: {}\n\n\
                 Write ONLY the introduction. Be engaging and set up the article.",
                self.content_type, self.topic, self.audience, 
                approach,
                intro.get("hook").and_then(|v| v.as_str()).unwrap_or(""),
                intro.get("estimated_words").and_then(|v| v.as_i64()).unwrap_or(200),
                &brief[..brief.len().min(300)]
            );

            let request = ModelRequest::builder()
                .messages(vec![Message::user(intro_prompt)])
                .model(Some(self.model_id.clone()))
                .max_tokens(500)
                .routing_policy(RoutingPolicy::default())
                .build()?;
            let response = router.complete(request, agent_id).await?;
            full_draft.push_str(&response.message.text_content());
            full_draft.push_str("\n\n");
            previous_sections = response.message.text_content();
        }

        // Write each section
        for section in sections {
            if let Some(heading) = section.get("heading").and_then(|v| v.as_str()) {
                let points = section.get("points").and_then(|v| v.as_array()).unwrap_or(&[]);
                let points_text: Vec<String> = points.iter()
                    .filter_map(|p| p.as_str().map(String::from))
                    .collect();
                
                let section_prompt = format!(
                    "Write the '{}' section of a {} about '{}'.\n\
                     Target audience: {}\n\
                     Key points to cover: {}\n\
                     Target length: {} words\n\
                     Previous sections: {}\n\
                     Research brief: {}\n\n\
                     Write ONLY this section. Be engaging, specific, and substantive.",
                    heading, self.content_type, self.topic, self.audience,
                    points_text.join(", "),
                    section.get("estimated_words").and_then(|v| v.as_i64()).unwrap_or(300),
                    &previous_sections[..previous_sections.len().min(200)],
                    &brief[..brief.len().min(300)]
                );

                let request = ModelRequest::builder()
                    .messages(vec![Message::user(section_prompt)])
                    .model(Some(self.model_id.clone()))
                    .max_tokens(800)
                    .routing_policy(RoutingPolicy::default())
                    .build()?;

                let response = router.complete(request, agent_id).await?;
                let section_content = response.message.text_content();
                full_draft.push_str(&format!("## {}\n\n{}\n\n", heading, section_content));
                previous_sections.push_str(&section_content);
            }
        }

        // Write conclusion
        if let Some(approach) = conclusion.get("approach").and_then(|v| v.as_str()) {
            let conclusion_prompt = format!(
                "Write the conclusion for a {} about '{}'.\n\
                 Target audience: {}\n\
                 Approach: {}\n\
                 Target length: {} words\n\                 Full draft so far: {}\n\n\
                 Write ONLY the conclusion. Summarize key points and provide closure.",
                self.content_type, self.topic, self.audience,
                approach,
                conclusion.get("estimated_words").and_then(|v| v.as_i64()).unwrap_or(200),
                &full_draft[..full_draft.len().min(500)]
            );

            let request = ModelRequest::builder()
                .messages(vec![Message::user(conclusion_prompt)])
                .model(Some(self.model_id.clone()))
                .max_tokens(500)
                .routing_policy(RoutingPolicy::default())
                .build()?;

            let response = router.complete(request, agent_id).await?;
            full_draft.push_str(&format!("\n\n## Conclusion\n\n{}", response.message.text_content()));
        }

        // Store draft in semantic memory
        let entry = MemoryEntry {
            key: MemoryKey::new("pipeline", "draft::content"),
            tier: MemoryTier::Semantic,
            scope: MemoryScope::Global,
            owner_id: agent_id,
            value: json!({
                "draft": full_draft,
                "word_count": full_draft.split_whitespace().count(),
                "section_count": sections.len()
            }),
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
            version: 1,
            tags: vec!["draft".to_string(), self.topic.clone()],
        };

        memory.semantic_write(agent_id, entry).await?;

        Ok(json!({
            "draft": full_draft,
            "word_count": full_draft.split_whitespace().count(),
            "section_count": sections.len()
        }))
    }
}

// =============================================================================
// EditorAgent// =============================================================================

struct EditorAgent {
    topic: String,
    audience: String,
    word_count: usize,
    model_id: ModelId,
}

impl EditorAgent {
    fn new(topic: String, audience: String, word_count: usize, model: String) -> Self {
        let (provider_id, model_name) = parse_model_spec(&model);
        Self {
            topic,
            audience,
            word_count,
            model_id: ModelId::new(provider_id, &model_name),
        }
    }
}

impl AgentTask for EditorAgent {
    fn name(&self) -> &str { "editor-agent" }
    fn kind(&self) -> AgentKind { AgentKind::Writing }
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::new()
            .with_memory(nexus_proto::memory::MemoryAccess::ReadWrite)
            .with_model("anthropic/*")
            .with_model("openai/*")
    }

    #[instrument(skip(self, ctx))]
    async fn run(&mut self, ctx: AgentContext) -> Result<Value> {
        let agent_id = ctx.agent_id();
        let memory = ctx.memory();
        let router = ctx.router();

        // Wait for draft and research
        let draft_key = MemoryKey::new("pipeline", "draft::content");
        let research_key = MemoryKey::new("pipeline", "research::results");
        
        let draft_data = wait_for_memory_key(memory, agent_id, &draft_key, Duration::from_secs(300)).await?;
        let research_data = wait_for_memory_key(memory, agent_id, &research_key, Duration::from_secs(300)).await?;
        
        let draft = draft_data.get("draft").and_then(|v| v.as_str()).unwrap_or("");
        let brief = research_data.get("brief").and_then(|v| v.as_str()).unwrap_or("");

        // Pass 1: Structural edit
        let structural_prompt = format!(
            "Review this {} draft for structure and flow:\n\             {}\n\n\
             Provide:\n\
             1. Overall assessment (2-3 sentences)\n\
             2. List of structural issues (if any)\n\
             3. Suggested paragraph reorganizations\n\
             4. Transitions that need improvement\n\
             5. Sections that are too long/short\n\
             Return as JSON: {{'assessment': str, 'issues': [str], 'suggestions': [str]}}",
            self.content_type, draft
        );

        let structural_request = ModelRequest::builder()
            .messages(vec![Message::user(structural_prompt)])
            .model(Some(self.model_id.clone()))
            .max_tokens(1000)
            .routing_policy(RoutingPolicy::default())
            .build()?;

        let structural_response = router.complete(structural_request, agent_id).await?;
        let structural_feedback: Value = serde_json::from_str(&structural_response.message.text_content())
            .unwrap_or_else(|_| json!({}));

        // Pass 2: Line edit + rewrite
        let rewrite_prompt = format!(
            "You are a professional editor. Rewrite this draft with these improvements:\n\
             - Fix any factual inconsistencies with the research brief\n\
             - Improve clarity and flow\n\
             - Strengthen the introduction and conclusion\n\
             - Ensure the right tone for a {} audience\n\
             - Target exactly {} words\n\n\
             Research brief for fact-checking: {}\n\n\
             Structural feedback: {}\n\n\
             Original draft:\n\
             {}\n\n\
             Return the complete edited version.",
            self.audience, self.word_count, brief,
            structural_feedback, draft
        );

        let rewrite_request = ModelRequest::builder()
            .messages(vec![Message::user(rewrite_prompt)])
            .model(Some(self.model_id.clone()))
            .max_tokens(MAX_REPORT_TOKENS)
            .routing_policy(RoutingPolicy::default())
            .build()?;

        let rewrite_response = router.complete(rewrite_request, agent_id).await?;
        let edited_content = rewrite_response.message.text_content();

        // Store edited content in semantic memory        let entry = MemoryEntry {
            key: MemoryKey::new("pipeline", "edited::content"),
            tier: MemoryTier::Semantic,
            scope: MemoryScope::Global,
            owner_id: agent_id,
            value: json!({
                "edited": edited_content,
                "word_count": edited_content.split_whitespace().count(),
                "changes_summary": structural_feedback.get("assessment").and_then(|v| v.as_str()).unwrap_or("Edited for structure and flow")
            }),
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
            version: 1,
            tags: vec!["edited".to_string(), self.topic.clone()],
        };

        memory.semantic_write(agent_id, entry).await?;

        Ok(json!({
            "edited": edited_content,
            "word_count": edited_content.split_whitespace().count(),
            "changes_summary": structural_feedback.get("assessment").and_then(|v| v.as_str()).unwrap_or("Edited for structure and flow")
        }))
    }
}

// =============================================================================
// SEOAgent
// =============================================================================

struct SEOAgent {
    topic: String,
    audience: String,
    content_type: String,
    model_id: ModelId,
}

impl SEOAgent {
    fn new(topic: String, audience: String, content_type: String, model: String) -> Self {
        let (provider_id, model_name) = parse_model_spec(&model);
        Self {
            topic,
            audience,
            content_type,
            model_id: ModelId::new(provider_id, &model_name),
        }
    }
}
impl AgentTask for SEOAgent {
    fn name(&self) -> &str { "seo-agent" }
    fn kind(&self) -> AgentKind { AgentKind::Analysis }
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::new()
            .with_memory(nexus_proto::memory::MemoryAccess::ReadWrite)
            .with_model("anthropic/*")
            .with_model("openai/*")
    }

    #[instrument(skip(self, ctx))]
    async fn run(&mut self, ctx: AgentContext) -> Result<Value> {
        let agent_id = ctx.agent_id();
        let memory = ctx.memory();
        let router = ctx.router();

        // Wait for edited content
        let edited_key = MemoryKey::new("pipeline", "edited::content");
        let edited_data = wait_for_memory_key(memory, agent_id, &edited_key, Duration::from_secs(300)).await?;
        
        let edited_content = edited_data.get("edited").and_then(|v| v.as_str()).unwrap_or("");

        let seo_prompt = format!(
            "Analyze this {} and produce an SEO and metadata package:\n\
             {}\n\
             Topic: {}\n\
             Audience: {}\n\n\
             Return as JSON:\n\
             {{\n\
               \"meta_title\": str,           // 50-60 chars\n\
               \"meta_description\": str,     // 150-160 chars\n\
               \"focus_keyword\": str,\n\
               \"secondary_keywords\": [str], // 5-8 keywords\n\
               \"tags\": [str],               // 5-10 tags\n\
               \"excerpt\": str,              // 100-150 word excerpt\n\
               \"reading_time_minutes\": int,\n\
               \"content_score\": int,        // 0-100 quality assessment\n\
               \"improvement_suggestions\": [str]  // 3-5 quick wins\n\
             }}",
            self.content_type, edited_content, self.topic, self.audience
        );

        let request = ModelRequest::builder()
            .messages(vec![Message::user(seo_prompt)])
            .model(Some(self.model_id.clone()))
            .max_tokens(1000)
            .routing_policy(RoutingPolicy::default())
            .build()?;
        let response = router.complete(request, agent_id).await?;
        let response_text = response.message.text_content();
        
        // Parse JSON with fallback
        let seo_package: Value = serde_json::from_str(&response_text).or_else(|_| {
            if let Some(start) = response_text.find('{') {
                if let Some(end) = response_text[start..].rfind('}').map(|i| start + i + 1) {
                    serde_json::from_str(&response_text[start..end])
                } else {
                    Err(serde_json::Error::custom("No JSON object found"))
                }
            } else {
                Err(serde_json::Error::custom("No JSON object found"))
            }
        })?;

        // Combine into final package
        let final_package = json!({
            "topic": self.topic,
            "audience": self.audience,
            "content_type": self.content_type,
            "content": edited_content,
            "seo": seo_package,
            "generated_at": Utc::now()
        });

        // Store final package in semantic memory
        let entry = MemoryEntry {
            key: MemoryKey::new("pipeline", "final::package"),
            tier: MemoryTier::Semantic,
            scope: MemoryScope::Global,
            owner_id: agent_id,
            value: final_package.clone(),
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
            version: 1,
            tags: vec!["final".to_string(), "package".to_string(), self.topic.clone()],
        };

        memory.semantic_write(agent_id, entry).await?;

        Ok(final_package)
    }
}

// =============================================================================
// Pipeline Coordinator
// =============================================================================
struct PipelineCoordinator {
    kernel: Arc<KernelHandle>,
    memory: Arc<MemoryStore>,
    router: Arc<ModelRouter>,
    tools: Arc<nexus_tools::ToolEngine>,
    config: PipelineConfig,
    agent_ids: HashMap<String, AgentId>,
    start_time: Instant,
    progress_bars: MultiProgress,
    agent_bars: HashMap<String, ProgressBar>,
}

impl PipelineCoordinator {
    fn new(
        kernel: Arc<KernelHandle>,
        memory: Arc<MemoryStore>,
        router: Arc<ModelRouter>,
        tools: Arc<nexus_tools::ToolEngine>,
        config: PipelineConfig,
    ) -> Self {
        let mp = MultiProgress::new();
        Self {
            kernel,
            memory,
            router,
            tools,
            config,
            agent_ids: HashMap::new(),
            start_time: Instant::now(),
            progress_bars: mp,
            agent_bars: HashMap::new(),
        }
    }

    async fn run(&mut self) -> Result<PipelineResult> {
        let agent_id = AgentId::new(); // Root agent ID for the pipeline
        
        // Create progress bars for each agent
        let agents = vec!["Research", "Outline", "Writer", "Editor", "SEO"];
        for agent_name in &agents {
            let pb = self.progress_bars.add(ProgressBar::new(100));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(&format!("{{prefix:.bold}} [{{bar:40.cyan/blue}}] {{msg}}"))
                    .unwrap()
                    .prefix(&format!("[{:<9}]", agent_name)),
            );
            pb.set_message("Waiting...");
            self.agent_bars.insert(agent_name.to_string(), pb);        }

        // Spawn agents
        if self.config.parallel {
            // Run Research and Outline in parallel
            let research_agent = ResearchAgent::new(
                self.config.topic.clone(),
                self.config.audience.clone(),
                self.config.model.clone(),
            );
            let outline_agent = OutlineAgent::new(
                self.config.topic.clone(),
                self.config.audience.clone(),
                self.config.content_type.clone(),
                self.config.word_count,
                self.config.model.clone(),
            );

            let research_opts = SpawnOptions {
                name: Some("researcher".to_string()),
                priority: AgentPriority::High,
                capabilities: research_agent.capabilities(),
                supervisor_id: Some("pipeline-agents".to_string()),
                ..Default::default()
            };

            let outline_opts = SpawnOptions {
                name: Some("outliner".to_string()),
                priority: AgentPriority::High,
                capabilities: outline_agent.capabilities(),
                supervisor_id: Some("pipeline-agents".to_string()),
                ..Default::default()
            };

            let (research_id, outline_id) = tokio::try_join!(
                self.kernel.spawn(research_agent, research_opts),
                self.kernel.spawn(outline_agent, outline_opts)
            )?;

            self.agent_ids.insert("Research".to_string(), research_id);
            self.agent_ids.insert("Outline".to_string(), outline_id);

            // Wait for both to complete
            self.wait_for_agents(vec!["Research", "Outline"]).await?;
        } else {
            // Run sequentially
            let agents = vec![
                ("Research", ResearchAgent::new(
                    self.config.topic.clone(),
                    self.config.audience.clone(),                    self.config.model.clone(),
                )),
                ("Outline", OutlineAgent::new(
                    self.config.topic.clone(),
                    self.config.audience.clone(),
                    self.config.content_type.clone(),
                    self.config.word_count,
                    self.config.model.clone(),
                )),
                ("Writer", WriterAgent::new(
                    self.config.topic.clone(),
                    self.config.audience.clone(),
                    self.config.content_type.clone(),
                    self.config.word_count,
                    self.config.model.clone(),
                )),
                ("Editor", EditorAgent::new(
                    self.config.topic.clone(),
                    self.config.audience.clone(),
                    self.config.word_count,
                    self.config.model.clone(),
                )),
                ("SEO", SEOAgent::new(
                    self.config.topic.clone(),
                    self.config.audience.clone(),
                    self.config.content_type.clone(),
                    self.config.model.clone(),
                )),
            ];

            for (name, agent) in agents {
                if let Some(pb) = self.agent_bars.get(name) {
                    pb.set_message("Starting...");
                }

                let opts = SpawnOptions {
                    name: Some(format!("{}-agent", name.to_lowercase())),
                    priority: AgentPriority::High,
                    capabilities: agent.capabilities(),
                    supervisor_id: Some("pipeline-agents".to_string()),
                    ..Default::default()
                };

                let agent_id = self.kernel.spawn(agent, opts).await?;
                self.agent_ids.insert(name.to_string(), agent_id);

                // Wait for this agent to complete
                self.wait_for_agent(name, agent_id).await?;
            }
        }
        // Retrieve final package
        let final_key = MemoryKey::new("pipeline", "final::package");
        let final_package = wait_for_memory_key(&self.memory, agent_id, &final_key, Duration::from_secs(60)).await?;

        // Collect agent results (simplified - in production, would track via episodic memory)
        let mut agent_results = HashMap::new();
        for (name, id) in &self.agent_ids {
            // In production, would query episodic memory for each agent's result
            agent_results.insert(name.clone(), json!({}));
        }

        let total_duration = self.start_time.elapsed();
        let cost_summary = CostSummary::default(); // Would calculate from cost ledger in production

        Ok(PipelineResult {
            topic: self.config.topic.clone(),
            content_type: self.config.content_type.clone(),
            audience: self.config.audience.clone(),
            final_package,
            agent_results,
            total_duration,
            cost_summary,
        })
    }

    async fn wait_for_agent(&self, name: &str, agent_id: AgentId) -> Result<()> {
        let mut interval = interval(Duration::from_millis(500));
        let mut event_rx = self.kernel.subscribe_events();
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Some(meta) = self.kernel.get_meta(agent_id) {
                        match meta.status {
                            nexus_proto::agent::AgentStatus::Running { .. } => {
                                if let Some(pb) = self.agent_bars.get(name) {
                                    pb.set_message("Running...");
                                    pb.set_position(50);
                                }
                            }
                            nexus_proto::agent::AgentStatus::Completed { .. } => {
                                if let Some(pb) = self.agent_bars.get(name) {
                                    pb.finish_with_message("✓ Done");
                                }
                                break;
                            }
                            nexus_proto::agent::AgentStatus::Failed { error, .. } => {
                                if let Some(pb) = self.agent_bars.get(name) {
                                    pb.finish_with_message(&format!("✗ Failed: {}", error));                                }
                                bail!("Agent {} failed: {}", name, error);
                            }
                            _ => {}
                        }
                    }
                }
                Ok(event) = event_rx.recv() => {
                    use nexus_kernel::KernelEvent;
                    match event {
                        KernelEvent::AgentTerminated { id, success, .. } if id == agent_id => {
                            if success {
                                if let Some(pb) = self.agent_bars.get(name) {
                                    pb.finish_with_message("✓ Done");
                                }
                            } else {
                                if let Some(pb) = self.agent_bars.get(name) {
                                    pb.finish_with_message("✗ Failed");
                                }
                                bail!("Agent {} terminated unsuccessfully", name);
                            }
                            break;
                        }
                        _ => {}
                    }
                }
                _ = sleep(Duration::from_secs(300)) => {
                    bail!("Timeout waiting for agent {}", name);
                }
            }
        }
        Ok(())
    }

    async fn wait_for_agents(&self, names: Vec<&str>) -> Result<()> {
        let mut handles = Vec::new();
        for name in names {
            if let Some(&agent_id) = self.agent_ids.get(name) {
                handles.push(self.wait_for_agent(name, agent_id));
            }
        }
        futures::future::join_all(handles)
            .await
            .into_iter()
            .collect::<Result<()>>()
    }
}

// =============================================================================
// Main Function// =============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Print banner
    println!("{}", "NEXUS MULTI-AGENT PIPELINE".bold().cyan());
    println!("{}", "Content Production Assembly Line".dimmed());
    println!("{}", "══════════════════════════════════".dimmed());

    // Parse CLI arguments
    let cli = Cli::parse();
    
    // Validate word count
    if cli.word_count < 500 || cli.word_count > 10000 {
        bail!("Word count must be between 500 and 10000");
    }

    // Check for API keys
    let has_anthropic = std::env::var("ANTHROPIC_API_KEY").is_ok();
    let has_openai = std::env::var("OPENAI_API_KEY").is_ok();
    
    if !has_anthropic && !has_openai {
        eprintln!("❌ No API keys found. Please set one of:");
        eprintln!("   export ANTHROPIC_API_KEY='your-key-here'");
        eprintln!("   export OPENAI_API_KEY='your-key-here'");
        std::process::exit(1);
    }

    // Print provider info
    if has_anthropic {
        println!("{} Provider: Anthropic Claude 3.5 Sonnet", "✓".green().bold());
    } else if has_openai {
        println!("{} Provider: OpenAI GPT-4o", "✓".green().bold());
    }

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Start runtime with progress
    println!("\n{} Starting Nexus runtime...", "⚡".bold().cyan());
    
    // Build minimal config
    let memory_config = MemoryConfig {
        episodic_db_path: "./data/pipeline-episodic.db".into(),
        semantic_index_path: "./data/pipeline-semantic.index".into(),
        semantic_meta_db_path: "./data/pipeline-semantic-meta.db".into(),
        semantic_dimensions: 768,        ..Default::default()
    };

    // Initialize memory store with local embeddings
    let embedder = Arc::new(LocalEmbeddingProvider::default());
    let memory = Arc::new(MemoryStore::new(memory_config).await?);

    // Initialize tool engine
    let tool_config = ToolEngineConfig {
        registry_path: "./tools/registry".into(),
        ..Default::default()
    };
    let root_guard = Arc::new(CapabilityGuard::new(Uuid::nil(), CapabilitySet::all()));
    let tools = Arc::new(nexus_tools::ToolEngine::new(tool_config, root_guard).await?);

    // Initialize model router
    let mut provider_registry = ProviderRegistry::new();
    
    if has_anthropic {
        let anthropic_config = AnthropicConfig {
            api_key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            ..Default::default()
        };
        let provider = Arc::new(nexus_router::providers::anthropic::AnthropicProvider::new(anthropic_config));
        provider_registry.register(provider);
    }
    
    if has_openai {
        let openai_config = OpenAIConfig {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            ..Default::default()
        };
        let provider = Arc::new(nexus_router::providers::openai::OpenAIProvider::new(openai_config));
        provider_registry.register(provider);
    }

    let router_config = RouterConfig {
        default_policy: RoutingPolicy::CostOptimized { max_latency_ms: 5000 },
        ..Default::default()
    };
    let cost_ledger = Arc::new(nexus_obs::ledger::PersistentCostLedger::new("./data/cost.db").await?);
    let router = Arc::new(ModelRouter::new(router_config, Arc::new(provider_registry), cost_ledger.memory_ledger()).await);

    // Initialize kernel
    let kernel_config = KernelConfig {
        max_agents: 20,
        ..Default::default()
    };
    let (kernel, _msg_rx) = Kernel::new(kernel_config).await?;
    let kernel_handle = Arc::new(kernel.handle());
    println!("{} Runtime ready", "✓".green().bold());

    // Create supervisor tree
    kernel_handle
        .add_supervisor(
            "pipeline-root",
            RestartStrategy::OneForAll {
                max_restarts: 1,
                window_secs: 300,
            },
            None,
        )
        .await?;
    
    kernel_handle
        .add_supervisor(
            "pipeline-agents",
            RestartStrategy::RestForOne {
                max_restarts: 2,
                window_secs: 120,
            },
            Some("pipeline-root".to_string()),
        )
        .await?;

    // Print pipeline visualization
    println!("\n{}", "Pipeline Visualization:".bold().cyan());
    if cli.parallel {
        println!("  ┌─────────────┐ ═══ ┌──────────────┐");
        println!("  │  Research   │     │   Outline    │");
        println!("  │   Agent     │     │    Agent     │");
        println!("  └─────────────┘     └──────┬───────┘");
        println!("                              │");
        println!("                              ▼");
        println!("  ┌──────────────┐     ┌──────────────┐");
        println!("  │    Writer    │────▶│    Editor    │");
        println!("  │    Agent     │     │    Agent     │");
        println!("  └──────────────┘     └──────┬───────┘");
        println!("                              │");
        println!("                              ▼");
        println!("  ┌──────────────┐");
        println!("  │     SEO      │");
        println!("  │    Agent     │");
        println!("  └──────────────┘");
    } else {
        println!("  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐");
        println!("  │  Research   │────▶│   Outline    │────▶│    Writer    │");
        println!("  │   Agent     │     │    Agent     │     │    Agent     │");
        println!("  └─────────────┘     └──────────────┘     └──────┬───────┘");        println!("                                                  │");
        println!("                         ┌──────────────┐     ┌──▼───────┐");
        println!("                         │     SEO      │◀────│  Editor  │");
        println!("                         │    Agent     │     │  Agent   │");
        println!("                         └──────────────┘     └──────────┘");
    }

    // Print pipeline config
    println!("\n{} Pipeline Configuration:", "⚙️".bold().cyan());
    println!("  Topic:         {}", cli.topic.bold().yellow());
    println!("  Audience:      {}", cli.audience);
    println!("  Content type:  {}", cli.content_type);
    println!("  Target words:  {}", cli.word_count);
    println!("  Model:         {}", cli.model);
    println!("  Mode:          {}", if cli.parallel { "Parallel".green() } else { "Sequential".blue() });

    // Create and run pipeline coordinator
    let pipeline_config = PipelineConfig {
        topic: cli.topic.clone(),
        audience: cli.audience.clone(),
        content_type: cli.content_type.clone(),
        word_count: cli.word_count,
        model: cli.model.clone(),
        parallel: cli.parallel,
        live_output: cli.live_output,
        save_dir: cli.save_dir.clone(),
    };

    let mut coordinator = PipelineCoordinator::new(
        kernel_handle,
        memory,
        router,
        tools,
        pipeline_config,
    );

    println!("\n{} Starting pipeline execution...", "🚀".bold().green());
    let result = coordinator.run().await?;

    // Display final content package
    println!("\n{}", "═".repeat(50).dimmed());
    println!("{} {}", "📦".bold().cyan(), "CONTENT PACKAGE READY".bold());
    println!("{}", "═".repeat(50).dimmed());
    
    if let Some(title) = result.final_package.get("seo").and_then(|v| v.get("meta_title")).and_then(|v| v.as_str()) {
        println!("\n📰 TITLE: {}", title.bold().yellow());
    }
    if let Some(keyword) = result.final_package.get("seo").and_then(|v| v.get("focus_keyword")).and_then(|v| v.as_str()) {
        println!("🔑 FOCUS KEYWORD: {}", keyword.bold().magenta());
    }    if let Some(read_time) = result.final_package.get("seo").and_then(|v| v.get("reading_time_minutes")).and_then(|v| v.as_i64()) {
        println!("⏱️  READING TIME: {} minutes", read_time);
    }
    if let Some(score) = result.final_package.get("seo").and_then(|v| v.get("content_score")).and_then(|v| v.as_i64()) {
        println!("📊 CONTENT SCORE: {}/100", score);
    }

    println!("\n{} META", "─".repeat(50).dimmed());
    if let Some(meta_title) = result.final_package.get("seo").and_then(|v| v.get("meta_title")).and_then(|v| v.as_str()) {
        println!("Title:       {}", meta_title);
    }
    if let Some(meta_desc) = result.final_package.get("seo").and_then(|v| v.get("meta_description")).and_then(|v| v.as_str()) {
        println!("Description: {}", meta_desc);
    }
    if let Some(tags) = result.final_package.get("seo").and_then(|v| v.get("tags")).and_then(|v| v.as_array()) {
        let tags_str: Vec<String> = tags.iter().filter_map(|v| v.as_str().map(String::from)).collect();
        println!("Tags:        {}", tags_str.join(", "));
    }

    println!("\n{} ARTICLE", "─".repeat(50).dimmed());
    if let Some(content) = result.final_package.get("content").and_then(|v| v.as_str()) {
        // Word-wrap at 80 columns
        let wrapped = textwrap::fill(content, 80);
        println!("{}", wrapped);
    }

    println!("\n{} EXCERPT", "─".repeat(50).dimmed());
    if let Some(excerpt) = result.final_package.get("seo").and_then(|v| v.get("excerpt")).and_then(|v| v.as_str()) {
        println!("{}", excerpt);
    }

    println!("\n{} SEO SUGGESTIONS", "─".repeat(50).dimmed());
    if let Some(suggestions) = result.final_package.get("seo").and_then(|v| v.get("improvement_suggestions")).and_then(|v| v.as_array()) {
        for (i, suggestion) in suggestions.iter().enumerate() {
            if let Some(s) = suggestion.as_str() {
                println!("{}. {}", i + 1, s);
            }
        }
    }

    // Print pipeline stats
    println!("\n{} Pipeline Summary:", "⚡".bold().cyan());
    println!("  Total time:      {:.2}s", result.total_duration.as_secs_f32());
    println!("  Agents run:      {}", result.agent_results.len());
    println!("  Total words:     {}", result.final_package.get("content").and_then(|v| v.as_str()).map(|s| s.split_whitespace().count()).unwrap_or(0));
    
    // Cost breakdown (simplified)
    println!("\n{} Cost Breakdown:", "💰".bold().magenta());
    println!("  {:<15} {:<12} {:<8} {:<8} {:<8}", "Agent", "Provider", "Input", "Output", "Cost");
    println!("  {}", "─".repeat(50));    for (agent, cost) in &result.cost_summary.by_agent {
        println!("  {:<15} {:<12} {:<8} {:<8} ${:<7.3}", 
            agent, cost.provider, cost.input_tokens, cost.output_tokens, cost.cost_usd);
    }
    println!("  {}", "─".repeat(50));
    println!("  {:<15} {:<12} {:<8} {:<8} ${:<7.3}", 
        "TOTAL", "", 
        result.cost_summary.total_input_tokens, 
        result.cost_summary.total_output_tokens, 
        result.cost_summary.total_cost_usd);

    // Save to directory if requested
    if let Some(save_dir) = cli.save_dir {
        std::fs::create_dir_all(&save_dir)?;
        
        if let Some(content) = result.final_package.get("content").and_then(|v| v.as_str()) {
            std::fs::write(save_dir.join("article.md"), content)?;
        }
        if let Some(meta) = result.final_package.get("seo") {
            std::fs::write(save_dir.join("meta.json"), serde_json::to_string_pretty(meta)?)?;
        }
        // Would save outline, research, etc. in production
        
        println!("\n{} Outputs saved to: {}", "✓".green(), save_dir.display());
    }

    // Shutdown cleanly
    println!("\n{} Shutting down runtime...", "🛑".bold().yellow());
    kernel_handle.shutdown(Some(Duration::from_secs(10))).await?;
    println!("{} Done. Runtime shut down cleanly.", "✓".green().bold());

    Ok(())
}

// Helper for text wrapping (simple implementation)
mod textwrap {
    pub fn fill(text: &str, width: usize) -> String {
        let mut result = String::new();
        let mut line = String::new();
        
        for word in text.split_whitespace() {
            if line.len() + word.len() + 1 > width && !line.is_empty() {
                result.push_str(&line);
                result.push('\n');
                line.clear();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);        }
        if !line.is_empty() {
            result.push_str(&line);
        }
        result
    }
}
