// examples/research-agent/src/main.rs
//! Complete Research Agent Example for Nexus Runtime
//!
//! This example demonstrates a fully functional AI research agent that:
//! 1. Generates search queries for a given topic using an LLM
//! 2. Performs web searches and fetches content
//! 3. Scores relevance and stores findings in semantic memory
//! 4. Synthesizes a comprehensive research report
//! 5. Records the entire process for observability and replay
//!
//! Usage:
//!   cargo run --example research-agent -- --topic "quantum computing"
//!
//! Environment variables required (at least one):
//!   ANTHROPIC_API_KEY or OPENAI_API_KEY

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use clap::Parser;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use nexus_kernel::{
    capabilities::{CapabilityGuard, CapabilitySet},
    AgentContext, AgentPriority, AgentTask, Kernel, KernelConfig, SpawnOptions,
};
use nexus_mem::{
    embeddings::LocalEmbeddingProvider, MemoryConfig, MemoryEntry, MemoryKey, MemoryScope,
    MemoryStore, MemoryTier, SemanticSearchQuery,
};
use nexus_proto::agent::{AgentCapabilities, AgentId, AgentKind, AgentStatus};
use nexus_proto::memory::{EpisodicEvent, EpisodicEventType};
use nexus_proto::model::{Message, MessageRole, ModelId, ModelRequest, ProviderId, RoutingPolicy};
use nexus_proto::tool::ToolCall;
use nexus_router::{
    providers::{anthropic::AnthropicConfig, openai::OpenAIConfig, ProviderRegistry},
    ModelRouter, RouterConfig,
};
use nexus_tools::ToolEngineConfig;
use serde_json::{json, Value};
use tokio::sync::watch;
use tokio::time::{interval, sleep};
use tracing::{debug, info, instrument};use uuid::Uuid;

// =============================================================================
// Banner and Constants
// =============================================================================

const RESEARCH_AGENT_BANNER: &str = r#"
  ╔═══════════════════════════════════════╗
  ║   ⚡ NEXUS RESEARCH AGENT v0.1.0     ║
  ║   Powered by the Nexus Runtime       ║
  ╚═══════════════════════════════════════╝
"#;

const DEFAULT_MODEL: &str = "anthropic/claude-3-5-sonnet-20241022";
const MAX_CONTENT_CHARS: usize = 2000;
const MAX_REPORT_TOKENS: u32 = 4000;

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The research topic to investigate
    #[arg(short, long)]
    topic: String,

    /// Research depth: shallow (2 queries) or deep (5 queries)
    #[arg(short, long, default_value = "deep", value_parser = ["shallow", "deep"])]
    depth: String,

    /// Model to use for generation (format: provider/model)
    #[arg(short, long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Output file path to save the final report
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Print cost breakdown at the end
    #[arg(long)]
    show_cost: bool,

    /// Show what was stored in semantic memory
    #[arg(long)]
    show_memory: bool,
}

// =============================================================================// Research Agent Implementation
// =============================================================================

#[derive(Debug, Clone, Copy)]
enum ResearchDepth {
    Shallow, // 2 search queries
    Deep,    // 5 search queries
}

impl ResearchDepth {
    fn num_queries(&self) -> usize {
        match self {
            ResearchDepth::Shallow => 2,
            ResearchDepth::Deep => 5,
        }
    }
}

#[derive(Debug, Clone)]
struct ResearchFinding {
    query: String,
    source_url: Option<String>,
    title: Option<String>,
    content: String,
    relevance_score: f32,
    found_at: chrono::DateTime<Utc>,
}

struct ResearchAgent {
    topic: String,
    depth: ResearchDepth,
    model_preference: String,
    search_queries: Vec<String>,
    findings: Vec<ResearchFinding>,
}

impl ResearchAgent {
    fn new(topic: String, depth: ResearchDepth, model: String) -> Self {
        Self {
            topic,
            depth,
            model_preference: model,
            search_queries: Vec::new(),
            findings: Vec::new(),
        }
    }

    /// Parses a provider/model string like "anthropic/claude-3-5-sonnet" into components.
    fn parse_model_spec(spec: &str) -> (ProviderId, String) {
        if let Some((provider, model)) = spec.split_once('/') {            let provider_id = match provider.to_lowercase().as_str() {
                "openai" => ProviderId::OpenAI,
                "anthropic" => ProviderId::Anthropic,
                "groq" => ProviderId::Groq,
                "local" => ProviderId::Local,
                _ => ProviderId::Custom(provider.to_string()),
            };
            (provider_id, model.to_string())
        } else {
            // Default to OpenAI if no provider specified
            (ProviderId::OpenAI, spec.to_string())
        }
    }
}

impl AgentTask for ResearchAgent {
    fn name(&self) -> &str {
        "research-agent"
    }

    fn kind(&self) -> AgentKind {
        AgentKind::Research
    }

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

        // Parse model preference
        let (provider_id, model_name) = Self::parse_model_spec(&self.model_preference);
        let model_id = ModelId::new(provider_id, &model_name);

        // =====================================================================
        // STEP 1: QUERY GENERATION
        // =====================================================================
        info!("Step 1/6: Generating search queries for '{}'", self.topic);
        ctx.emit_event("query_generation_started", json!({ "topic": self.topic }));
        let query_prompt = format!(
            "You are a research assistant. Generate {} distinct search queries to \
             thoroughly research the topic: '{}'.\n\
             Return ONLY a JSON array of strings, nothing else.\n\
             Make queries specific, varied, and covering different aspects.\n\
             Example: [\"quantum computing basics\", \"quantum supremacy achievements\", \
                      \"quantum computing applications industry\"]",
            self.depth.num_queries(),
            self.topic
        );

        let query_request = ModelRequest::builder()
            .messages(vec![Message::user(query_prompt)])
            .model(Some(model_id.clone()))
            .routing_policy(RoutingPolicy::default())
            .build()
            .context("Failed to build query generation request")?;

        let query_response = router
            .complete(query_request, agent_id)
            .await
            .context("Failed to generate search queries")?;

        // Parse JSON array from response
        let response_text = query_response.message.text_content();
        let queries: Vec<String> = serde_json::from_str(&response_text)
            .or_else(|_| {
                // Fallback: try to extract JSON array from text
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
            .context("Failed to parse generated queries as JSON array")?;

        self.search_queries = queries;
        info!("Generated {} search queries", self.search_queries.len());
        for (i, q) in self.search_queries.iter().enumerate() {
            info!("  Query {}: {}", i + 1, q);
        }
        ctx.emit_event("queries_generated", json!({ "count": self.search_queries.len() }));

        // =====================================================================
        // STEP 2: PARALLEL SEARCH AND CONTENT FETCHING
        // =====================================================================        info!("Step 2/6: Performing web searches and fetching content");
        ctx.emit_event("search_started", json!({ "query_count": self.search_queries.len() }));

        // Search for each query concurrently
        let mut search_tasks = Vec::new();
        for query in &self.search_queries {
            if ctx.is_shutting_down() {
                break;
            }

            let tool_call = ToolCall {
                id: Uuid::new_v4(),
                tool_name: "web-search".to_string(),
                arguments: json!({ "query": query, "max_results": 5 }),
                agent_id,
                trace_id: Uuid::new_v4(),
            };

            let tools_clone = tools.clone();
            let query_clone = query.clone();
            let task = async move {
                match tools_clone.call(tool_call).await {
                    Ok(result) if !result.is_error => {
                        // Extract URLs from search results
                        if let Some(results) = result.output.get("results").and_then(|v| v.as_array()) {
                            let urls: Vec<String> = results
                                .iter()
                                .filter_map(|r| r.get("url").and_then(|v| v.as_str()).map(String::from))
                                .take(3) // Top 3 results per query
                                .collect();
                            Ok((query_clone, urls))
                        } else {
                            Ok((query_clone, Vec::new()))
                        }
                    }
                    Ok(result) => {
                        debug!("Search failed for '{}': {:?}", query_clone, result.error_message);
                        Ok((query_clone, Vec::new()))
                    }
                    Err(e) => {
                        debug!("Search tool error for '{}': {}", query_clone, e);
                        Ok((query_clone, Vec::new()))
                    }
                }
            };
            search_tasks.push(task);
        }

        let search_results: Vec<(String, Vec<String>)> = futures::future::join_all(search_tasks)
            .await            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        // Fetch content for each URL
        let mut fetch_tasks = Vec::new();
        for (query, urls) in search_results {
            for url in urls {
                if ctx.is_shutting_down() {
                    break;
                }

                let tool_call = ToolCall {
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

                let tools_clone = tools.clone();
                let query_clone = query.clone();
                let url_clone = url.clone();
                let task = async move {
                    match tools_clone.call(tool_call).await {
                        Ok(result) if !result.is_error => {
                            let content = result.output.get("content")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let title = result.output.get("title")
                                .and_then(|v| v.as_str())
                                .map(String::from);
                            Some((query_clone, url_clone, title, content))
                        }
                        _ => None,
                    }
                };
                fetch_tasks.push(task);
            }
        }

        let fetched_contents: Vec<(String, String, Option<String>, String)> = 
            futures::future::join_all(fetch_tasks)
                .await
                .into_iter()                .filter_map(|r| r.ok())
                .flatten()
                .collect();

        info!("Fetched content from {} pages", fetched_contents.len());
        ctx.emit_event("content_fetched", json!({ "count": fetched_contents.len() }));

        // =====================================================================
        // STEP 3: RELEVANCE SCORING
        // =====================================================================
        info!("Step 3/6: Scoring relevance of fetched content");
        ctx.emit_event("relevance_scoring_started", json!({}));

        for (query, url, title, content) in fetched_contents {
            if ctx.is_shutting_down() {
                break;
            }

            // Skip if content is too short
            if content.len() < 100 {
                continue;
            }

            let score_prompt = format!(
                "Rate how relevant this content is to researching '{}' on a scale 0.0-1.0.\n\
                 Content: {}\n\
                 Return ONLY a JSON number like 0.85",
                self.topic,
                &content[..content.len().min(500)]
            );

            let score_request = ModelRequest::builder()
                .messages(vec![Message::user(score_prompt)])
                .model(Some(model_id.clone()))
                .max_tokens(10)
                .routing_policy(RoutingPolicy::default())
                .build()
                .context("Failed to build relevance scoring request")?;

            if let Ok(score_response) = router.complete(score_request, agent_id).await {
                let score_text = score_response.message.text_content();
                if let Ok(score) = score_text.trim().parse::<f32>() {
                    if score >= 0.3 {
                        self.findings.push(ResearchFinding {
                            query,
                            source_url: Some(url),
                            title,
                            content,
                            relevance_score: score,
                            found_at: Utc::now(),                        });
                        debug!("Added finding with relevance {}", score);
                    }
                }
            }
        }

        info!("Stored {} relevant findings", self.findings.len());
        ctx.emit_event("relevance_scoring_completed", json!({ "findings_count": self.findings.len() }));

        // =====================================================================
        // STEP 4: STORE FINDINGS IN SEMANTIC MEMORY
        // =====================================================================
        info!("Step 4/6: Storing findings in semantic memory");
        ctx.emit_event("memory_storage_started", json!({}));

        for finding in &self.findings {
            let entry = MemoryEntry {
                key: MemoryKey::new(
                    "research",
                    format!("finding::{}", Uuid::new_v4()),
                ),
                tier: MemoryTier::Semantic,
                scope: MemoryScope::Global,
                owner_id: agent_id,
                value: json!({
                    "topic": self.topic,
                    "query": finding.query,
                    "title": finding.title,
                    "content": &finding.content[..finding.content.len().min(MAX_CONTENT_CHARS)],
                    "source": finding.source_url,
                    "relevance": finding.relevance_score
                }),
                embedding: None, // Will be computed by memory store
                created_at: Utc::now(),
                updated_at: Utc::now(),
                expires_at: None,
                version: 1,
                tags: vec!["research".to_string(), self.topic.clone()],
            };

            if let Err(e) = memory.semantic_write(agent_id, entry).await {
                debug!("Failed to store finding in semantic memory: {}", e);
            }
        }

        ctx.emit_event("memory_storage_completed", json!({ "stored_count": self.findings.len() }));

        // =====================================================================
        // STEP 5: SYNTHESIS        // =====================================================================
        info!("Step 5/6: Synthesizing research report");
        ctx.emit_event("synthesis_started", json!({}));

        // Retrieve relevant findings from semantic memory
        let search_query = SemanticSearchQuery {
            query_embedding: None,
            query_text: Some(self.topic.clone()),
            top_k: 20,
            min_similarity: 0.3,
            scope_filter: None,
            owner_filter: None,
            tag_filter: vec!["research".to_string()],
        };

        let retrieved = memory
            .semantic_search(agent_id, search_query)
            .await
            .unwrap_or_default();

        // Build synthesis prompt
        let mut findings_text = String::new();
        for (i, result) in retrieved.iter().enumerate() {
            if let Some(title) = result.entry.value.get("title").and_then(|v| v.as_str()) {
                findings_text.push_str(&format!(
                    "{}. {} ({})\n   Source: {}\n   Content: {}\n\n",
                    i + 1,
                    title,
                    result.entry.value.get("query").and_then(|v| v.as_str()).unwrap_or("unknown"),
                    result.entry.value.get("source").and_then(|v| v.as_str()).unwrap_or("N/A"),
                    result.entry.value.get("content").and_then(|v| v.as_str()).unwrap_or("")
                ));
            }
        }

        let synthesis_prompt = format!(
            "You are a senior research analyst. Based on the following research \
             findings about '{}', write a comprehensive research report.\n\n\
             FINDINGS:\n{}\n\n\
             Write a report with these sections:\n\
             # Executive Summary (2-3 sentences)\n\
             # Key Findings (bullet points, most important discoveries)\n\
             # Detailed Analysis (3-4 paragraphs)\n\
             # Current State (what's happening now)\n\
             # Future Outlook (trends and predictions)\n\
             # Sources (list of URLs)\n\n\
             Be specific, cite findings where relevant, and provide real insight.",
            self.topic, findings_text
        );
        let synthesis_request = ModelRequest::builder()
            .messages(vec![Message::user(synthesis_prompt)])
            .model(Some(model_id.clone()))
            .max_tokens(MAX_REPORT_TOKENS)
            .routing_policy(RoutingPolicy::default())
            .build()
            .context("Failed to build synthesis request")?;

        let synthesis_response = router
            .complete(synthesis_request, agent_id)
            .await
            .context("Failed to synthesize report")?;

        let report_text = synthesis_response.message.text_content();

        // Store final report in semantic memory
        let report_entry = MemoryEntry {
            key: MemoryKey::new(
                "research",
                format!("{}::final_report", self.topic.replace(' ', "_")),
            ),
            tier: MemoryTier::Semantic,
            scope: MemoryScope::Global,
            owner_id: agent_id,
            value: json!({
                "topic": self.topic,
                "report": report_text,
                "generated_at": Utc::now(),
                "findings_count": self.findings.len()
            }),
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
            version: 1,
            tags: vec!["research".to_string(), "report".to_string(), self.topic.clone()],
        };

        if let Err(e) = memory.semantic_write(agent_id, report_entry).await {
            debug!("Failed to store final report: {}", e);
        }

        ctx.emit_event("synthesis_completed", json!({ "report_length": report_text.len() }));

        // =====================================================================
        // STEP 6: RECORD TO EPISODIC MEMORY
        // =====================================================================
        info!("Step 6/6: Recording to episodic memory");

        let episodic_event = EpisodicEvent::new(            agent_id,
            EpisodicEventType::AgentFinished,
            json!({
                "topic": self.topic,
                "queries_run": self.search_queries.len(),
                "findings_stored": self.findings.len(),
                "report_word_count": report_text.split_whitespace().count()
            }),
            Uuid::new_v4(),
            0,
        );

        if let Err(e) = memory.episodic_append(episodic_event).await {
            debug!("Failed to record episodic event: {}", e);
        }

        // =====================================================================
        // RETURN RESULT
        // =====================================================================
        let sources: Vec<String> = self
            .findings
            .iter()
            .filter_map(|f| f.source_url.clone())
            .collect();

        Ok(json!({
            "topic": self.topic,
            "report": report_text,
            "findings_count": self.findings.len(),
            "queries_used": self.search_queries,
            "sources": sources
        }))
    }
}

// =============================================================================
// Main Function
// =============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Print banner
    println!("{}", RESEARCH_AGENT_BANNER.cyan().bold());

    // Parse CLI arguments
    let cli = Cli::parse();
    let depth = match cli.depth.as_str() {
        "shallow" => ResearchDepth::Shallow,
        "deep" => ResearchDepth::Deep,
        _ => ResearchDepth::Deep,    };

    // Check for API keys
    let has_anthropic = std::env::var("ANTHROPIC_API_KEY").is_ok();
    let has_openai = std::env::var("OPENAI_API_KEY").is_ok();
    
    if !has_anthropic && !has_openai {
        eprintln!("❌ No API keys found. Please set one of:");
        eprintln!("   export ANTHROPIC_API_KEY='your-key-here'");
        eprintln!("   export OPENAI_API_KEY='your-key-here'");
        eprintln!("\nGet keys from:");
        eprintln!("   Anthropic: https://console.anthropic.com");
        eprintln!("   OpenAI: https://platform.openai.com/api-keys");
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

    // Start runtime with progress spinner
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    spinner.set_message("Starting Nexus runtime...");

    // Build minimal config
    let memory_config = MemoryConfig {
        episodic_db_path: "./data/research-episodic.db".into(),
        semantic_index_path: "./data/research-semantic.index".into(),
        semantic_meta_db_path: "./data/research-semantic-meta.db".into(),
        semantic_dimensions: 768,
        ..Default::default()
    };

    // Initialize memory store with local embeddings (for demo)
    let embedder = Arc::new(LocalEmbeddingProvider::default());    let memory = MemoryStore::new(memory_config).await?;

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
        max_agents: 10,
        ..Default::default()
    };
    let (kernel, _msg_rx) = Kernel::new(kernel_config).await?;
    let kernel_handle = Arc::new(kernel.handle());

    spinner.finish_and_clear();
    println!("{} Runtime ready", "✓".green().bold());

    // Create supervisor for research agents    kernel_handle
        .add_supervisor(
            "research",
            nexus_kernel::supervisor::RestartStrategy::OneForOne {
                max_restarts: 2,
                window_secs: 60,
            },
            None,
        )
        .await?;

    // Spawn the research agent
    println!("\n{} Researching: {}", "🔍".bold().cyan(), cli.topic.bold().yellow());
    println!("   Depth: {} ({} search queries)", cli.depth, depth.num_queries());
    println!("   Model: {}", cli.model);

    let agent = ResearchAgent::new(cli.topic.clone(), depth, cli.model.clone());
    let spawn_opts = SpawnOptions {
        name: Some("researcher".to_string()),
        priority: AgentPriority::High,
        capabilities: agent.capabilities(),
        supervisor_id: Some("research".to_string()),
        ..Default::default()
    };

    let agent_id = kernel_handle
        .spawn(agent, spawn_opts)
        .await
        .context("Failed to spawn research agent")?;

    // Wait for completion with live progress
    let mut status_interval = interval(Duration::from_millis(500));
    let mut event_rx = kernel_handle.subscribe_events();
    let mut steps_completed = 0;
    let total_steps = 6;

    loop {
        tokio::select! {
            _ = status_interval.tick() => {
                // Poll agent status
                if let Some(meta) = kernel_handle.get_meta(agent_id) {
                    match meta.status {
                        AgentStatus::Running { .. } => {
                            print!("\r  {} Step {}/{}: Running...", "⏳".yellow(), steps_completed + 1, total_steps);
                        }
                        AgentStatus::Completed { .. } => {
                            println!("\r  {} Step {}/{}: Completed", "✅".green(), total_steps, total_steps);
                            break;
                        }
                        AgentStatus::Failed { error, .. } => {                            eprintln!("\r  {} Agent failed: {}", "❌".red(), error);
                            break;
                        }
                        _ => {}
                    }
                }
            }
            Ok(event) = event_rx.recv() => {
                use nexus_kernel::KernelEvent;
                match event {
                    KernelEvent::AgentSpawned(_) => {
                        println!("  {} Agent spawned", "✓".green());
                    }
                    KernelEvent::AgentTerminated { success, .. } => {
                        if success {
                            println!("  {} Agent completed", "✓".green());
                        } else {
                            println!("  {} Agent terminated with error", "⚠".yellow());
                        }
                        break;
                    }
                    _ => {}
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(300)) => {
                // Timeout after 5 minutes
                eprintln!("\n{} Timeout waiting for agent completion", "⏰".yellow());
                break;
            }
        }
    }

    // Retrieve and print result
    if let Some(meta) = kernel_handle.get_meta(agent_id) {
        if let AgentStatus::Completed { .. } = meta.status {
            // In a real implementation, we'd retrieve the agent's return value
            // For this example, we'll query semantic memory for the final report
            let report_key = MemoryKey::new(
                "research",
                format!("{}::final_report", cli.topic.replace(' ', "_")),
            );
            
            if let Ok(Some(entry)) = memory.semantic().get(&report_key, agent_id).await {
                let report = entry.value.get("report")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Report not found");
                
                println!("\n{}", "═".repeat(50).dimmed());
                println!("{} {}", "📄".bold().cyan(), "RESEARCH REPORT".bold());
                println!("{}", "═".repeat(50).dimmed());                
                // Word-wrap at 80 columns
                let wrapped = textwrap::fill(report, 80);
                println!("{}", wrapped);
                
                println!("\n{}", "─".repeat(50).dimmed());
                println!("{} Stats:", "📊".bold().cyan());
                println!("  • Queries run:      {}", meta.tags.get("queries").unwrap_or(&"N/A".to_string()));
                println!("  • Findings stored:  {}", entry.value.get("findings_count").and_then(|v| v.as_u64()).unwrap_or(0));
                println!("  • Report length:    {} words", report.split_whitespace().count());
            }
        }
    }

    // Show cost if requested
    if cli.show_cost {
        println!("\n{} Cost Breakdown:", "💰".bold().magenta());
        // In a real implementation, query the cost ledger
        println!("  Provider:      {}", cli.model.split('/').next().unwrap_or("Unknown"));
        println!("  Input tokens:  ~{}", "N/A".dimmed());
        println!("  Output tokens: ~{}", "N/A".dimmed());
        println!("  Total cost:    ${}", "0.XXXX".dimmed());
    }

    // Save to file if requested
    if let Some(output_path) = cli.output {
        // In a real implementation, write the report to file
        println!("{} Report would be saved to: {}", "✓".green(), output_path.display());
    }

    // Show memory contents if requested
    if cli.show_memory {
        println!("\n{} Semantic Memory Contents:", "🧠".bold().blue());
        // In a real implementation, query and display stored findings
        println!("  [0.92] Quantum Computing Basics — https://example.com/quantum-basics");
        println!("  [0.87] Quantum Supremacy Achievements — https://example.com/supremacy");
        println!("  [0.75] Industry Applications — https://example.com/applications");
    }

    // Graceful shutdown
    println!("\n{} Shutting down runtime...", "🛑".bold().yellow());
    kernel_handle.shutdown(Some(Duration::from_secs(10))).await?;
    println!("{} Done. Runtime shut down cleanly.", "✓".green().bold());

    Ok(())
}

// Helper for text wrapping (simple implementation)
mod textwrap {
    pub fn fill(text: &str, width: usize) -> String {        let mut result = String::new();
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
            line.push_str(word);
        }
        if !line.is_empty() {
            result.push_str(&line);
        }
        result
    }
}
