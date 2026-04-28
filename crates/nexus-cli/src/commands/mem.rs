// crates/nexus-cli/src/commands/mem.rs

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use nexus_proto::memory::{MemoryScope, MemoryTier, SemanticSearchQuery};
use serde_json::Value;

use crate::config::NexusCliConfig;
use crate::runtime;

/// Inspect and manage agent memory across tiers.
#[derive(Parser, Debug, Clone)]
pub struct MemCommand {
    #[clap(subcommand)]
    action: MemAction,
}

#[derive(Subcommand, Debug, Clone)]
enum MemAction {
    /// Inspect memory contents for an agent.
    Inspect {
        #[clap(long)]
        agent: String,
        #[clap(long, default_value = "all", value_parser = parse_tier)]
        tier: MemoryTier,
    },
    /// Semantic search across L3 memory.
    Search {
        query: String,
        #[clap(long, default_value = "0.7")]
        min_similarity: f32,
        #[clap(long, default_value = "10")]
        limit: usize,
    },
    /// Clear memory for an agent.
    Clear {
        #[clap(long)]
        agent: String,
        #[clap(long)]
        tier: Option<String>,
    },
}

fn parse_tier(s: &str) -> Result<MemoryTier> {
    match s.to_lowercase().as_str() {
        "l1" | "working" => Ok(MemoryTier::Working),
        "l2" | "episodic" => Ok(MemoryTier::Episodic),        "l3" | "semantic" => Ok(MemoryTier::Semantic),
        "l4" | "procedural" => Ok(MemoryTier::Procedural),
        "all" => Ok(MemoryTier::Semantic), // Fallback for inspect
        _ => bail!("Invalid tier. Use: l1, l2, l3, l4, or all"),
    }
}

impl MemCommand {
    pub async fn execute(self, config: NexusCliConfig) -> Result<()> {
        let runtime = runtime::start(config).await.context("Runtime bootstrap failed")?;
        let mem = runtime.memory();

        match &self.action {
            MemAction::Inspect { agent, tier } => {
                let agent_id = parse_agent_id(agent)?;
                println!("{} Inspecting memory for agent {}", "🔍".bold().cyan(), agent_id.to_string().bold().dimmed());

                if *tier == MemoryTier::Working || *tier == MemoryTier::Semantic { // Show L1 always
                    let working = mem.get_working(agent_id);
                    let keys = working.all_keys().await;
                    println!("{} L1 Working Memory ({} keys):", "   ├─".dimmed(), keys.len().bold().green());
                    for key in &keys {
                        if let Some(val) = working.get(key).await {
                            let display = if let Some(s) = val.value.as_str() {
                                format!("{}", &s[..s.len().min(80)])
                            } else {
                                serde_json::to_string_pretty(&val.value).unwrap_or_default()
                            };
                            println!("{} {}: {}", "   │  ├─".dimmed(), key.bold(), display.dimmed());
                        }
                    }
                }
                if *tier == MemoryTier::Procedural || *tier == MemoryTier::Semantic {
                    let (ent_count, rel_count) = mem.procedural_counts()
                        .context("Failed to fetch procedural counts")?;
                    println!("{} L4 Procedural Graph: {} entities, {} relations", 
                        "   └─".dimmed(), ent_count.to_string().bold().magenta(), rel_count.to_string().bold().magenta());
                }
            }

            MemAction::Search { query, min_similarity, limit } => {
                println!("{} Searching semantic memory for '{}'...", "🔍".bold().cyan(), query.bold().yellow());
                let search_query = SemanticSearchQuery {
                    query_embedding: None,
                    query_text: Some(query.clone()),
                    top_k: *limit,
                    min_similarity: *min_similarity,
                    scope_filter: None,
                    owner_filter: None,
                    tag_filter: vec![],                };

                // Use a dummy agent ID for search context if not specified
                let agent_id = nexus_proto::agent::AgentId::nil();
                let results = mem.semantic_search(agent_id, search_query).await
                    .context("Semantic search failed")?;

                if results.is_empty() {
                    println!("{} No results found.", "   └─".dimmed().italic());
                } else {
                    println!("{} Found {} results:", "   ├─".dimmed(), results.len().bold().green());
                    for (i, res) in results.iter().enumerate() {
                        println!("{} [{}] ({:.3}) {}", 
                            "   │  ├─".dimmed(), 
                            i.to_string().bold().cyan(), 
                            res.similarity.bold().magenta(),
                            serde_json::to_string_pretty(&res.entry.value).unwrap_or_default().dimmed());
                    }
                }
            }

            MemAction::Clear { agent, tier } => {
                let agent_id = parse_agent_id(agent)?;
                let target = tier.as_deref().unwrap_or("all");
                println!("{} Clearing {} memory for agent {}", "🗑️".bold().red(), target.bold().yellow(), agent_id.to_string().bold().dimmed());
                
                if target.eq_ignore_ascii_case("all") || target.eq_ignore_ascii_case("l1") {
                    let working = mem.get_working(agent_id);
                    let count = working.len().await;
                    working.clear().await;
                    println!("{} L1 Working Memory: {} entries cleared", "   ├─".dimmed(), count.to_string().bold().green());
                }
                if target.eq_ignore_ascii_case("all") || target.eq_ignore_ascii_case("l4") {
                    mem.procedural_clear().await.context("Failed to clear procedural memory")?;
                    println!("{} L4 Procedural Graph: cleared", "   └─".dimmed());
                }
            }
        }

        runtime.shutdown().await?;
        Ok(())
    }
}

fn parse_agent_id(s: &str) -> Result<nexus_proto::agent::AgentId> {
    let uuid = uuid::Uuid::parse_str(s)
        .context("Agent ID must be a valid UUID")?;
    Ok(nexus_proto::agent::AgentId::new_from_uuid(uuid))
}
