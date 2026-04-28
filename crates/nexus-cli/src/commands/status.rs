// crates/nexus-cli/src/commands/status.rs

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use nexus_proto::agent::{AgentStatus, AgentPriority};

use crate::config::NexusCliConfig;
use crate::runtime;

/// Shows the current runtime status: agents, memory, cost, mesh.
#[derive(Parser, Debug, Clone)]
pub struct StatusCommand {}

impl StatusCommand {
    pub async fn execute(self, config: NexusCliConfig) -> Result<()> {
        println!("{}", "📊 Nexus Runtime Status".bold().cyan());
        println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed());

        let runtime = runtime::start(config).await.context("Failed to start runtime")?;

        // 1. Running Agents Table
        let agents = runtime.kernel.list_agents();
        println!("\n{} Active Agents ({})", "🤖".bold().cyan(), agents.len().bold().green());
        if agents.is_empty() {
            println!("{} {}", "   └─".dimmed(), "No agents currently running.".dimmed());
        } else {
            // Header
            println!("{}", format!("{:<38} {:<15} {:<12} {:<12}", 
                "Name".bold().underline(), 
                "Kind".bold().underline(), 
                "Priority".bold().underline(), 
                "Status".bold().underline()).dimmed());
            println!("{}", "─".repeat(80).dimmed());
            
            for agent in &agents {
                let status_color = match &agent.status {
                    AgentStatus::Running { .. } => agent.status.to_string().green(),
                    AgentStatus::Pending { .. } => agent.status.to_string().yellow(),
                    AgentStatus::Completed { success: true, .. } | AgentStatus::Suspended { .. } => agent.status.to_string().cyan(),
                    _ => agent.status.to_string().red(),
                };
                let priority_color = match agent.priority {
                    AgentPriority::Critical | AgentPriority::High => format!("{:?}", agent.priority).yellow(),
                    AgentPriority::Normal => format!("{:?}", agent.priority).green(),
                    _ => format!("{:?}", agent.priority).dimmed(),
                };
                println!("{} {:<38} {:<15} {:<12} {}", 
                    "   ├─".dimmed(), 
                    agent.name.bold(), 
                    format!("{:?}", agent.kind),
                    priority_color,
                    status_color);
            }
        }

        // 2. Memory Usage
        println!("\n{} Memory Subsystem", "💾".bold().cyan());
        let mem_stats = runtime.memory.working_agent_count();
        println!("{} L1 Working Memory agents: {}", "   ├─".dimmed(), mem_stats.to_string().bold());
        println!("{} L2 Episodic DB: {}", "   ├─".dimmed(), runtime.config.memory.episodic_db_path.dimmed());
        println!("{} L3 Semantic Index: {}", "   ├─".dimmed(), runtime.config.memory.semantic_index_path.dimmed());
        println!("{} L4 Procedural Graph: {}", "   └─".dimmed(), runtime.config.memory.procedural_db_path.dimmed());

        // 3. Cost Today
        let total_cost = runtime.obs.ledger().total_spent().await.context("Failed to fetch cost ledger")?;
        println!("\n{} Cost Tracking", "💰".bold().cyan());
        println!("{} Total LLM Spend: {}", "   └─".dimmed(), format!("${:.6}", total_cost).bold().magenta());

        // 4. Mesh Status
        if let Some(mesh) = runtime.mesh.as_ref() {
            println!("\n{} P2P Mesh", "🌐".bold().cyan());
            println!("{} Peers connected: {}", "   └─".dimmed(), mesh.peer_count().to_string().bold().blue());
        } else {
            println!("\n{} P2P Mesh: {}", "🌐".bold().cyan(), "Disabled".dimmed().italic());
        }

        println!("\n{}", "─────────────────────────────────────────".dimmed());
        runtime.shutdown().await?;
        Ok(())
    }
}
