// crates/nexus-cli/src/commands/up.rs

use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use tokio::signal;
use tokio::time::sleep;

use crate::config::NexusCliConfig;
use crate::runtime;

/// Starts the Nexus runtime and keeps it running until interrupted.
#[derive(Parser, Debug, Clone)]
pub struct UpCommand {}

impl UpCommand {
    pub async fn execute(self, config: NexusCliConfig) -> Result<()> {
        println!();
        println!("{}", "⚡ NEXUS v0.1.0 — AI Agent Runtime".bold().cyan());
        println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed());
        println!("{} {}", "🚀".bold().green(), "Starting subsystems...".bold());

        let runtime = runtime::start(config)
            .await
            .context("Failed to bootstrap Nexus runtime")?;

        println!("{} {}", "✅".bold().green(), "All subsystems initialized successfully.");
        println!("{} Kernel:        max_agents={}", "   ├─".dimmed(), runtime.config.scheduler.max_agents);
        println!("{} Memory:        tiers=[L1, L2, L3, L4] active={}", "   ├─".dimmed(), runtime.memory.working_agent_count());
        
        let tool_count = runtime.tools.list_tools().len();
        println!("{} Tools:        {} registered", "   ├─".dimmed(), tool_count);
        println!("{} Router:        providers=[{}]", "   ├─".dimmed(), runtime.router.providers.available_ids().iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", "));
        println!("{} Observability: log={}, cost_db={}", "   ├─".dimmed(), runtime.config.observability.log_level, runtime.config.observability.cost_ledger_db_path);
        println!("{} {}", "   └─".dimmed(), "Runtime ready. Press Ctrl+C to shutdown.".bold().yellow());
        println!();

        // Wait for shutdown signal (Ctrl+C is already registered in runtime::start, but we block here)
        loop {
            sleep(Duration::from_secs(1)).await;
            if *runtime.shutdown_tx.borrow() {
                break;
            }
        }

        println!("\n{} {}", "🛑".bold().red(), "Shutting down gracefully...".bold());
        runtime.shutdown().await.context("Graceful shutdown failed")?;
        println!("{} {}", "✅".bold().green(), "Nexus runtime stopped cleanly.".bold());
        Ok(())
    }
}
