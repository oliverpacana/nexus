// crates/nexus-cli/src/commands/tool.rs

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;

use crate::config::NexusCliConfig;
use crate::runtime;

/// Manage WASM tools in the registry.
#[derive(Parser, Debug, Clone)]
pub struct ToolCommand {
    #[clap(subcommand)]
    action: ToolAction,
}

#[derive(Subcommand, Debug, Clone)]
enum ToolAction {
    /// List installed tools and their usage stats.
    List,
    /// Install a tool from a .wasm file.
    Install { path: PathBuf },
    /// Uninstall a tool by name.
    Remove { name: String },
    /// Hot-reload a tool from disk.
    Reload { name: String },
}

impl ToolCommand {
    pub async fn execute(self, config: NexusCliConfig) -> Result<()> {
        let runtime = runtime::start(config).await.context("Runtime bootstrap failed")?;
        let engine = runtime.tools();

        match &self.action {
            ToolAction::List => {
                println!("{}", "🔧 Installed Tools".bold().cyan());
                let tools = engine.list_tools();
                if tools.is_empty() {
                    println!("{} {}", "   └─".dimmed(), "No tools registered.".dimmed());
                } else {
                    println!("{}", format!("{:<25} {:<10} {:<8} {:<10} {:<12} {:<10}", 
                        "Name".bold().underline(), 
                        "Version".bold().underline(), 
                        "Calls".bold().underline(), 
                        "Success%".bold().underline(),
                        "Avg Latency".bold().underline(),
                        "Errors".bold().underline()).dimmed());
                    println!("{}", "─".repeat(80).dimmed());
                    
                    for t in &tools {
                        println!("{} {:<25} {:<10} {:<8} {:<10} {:<12} {:<10}", 
                            "   ├─".dimmed(),
                            t.name.bold(),
                            t.version.dimmed(),
                            t.call_count.to_string().bold().green(),
                            format!("{:.1}%", t.success_rate * 100.0).bold().cyan(),
                            format!("{:.2}ms", t.avg_execution_ms).dimmed(),
                            t.error_count.to_string().bold().red());
                    }
                }
            }

            ToolAction::Install { path } => {
                println!("{} Installing tool from {}...", "📦".bold().cyan(), path.display().bold().yellow());
                let tool_id = engine.install_tool(path)
                    .await
                    .context("Failed to install tool")?;
                println!("{} Tool installed successfully: {}", "✅".bold().green(), tool_id.to_string().bold().magenta());
            }

            ToolAction::Remove { name } => {
                println!("{} Removing tool '{}'...", "🗑️".bold().red(), name.bold().yellow());
                engine.registry().uninstall(name, None)
                    .await
                    .context("Failed to uninstall tool")?;
                println!("{} Tool '{}' removed.", "✅".bold().green(), name.bold());
            }

            ToolAction::Reload { name } => {
                println!("{} Reloading tool '{}'...", "🔄".bold().blue(), name.bold().yellow());
                engine.reload_tool(name)
                    .await
                    .context("Failed to reload tool")?;
                println!("{} Tool '{}' reloaded.", "✅".bold().green(), name.bold());
            }
        }

        runtime.shutdown().await?;
        Ok(())
    }
}
