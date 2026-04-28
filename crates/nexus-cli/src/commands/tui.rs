// crates/nexus-cli/src/commands/tui.rs

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;

use crate::config::NexusCliConfig;
use crate::runtime;

/// Launches the interactive terminal dashboard.
#[derive(Parser, Debug, Clone)]
pub struct TuiCommand {}

impl TuiCommand {
    pub async fn execute(self, config: NexusCliConfig) -> Result<()> {
        println!("{}", "🖥️  Launching Nexus TUI Dashboard...".bold().cyan());

        let runtime = runtime::start(config.clone()).await.context("Failed to start runtime")?;

        let tui_config = nexus_obs::tui::TuiConfig {
            refresh_rate_ms: config.observability.tui_refresh_rate_ms,
            kernel_handle: std::sync::Arc::clone(&runtime.kernel),
            cost_ledger: std::sync::Arc::clone(&runtime.obs.ledger()),
        };

        let mut tui_app = nexus_obs::tui::TuiApp::new(tui_config)
            .context("Failed to initialize TUI application")?;

        // The TUI app handles terminal setup and teardown via Drop
        // If it errors, we catch it and attempt shutdown anyway
        if let Err(e) = tui_app.run().await {
            eprintln!("{} TUI crashed: {}", "❌".bold().red(), e.to_string().bold().red());
        }

        runtime.shutdown().await.context("Runtime shutdown after TUI failed")?;
        println!("{}", "👋 TUI session ended.".dimmed());
        Ok(())
    }
}
