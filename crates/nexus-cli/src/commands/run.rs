// crates/nexus-cli/src/commands/run.rs

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use nexus_flow::loader::load_from_file;
use nexus_proto::workflow::WorkflowDefinition;
use serde_json::Value;
use tokio::sync::watch;

use crate::config::NexusCliConfig;
use crate::runtime;

/// Executes a workflow definition file.
#[derive(Parser, Debug, Clone)]
pub struct RunCommand {
    /// Path to workflow definition (TOML or YAML)
    pub workflow_path: PathBuf,

    /// Runtime variables in key=value format (e.g. --var topic=rust --var depth=3)
    #[clap(long, value_parser = parse_var_kv)]
    pub var: Vec<(String, Value)>,
}

fn parse_var_kv(s: &str) -> Result<(String, Value)> {
    let (k, v) = s.split_once('=').unwrap_or((s, "true"));
    let value = serde_json::from_str(v)
        .unwrap_or(Value::String(v.to_string()));
    Ok((k.to_string(), value))
}

impl RunCommand {
    pub async fn execute(self, config: NexusCliConfig) -> Result<()> {
        println!("{} Loading workflow from {}...", "📖".bold().cyan(), self.workflow_path.display());
        
        let definition: WorkflowDefinition = load_from_file(&self.workflow_path)
            .context("Failed to parse workflow definition")?;

        println!("{} Workflow '{}' loaded ({} steps, v{})", 
            "✅".bold().green(), definition.name.bold(), definition.steps.len(), definition.version);

        // Merge initial variables
        let mut vars: HashMap<String, Value> = definition.variables.clone();
        for (k, v) in self.var {
            vars.insert(k, v);        }

        println!("{} Starting runtime for workflow execution...", "⚡".bold().cyan());
        let runtime = runtime::start(config).await.context("Runtime bootstrap failed")?;

        // Setup progress bar
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
            .template("{spinner:.cyan} {msg}")
            .unwrap());
        pb.set_message(format!("Executing workflow '{}'...", definition.name));
        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        // Create executor
        let flow_cfg = nexus_flow::executor::ExecutorConfig::default();
        let flow_executor = nexus_flow::executor::WorkflowExecutor::new(
            flow_cfg,
            std::sync::Arc::clone(&runtime.kernel),
            std::sync::Arc::clone(&runtime.router),
            std::sync::Arc::clone(&runtime.tools),
            std::sync::Arc::clone(&runtime.memory),
            std::sync::Arc::clone(&runtime.obs.ledger()),
        );

        let start = Instant::now();
        let (_, mut shutdown_rx) = watch::channel(false);

        // Execute workflow
        let result = flow_executor.run(
            definition,
            vars,
            None, // No resume run ID
            shutdown_rx.clone(),
        ).await;

        pb.finish_and_clear();

        match result {
            Ok(run) => {
                let duration = start.elapsed();
                println!("{} Workflow '{}' completed in {:.2}s", "✅".bold().green(), run.workflow_id.to_string().bold().dimmed(), duration.as_secs_f32());
                println!("{} Final Status: {}", "   └─".dimmed(), format!("{:?}", run.status).bold().green());
                
                // Print cost summary if any
                let total_cost = runtime.obs.ledger().memory_ledger().total_spent().await;
                if total_cost > 0.0 {
                    println!("{} Total Cost: ${:.6}", "   └─".dimmed(), total_cost);
                }
            }            Err(e) => {
                println!("{} Workflow execution failed: {}", "❌".bold().red(), e.to_string().bold().red());
                println!("{} Check logs and checkpoint data for details.", "   └─".dimmed().yellow());
            }
        }

        // Graceful shutdown after run
        runtime.shutdown().await?;
        Ok(())
    }
}
