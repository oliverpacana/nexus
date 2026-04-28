use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, Args};
use colored::control::SHOULD_COLORIZE;
use colored::Colorize;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;

use crate::config::NexusCliConfig;
use crate::commands::{up, run, status, tui, mem, tool};

/// ⚡ Nexus — AI Agent Runtime
const BANNER: &str = r#"
 ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
 ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
 ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
 ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
 ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
                    AI Agent Operating System
"#;

#[derive(Parser, Debug)]
#[command(
    name = "nexus",
    about = "⚡ Nexus — AI Agent Runtime",
    version = env!("CARGO_PKG_VERSION"),
    long_about = None,
    disable_help_subcommand = true
)]
pub struct NexusCli {
    /// Path to custom configuration file (overrides default lookup order)
    #[arg(long, global = true, value_hint = clap::ValueHint::FilePath)]
    pub config_file: Option<PathBuf>,

    /// Override log level for this invocation (e.g., debug, info, warn, error)
    #[arg(long, global = true)]
    pub log_level: Option<String>,

    /// Disable colored output for terminals/pipes that don't support ANSI
    #[arg(long, global = true)]
    pub no_color: bool,

    #[command(subcommand)]
    pub command: NexusCommand,
}

#[derive(Subcommand, Debug)]
pub enum NexusCommand {
    /// Start the Nexus runtime and keep it running until interrupted    Up(UpArgs),
    /// Execute a workflow definition file
    Run(RunArgs),
    /// Show current runtime status, agent list, and cost tracking
    Status(StatusArgs),
    /// Launch the live terminal dashboard for real-time monitoring
    Tui(TuiArgs),
    /// Inspect, search, and manage agent memory across tiers
    Mem(MemArgs),
    /// Manage WASM tools in the registry
    Tool(ToolArgs),
}

#[derive(Args, Debug)]
pub struct UpArgs {}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Path to workflow definition (TOML or YAML)
    #[arg(value_hint = clap::ValueHint::FilePath)]
    pub workflow_path: PathBuf,

    /// Runtime variables in key=value format (e.g., --var topic=rust --var depth=3)
    #[arg(long, value_parser = parse_var_kv)]
    pub var: Vec<(String, serde_json::Value)>,
}

#[derive(Args, Debug)]
pub struct StatusArgs {}

#[derive(Args, Debug)]
pub struct TuiArgs {}

#[derive(Args, Debug)]
pub struct MemArgs {
    #[command(subcommand)]
    pub action: MemAction,
}

#[derive(Subcommand, Debug)]
pub enum MemAction {
    /// Inspect memory contents for a specific agent
    Inspect {
        #[arg(long)]
        agent: String,
        #[arg(long, default_value = "all", value_parser = parse_tier)]
        tier: String,
    },
    /// Semantic search across L3 vector memory
    Search {        /// Natural language query string
        query: String,
        #[arg(long, default_value = "0.7")]
        min_similarity: f32,
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Clear memory for a specific agent
    Clear {
        #[arg(long)]
        agent: String,
        #[arg(long)]
        tier: Option<String>,
    },
}

#[derive(Args, Debug)]
pub struct ToolArgs {
    #[command(subcommand)]
    pub action: ToolAction,
}

#[derive(Subcommand, Debug)]
pub enum ToolAction {
    /// List installed tools and their execution statistics
    List,
    /// Install a tool from a compiled .wasm file
    Install {
        #[arg(value_hint = clap::ValueHint::FilePath)]
        path: PathBuf,
    },
    /// Uninstall a tool by name
    Remove { name: String },
    /// Hot-reload a tool from disk without restarting the runtime
    Reload { name: String },
}

// =============================================================================
// Main Entry Point
// =============================================================================

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("\n{} Error: {}\n", "❌".bold().red(), e.to_string().bold());
        process::exit(1);
    }
}

async fn run() -> Result<()> {    let cli = NexusCli::parse();

    // Apply color override before any colored output
    if cli.no_color {
        SHOULD_COLORIZE.set_override(false);
    }

    println!("{}", BANNER.cyan().bold());

    // Load configuration with CLI overrides
    let mut config = crate::config::load_config(cli.config_file.as_deref())
        .context("Failed to load Nexus configuration")?;

    if let Some(level) = cli.log_level {
        config.observability.log_level = level;
    }

    // Validate provider configuration for commands that need LLM access
    let needs_providers = matches!(
        &cli.command,
        NexusCommand::Run(_) | NexusCommand::Up(_) | NexusCommand::Tui(_) | NexusCommand::Status(_)
    );
    if needs_providers {
        check_api_keys(&config)?;
    }

    // Dispatch to appropriate command handler
    let result = match cli.command {
        NexusCommand::Up(_) => up::UpCommand {}.execute(config).await,
        NexusCommand::Run(args) => {
            run::RunCommand {
                workflow_path: args.workflow_path,
                var: args.var,
            }
            .execute(config)
            .await
        }
        NexusCommand::Status(_) => status::StatusCommand {}.execute(config).await,
        NexusCommand::Tui(_) => tui::TuiCommand {}.execute(config).await,
        NexusCommand::Mem(args) => {
            mem::MemCommand {
                action: map_mem_action(args.action),
            }
            .execute(config)
            .await
        }
        NexusCommand::Tool(args) => {
            tool::ToolCommand {
                action: map_tool_action(args.action),
            }            .execute(config)
            .await
        }
    };

    result.map_err(|e| e.context("Command execution failed"))
}

// =============================================================================
// Helpers
// =============================================================================

/// Parses a `key=value` string into a typed tuple. Accepts JSON values.
fn parse_var_kv(s: &str) -> Result<(String, serde_json::Value), String> {
    let (k, v) = s.split_once('=').unwrap_or((s, "true"));
    let value = serde_json::from_str(v)
        .unwrap_or_else(|_| serde_json::Value::String(v.to_string()));
    Ok((k.to_string(), value))
}

/// Parses tier string for CLI validation
fn parse_tier(s: &str) -> Result<String, String> {
    match s.to_lowercase().as_str() {
        "l1" | "l2" | "l3" | "l4" | "all" => Ok(s.to_string()),
        _ => Err(format!("Invalid tier '{}'. Expected: l1, l2, l3, l4, or all", s)),
    }
}

/// Maps inline MemAction to command module's MemAction
fn map_mem_action(action: MemAction) -> mem::MemAction {
    match action {
        MemAction::Inspect { agent, tier } => mem::MemAction::Inspect { agent, tier },
        MemAction::Search { query, min_similarity, limit } => mem::MemAction::Search {
            query,
            min_similarity,
            limit,
        },
        MemAction::Clear { agent, tier } => mem::MemAction::Clear { agent, tier },
    }
}

/// Maps inline ToolAction to command module's ToolAction
fn map_tool_action(action: ToolAction) -> tool::ToolAction {
    match action {
        ToolAction::List => tool::ToolAction::List,
        ToolAction::Install { path } => tool::ToolAction::Install { path },
        ToolAction::Remove { name } => tool::ToolAction::Remove { name },
        ToolAction::Reload { name } => tool::ToolAction::Reload { name },
    }
}
/// Validates that at least one AI provider is configured when needed.
fn check_api_keys(config: &NexusCliConfig) -> Result<()> {
    let has_openai = config
        .providers
        .openai
        .as_ref()
        .map_or(false, |p| !p.api_key.is_empty());
    let has_anthropic = config
        .providers
        .anthropic
        .as_ref()
        .map_or(false, |p| !p.api_key.is_empty());
    let has_groq = config
        .providers
        .groq
        .as_ref()
        .map_or(false, |p| !p.api_key.is_empty());
    let has_local = config.providers.local.is_some();

    if !has_openai && !has_anthropic && !has_groq && !has_local {
        bail!(
            "No AI providers configured. At least one provider requires a valid API key or endpoint.\n\
             Quick start options:\n\
               1. Set environment variable: export OPENAI_API_KEY='sk-...'\n\
               2. Use local Ollama: configure providers.local.base_url in nexus.toml\n\
               3. Run `nexus status` to verify configuration"
        );
    }

    if has_openai && !has_anthropic && !has_local {
        println!("{} Using OpenAI as default LLM provider.", "ℹ️".bold().blue());
    } else if has_anthropic && !has_openai && !has_local {
        println!("{} Using Anthropic as default LLM provider.", "ℹ️".bold().blue());
    }

    Ok(())
}
