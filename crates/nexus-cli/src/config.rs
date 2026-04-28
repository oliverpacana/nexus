use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use config::{Config, Environment, File, FileFormat};
use colored::Colorize;
use dirs::config_dir;
use serde::{Deserialize, Serialize};

use nexus_mem::MemoryConfig;
use nexus_tools::ToolEngineConfig;
use nexus_router::RouterConfig;
use nexus_router::providers::{
    anthropic::AnthropicConfig, groq::GroqConfig, local::LocalConfig, openai::OpenAIConfig,
};
use nexus_mesh::MeshNodeConfig;

// =============================================================================
// Configuration Structures
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub data_dir: String,
    pub max_concurrent_workflows: usize,
    pub graceful_shutdown_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_agents: usize,
    pub default_token_capacity: u64,
    pub default_refill_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub openai: Option<OpenAIConfig>,
    #[serde(default)]
    pub anthropic: Option<AnthropicConfig>,
    #[serde(default)]
    pub groq: Option<GroqConfig>,
    #[serde(default)]
    pub local: Option<LocalConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsConfig {
    pub log_level: String,
    pub log_format: String,    pub otlp_endpoint: Option<String>,
    pub cost_ledger_db_path: String,
    pub tui_refresh_rate_ms: u64,
}

/// Fully resolved runtime configuration for the Nexus CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NexusCliConfig {
    pub runtime: RuntimeConfig,
    pub scheduler: SchedulerConfig,
    pub memory: MemoryConfig,
    pub tools: ToolEngineConfig,
    pub router: RouterConfig,
    pub providers: ProvidersConfig,
    pub mesh: MeshNodeConfig,
    pub observability: ObsConfig,
}

// =============================================================================
// Config Loader
// =============================================================================

/// Loads and resolves the complete Nexus configuration from defaults, files, and env vars.
///
/// Resolution order (later overrides earlier):
/// 1. Embedded `nexus.default.toml`
/// 2. Explicit `--config-file` path (if provided)
/// 3. `./nexus.toml`
/// 4. `~/.config/nexus/nexus.toml`
/// 5. `/etc/nexus/nexus.toml`
/// 6. Environment variables prefixed with `NEXUS_` (e.g., `NEXUS_ROUTER_DEFAULT_POLICY`)
pub fn load_config(config_path: Option<&Path>) -> Result<NexusCliConfig> {
    let mut builder = Config::builder();

    // 1. Load embedded default configuration
    let default_toml = include_str!("nexus.default.toml");
    builder = builder.add_source(File::from_str(default_toml, FileFormat::Toml));

    // 2. Layer file-based configurations in priority order
    let file_paths = vec![
        config_path.map(|p| p.to_path_buf()),
        Some(PathBuf::from("nexus.toml")),
        config_dir().map(|d| d.join("nexus").join("nexus.toml")),
        Some(PathBuf::from("/etc/nexus/nexus.toml")),
    ];

    for path in file_paths.into_iter().flatten() {
        if path.exists() {
            builder = builder.add_source(
                File::from(path)                    .format(FileFormat::Toml)
                    .required(true)
            );
        }
    }

    // 3. Layer environment variables
    builder = builder.add_source(
        Environment::with_prefix("NEXUS")
            .separator("_")
            .try_parsing(true),
    );

    // 4 & 5. Build and deserialize
    let config_obj = builder
        .build()
        .context("failed to build configuration from sources")?;

    let mut config: NexusCliConfig = config_obj
        .try_deserialize()
        .context("failed to deserialize configuration into NexusCliConfig")?;

    // 6. Substitute environment variable references in API keys
    substitute_env_keys(&mut config);

    // 7. Validate and warn
    validate_config(&config)?;

    Ok(config)
}

/// Replaces `${VAR_NAME}` or `$VAR_NAME` patterns in API key fields with actual env var values.
fn substitute_env_keys(config: &mut NexusCliConfig) {
    let resolve = |key: &mut String| {
        if key.starts_with('$') || key.contains("${") {
            let var_name = key
                .trim_start_matches('$')
                .trim_start_matches('{')
                .trim_end_matches('}');
            if let Ok(val) = std::env::var(var_name) {
                *key = val;
            }
        }
    };

    if let Some(ref mut p) = config.providers.openai {
        resolve(&mut p.api_key);
    }
    if let Some(ref mut p) = config.providers.anthropic {
        resolve(&mut p.api_key);    }
    if let Some(ref mut p) = config.providers.groq {
        resolve(&mut p.api_key);
    }
    if let Some(ref mut p) = config.providers.local {
        if let Some(ref mut k) = p.api_key {
            resolve(k);
        }
    }
}

/// Validates the resolved configuration and prints warnings for misconfigurations.
fn validate_config(config: &NexusCliConfig) -> Result<()> {
    if let Some(ref p) = config.providers.openai {
        if p.api_key.is_empty() {
            eprintln!(
                "{} OpenAI provider enabled but api_key is empty. Agent LLM calls will fail.",
                "WARNING:".yellow().bold()
            );
        }
    }
    if let Some(ref p) = config.providers.anthropic {
        if p.api_key.is_empty() {
            eprintln!(
                "{} Anthropic provider enabled but api_key is empty.",
                "WARNING:".yellow().bold()
            );
        }
    }

    if config.router.request_timeout_secs == 0 {
        return Err(anyhow::anyhow!("router.request_timeout_secs must be > 0"));
    }
    if config.scheduler.max_agents == 0 {
        return Err(anyhow::anyhow!("scheduler.max_agents must be > 0"));
    }

    Ok(())
}

/// Prints a formatted, human-readable summary of the active configuration.
pub fn print_config_summary(config: &NexusCliConfig) {
    println!("\n{}", "⚡ Nexus Runtime Configuration".bold().cyan());
    println!("{}", "─".repeat(40).dimmed());

    println!("  {}: {}", "Runtime".bold(), config.runtime.data_dir);
    println!("  {}: max_agents={}", "Scheduler".bold(), config.scheduler.max_agents);

    println!("\n  {}", "Providers".bold());
    let mask = |s: &str| if s.is_empty() { "[NONE]".into() else { format!("{}****", &s[..std::cmp::min(s.len(), 4)]) } };    
    if let Some(p) = &config.providers.openai {
        println!("    OpenAI:     base_url={}, api_key={}", p.base_url, mask(&p.api_key));
    }
    if let Some(p) = &config.providers.anthropic {
        println!("    Anthropic:  base_url={}, api_key={}", p.base_url, mask(&p.api_key));
    }
    if let Some(p) = &config.providers.groq {
        println!("    Groq:       base_url={}, api_key={}", p.base_url, mask(&p.api_key));
    }
    if let Some(p) = &config.providers.local {
        println!("    Local:      base_url={}", p.base_url);
    }

    println!("\n  {}: log={}, format={}, cost_db={}", 
        "Observability".bold(), 
        config.observability.log_level,
        config.observability.log_format,
        config.observability.cost_ledger_db_path
    );
    
    println!("  {}: default_policy={}, timeout={}s", 
        "Router".bold(),
        std::mem::variant_name(&config.router.default_policy),
        config.router.request_timeout_secs
    );

    println!("  {}: addr={}, mdns={}", 
        "Mesh".bold(),
        config.mesh.listen_addr,
        config.mesh.mdns_enabled
    );

    println!("{}", "─".repeat(40).dimmed());
    println!("{}\n", "Configuration loaded successfully.".green().bold());
}
