// crates/nexus-flow/src/loader.rs

use std::collections::HashMap;
use std::path::Path;
use std::fs;

use serde::Deserialize;
use serde_json::Value;

use nexus_proto::agent::AgentKind;
use nexus_proto::workflow::{
    RetryPolicy, StepDefinition, StepId, StepKind, TransformKind, WorkflowDefinition,
};

use crate::dsl::{StepBuilder, WorkflowBuilder};
use crate::error::FlowError;

/// Raw TOML/YAML representation of a workflow definition.
#[derive(Deserialize)]
struct RawWorkflow {
    workflow: RawWorkflowMeta,
    steps: Vec<RawStep>,
}

#[derive(Deserialize)]
struct RawWorkflowMeta {
    name: String,
    version: String,
    description: Option<String>,
    #[serde(default)]
    variables: HashMap<String, Value>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct RawStep {
    id: String,
    #[serde(rename = "type")]
    step_type: String,
    description: Option<String>,
    #[serde(default)]
    next: Vec<String>,
    timeout_ms: Option<u64>,
    #[serde(default)]
    max_attempts: Option<u32>,
    #[serde(default)]
    backoff_ms: Option<u64>,
    #[serde(default)]
    backoff_multiplier: Option<f32>,    #[serde(default)]
    max_backoff_ms: Option<u64>,

    // Agent fields
    agent_kind: Option<String>,
    prompt_template: Option<String>,
    output_key: Option<String>,

    // Tool fields
    tool_name: Option<String>,
    arguments_template: Option<Value>,

    // Conditional fields
    condition_prompt: Option<String>,
    output_schema: Option<Value>,
    branches: Option<HashMap<String, String>>,
    default_branch: Option<String>,

    // Parallel fields
    parallel_steps: Option<Vec<String>>,
    join_output_key: Option<String>,

    // Wait fields
    duration_ms: Option<u64>,

    // End fields
    success: Option<bool>,

    // Transform fields
    input_key: Option<String>,
    transform: Option<String>,
}

/// Loads a workflow definition from a TOML string.
pub fn load_from_toml(toml_str: &str) -> Result<WorkflowDefinition, FlowError> {
    let raw: RawWorkflow = toml::from_str(toml_str)
        .map_err(|e| FlowError::Load(format!("TOML parse error: {}", e)))?;
    build_from_raw(raw)
}

/// Loads a workflow definition from a YAML string.
pub fn load_from_yaml(yaml_str: &str) -> Result<WorkflowDefinition, FlowError> {
    let raw: RawWorkflow = serde_yaml::from_str(yaml_str)
        .map_err(|e| FlowError::Load(format!("YAML parse error: {}", e)))?;
    build_from_raw(raw)
}

/// Loads a workflow definition from a file.
/// Detects format by extension (`.toml` or `.yaml`/`.yml`).
pub fn load_from_file(path: &Path) -> Result<WorkflowDefinition, FlowError> {    let content = fs::read_to_string(path)
        .map_err(|e| FlowError::Load(format!("failed to read file {}: {}", path.display(), e)))?;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "toml" => load_from_toml(&content),
        "yaml" | "yml" => load_from_yaml(&content),
        _ => Err(FlowError::Load(format!("unsupported file extension: {}", ext))),
    }
}

/// Converts raw parsed data into a validated `WorkflowDefinition`.
fn build_from_raw(raw: RawWorkflow) -> Result<WorkflowDefinition, FlowError> {
    let mut builder = WorkflowBuilder::new(&raw.workflow.name)
        .version(&raw.workflow.version)
        .description(raw.workflow.description.unwrap_or_default());

    for (key, val) in raw.workflow.variables {
        builder = builder.var(key, val);
    }

    for tag in raw.workflow.tags {
        builder = builder.tag(tag);
    }

    for raw_step in raw.steps {
        let step = convert_step(raw_step)?;
        builder = builder.step(step);
    }

    builder.build()
}

/// Converts a `RawStep` into a `StepDefinition` using the DSL builders.
fn convert_step(raw: RawStep) -> Result<StepDefinition, FlowError> {
    let retry = RetryPolicy {
        max_attempts: raw.max_attempts.unwrap_or(3),
        backoff_ms: raw.backoff_ms.unwrap_or(1000),
        backoff_multiplier: raw.backoff_multiplier.unwrap_or(2.0),
        max_backoff_ms: raw.max_backoff_ms.unwrap_or(30000),
    };

    let step_def = match raw.step_type.as_str() {
        "agent" => {
            let kind = match raw.agent_kind.as_deref() {
                Some("research") => AgentKind::Research,
                Some("writing") => AgentKind::Writing,
                Some("code_review") => AgentKind::CodeReview,
                Some("analysis") => AgentKind::Analysis,
                Some("planning") => AgentKind::Planning,                Some(other) => AgentKind::Custom(other.into()),
                None => AgentKind::Custom("agent".into()),
            };

            let mut b = StepBuilder::agent(&raw.id)
                .kind(kind)
                .prompt(raw.prompt_template.unwrap_or_default())
                .output(raw.output_key.unwrap_or_else(|| "output".into()))
                .retry(retry);

            if let Some(ms) = raw.timeout_ms {
                b = b.timeout_ms(ms);
            }
            if let Some(desc) = raw.description {
                b = b.description(desc);
            }
            for next in raw.next {
                b = b.then(next);
            }
            b.build()?
        }
        "tool" => {
            let mut b = StepBuilder::tool(&raw.id)
                .name(raw.tool_name.unwrap_or_default())
                .arguments(raw.arguments_template.unwrap_or(Value::Object(Default::default())))
                .output(raw.output_key.unwrap_or_else(|| "output".into()))
                .retry(retry);

            if let Some(ms) = raw.timeout_ms {
                b = b.timeout_ms(ms);
            }
            if let Some(desc) = raw.description {
                b = b.description(desc);
            }
            for next in raw.next {
                b = b.then(next);
            }
            b.build()?
        }
        "conditional" => {
            let mut b = StepBuilder::conditional(&raw.id)
                .prompt(raw.condition_prompt.unwrap_or_default())
                .schema(raw.output_schema.unwrap_or(Value::Object(Default::default())))
                .retry(retry);

            if let Some(default) = raw.default_branch {
                b = b.default(default);
            }
            if let Some(branches) = raw.branches {
                for (cond, target) in branches {                    b = b.branch(cond, target);
                }
            }
            if let Some(ms) = raw.timeout_ms {
                b = b.timeout_ms(ms);
            }
            if let Some(desc) = raw.description {
                b = b.description(desc);
            }
            for next in raw.next {
                b = b.then(next);
            }
            b.build()?
        }
        "parallel" => {
            let mut b = StepBuilder::parallel(&raw.id)
                .join_key(raw.join_output_key.unwrap_or_else(|| "parallel_output".into()))
                .retry(retry);

            if let Some(steps) = raw.parallel_steps {
                for s in steps {
                    b = b.add(s);
                }
            }
            if let Some(ms) = raw.timeout_ms {
                b = b.timeout_ms(ms);
            }
            if let Some(desc) = raw.description {
                b = b.description(desc);
            }
            for next in raw.next {
                b = b.then(next);
            }
            b.build()?
        }
        "transform" => {
            let transform = match raw.transform.as_deref().unwrap_or("serialize_json") {
                "json_extract" => TransformKind::JsonExtract {
                    path: raw.input_key.clone().unwrap_or_else(|| "$".into()),
                },
                "text_summarize" => TransformKind::TextSummarize,
                "text_join" => TransformKind::TextJoin { separator: ", ".into() },
                "parse_json" => TransformKind::ParseJson,
                "serialize_json" => TransformKind::SerializeJson,
                other => TransformKind::Custom { code: other.into() },
            };

            let mut b = StepBuilder::transform(&raw.id)
                .input(raw.input_key.unwrap_or_else(|| "input".into()))
                .transform(transform)                .output(raw.output_key.unwrap_or_else(|| "output".into()));

            if let Some(ms) = raw.timeout_ms {
                b = b.timeout_ms(ms);
            }
            if let Some(desc) = raw.description {
                b = b.description(desc);
            }
            for next in raw.next {
                b = b.then(next);
            }
            b.build()?
        }
        "wait" => {
            let duration = raw.duration_ms.unwrap_or(0);
            let mut s = StepBuilder::wait(&raw.id, std::time::Duration::from_millis(duration));
            for next in raw.next {
                s.next.push(StepId::new(next));
            }
            if let Some(ms) = raw.timeout_ms {
                s.timeout_ms = Some(ms);
            }
            if let Some(desc) = raw.description {
                s.description = Some(desc);
            }
            s
        }
        "end" => {
            let success = raw.success.unwrap_or(true);
            let mut s = StepBuilder::end(&raw.id, success);
            if let Some(desc) = raw.description {
                s.description = Some(desc);
            }
            s
        }
        other => {
            return Err(FlowError::Load(format!("unknown step type: {}", other)));
        }
    };

    Ok(step_def)
}

// Dependencies note: This module assumes `toml = "0.8"` and `serde_yaml = "0.9"`
// are available in the crate's Cargo.toml. They are standard for TOML/YAML parsing.
