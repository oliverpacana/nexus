// crates/nexus-flow/src/dsl.rs

use std::collections::HashMap;
use std::time::Duration;

use chrono::Utc;
use serde_json::Value;

use nexus_proto::agent::{AgentCapabilities, AgentKind};
use nexus_proto::workflow::{
    RetryPolicy, StepDefinition, StepId, StepKind, TransformKind, WorkflowDefinition, WorkflowId,
    WorkflowRunStatus,
};

use crate::error::FlowError;

/// Fluent builder for constructing `WorkflowDefinition` instances.
pub struct WorkflowBuilder {
    id: WorkflowId,
    name: String,
    description: Option<String>,
    version: String,
    variables: HashMap<String, Value>,
    steps: HashMap<StepId, StepDefinition>,
    entry_step: Option<StepId>,
    tags: Vec<String>,
}

impl WorkflowBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: WorkflowId::new(),
            name: name.into(),
            description: None,
            version: "0.1.0".to_string(),
            variables: HashMap::new(),
            steps: HashMap::new(),
            entry_step: None,
            tags: Vec::new(),
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn version(mut self, v: impl Into<String>) -> Self {
        self.version = v.into();
        self    }

    pub fn var(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.variables.insert(key.into(), value.into());
        self
    }

    pub fn step(mut self, step: StepDefinition) -> Self {
        let id = step.id.clone();
        if self.entry_step.is_none() {
            self.entry_step = Some(id.clone());
        }
        self.steps.insert(id, step);
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Builds and validates the workflow definition.
    /// Returns an error if the DAG is invalid (cycles, unreachable nodes, etc.).
    pub fn build(self) -> Result<WorkflowDefinition, FlowError> {
        let entry_step = self.entry_step.ok_or_else(|| {
            FlowError::InvalidWorkflow("workflow has no steps defined".into())
        })?;

        let definition = WorkflowDefinition {
            id: self.id,
            name: self.name,
            description: self.description,
            version: self.version,
            steps: self.steps,
            entry_step,
            variables: self.variables,
            tags: self.tags,
            created_at: Utc::now(),
        };

        definition.validate().map_err(|e| FlowError::InvalidWorkflow(e.to_string()))?;
        Ok(definition)
    }
}

/// Namespace for creating step builders.
pub struct StepBuilder;

impl StepBuilder {
    pub fn agent(id: impl Into<String>) -> AgentStepBuilder {        AgentStepBuilder::new(id)
    }

    pub fn tool(id: impl Into<String>) -> ToolStepBuilder {
        ToolStepBuilder::new(id)
    }

    pub fn conditional(id: impl Into<String>) -> ConditionalStepBuilder {
        ConditionalStepBuilder::new(id)
    }

    pub fn parallel(id: impl Into<String>) -> ParallelStepBuilder {
        ParallelStepBuilder::new(id)
    }

    pub fn transform(id: impl Into<String>) -> TransformStepBuilder {
        TransformStepBuilder::new(id)
    }

    pub fn wait(id: impl Into<String>, duration: Duration) -> StepDefinition {
        StepDefinition {
            id: StepId::new(id),
            kind: StepKind::Wait {
                duration_ms: duration.as_millis() as u64,
            },
            next: Vec::new(),
            description: None,
            timeout_ms: None,
            retry_policy: RetryPolicy::no_retry(),
            meta HashMap::new(),
        }
    }

    pub fn end(id: impl Into<String>, success: bool) -> StepDefinition {
        StepDefinition {
            id: StepId::new(id),
            kind: StepKind::End { success },
            next: Vec::new(),
            description: None,
            timeout_ms: None,
            retry_policy: RetryPolicy::no_retry(),
            meta HashMap::new(),
        }
    }
}

/// Builder for `StepKind::Agent` steps.
pub struct AgentStepBuilder {
    id: StepId,
    kind: AgentKind,    prompt_template: String,
    capabilities: AgentCapabilities,
    output_key: String,
    next: Vec<StepId>,
    description: Option<String>,
    timeout_ms: Option<u64>,
    retry_policy: RetryPolicy,
    meta HashMap<String, Value>,
}

impl AgentStepBuilder {
    fn new(id: impl Into<String>) -> Self {
        Self {
            id: StepId::new(id),
            kind: AgentKind::Custom("agent".into()),
            prompt_template: String::new(),
            capabilities: AgentCapabilities::new(),
            output_key: "output".into(),
            next: Vec::new(),
            description: None,
            timeout_ms: None,
            retry_policy: RetryPolicy::default(),
            meta HashMap::new(),
        }
    }

    pub fn kind(mut self, kind: AgentKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn prompt(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    pub fn capability(mut self, cap: AgentCapabilities) -> Self {
        self.capabilities = cap;
        self
    }

    pub fn output(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    pub fn then(mut self, next: impl Into<StepId>) -> Self {
        self.next.push(next.into());
        self
    }
    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn build(self) -> Result<StepDefinition, FlowError> {
        if self.prompt_template.is_empty() {
            return Err(FlowError::InvalidWorkflow("agent step requires a prompt_template".into()));
        }

        Ok(StepDefinition {
            id: self.id,
            kind: StepKind::Agent {
                agent_kind: self.kind,
                prompt_template: self.prompt_template,
                capabilities: self.capabilities,
                output_key: self.output_key,
            },
            next: self.next,
            description: self.description,
            timeout_ms: self.timeout_ms,
            retry_policy: self.retry_policy,
            meta self.meta,
        })
    }
}

/// Builder for `StepKind::Tool` steps.
pub struct ToolStepBuilder {
    id: StepId,
    tool_name: String,
    arguments_template: Value,
    output_key: String,
    next: Vec<StepId>,
    description: Option<String>,
    timeout_ms: Option<u64>,
    retry_policy: RetryPolicy,
    meta HashMap<String, Value>,
}
impl ToolStepBuilder {
    fn new(id: impl Into<String>) -> Self {
        Self {
            id: StepId::new(id),
            tool_name: String::new(),
            arguments_template: Value::Object(Default::default()),
            output_key: "output".into(),
            next: Vec::new(),
            description: None,
            timeout_ms: None,
            retry_policy: RetryPolicy::default(),
            meta HashMap::new(),
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }

    pub fn arguments(mut self, args: Value) -> Self {
        self.arguments_template = args;
        self
    }

    pub fn output(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    pub fn then(mut self, next: impl Into<StepId>) -> Self {
        self.next.push(next.into());
        self
    }

    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
    pub fn build(self) -> Result<StepDefinition, FlowError> {
        if self.tool_name.is_empty() {
            return Err(FlowError::InvalidWorkflow("tool step requires a name".into()));
        }

        Ok(StepDefinition {
            id: self.id,
            kind: StepKind::Tool {
                tool_name: self.tool_name,
                arguments_template: self.arguments_template,
                output_key: self.output_key,
            },
            next: self.next,
            description: self.description,
            timeout_ms: self.timeout_ms,
            retry_policy: self.retry_policy,
            meta self.meta,
        })
    }
}

/// Builder for `StepKind::Conditional` steps.
pub struct ConditionalStepBuilder {
    id: StepId,
    condition_prompt: String,
    output_schema: Value,
    branches: HashMap<String, StepId>,
    default_branch: Option<StepId>,
    next: Vec<StepId>,
    description: Option<String>,
    timeout_ms: Option<u64>,
    retry_policy: RetryPolicy,
    meta HashMap<String, Value>,
}

impl ConditionalStepBuilder {
    fn new(id: impl Into<String>) -> Self {
        Self {
            id: StepId::new(id),
            condition_prompt: String::new(),
            output_schema: Value::Object(Default::default()),
            branches: HashMap::new(),
            default_branch: None,
            next: Vec::new(),
            description: None,
            timeout_ms: None,
            retry_policy: RetryPolicy::default(),
            meta HashMap::new(),
        }    }

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.condition_prompt = prompt.into();
        self
    }

    pub fn schema(mut self, schema: Value) -> Self {
        self.output_schema = schema;
        self
    }

    pub fn branch(mut self, condition_value: impl Into<String>, target: impl Into<StepId>) -> Self {
        self.branches.insert(condition_value.into(), target.into());
        self
    }

    pub fn default(mut self, target: impl Into<StepId>) -> Self {
        self.default_branch = Some(target.into());
        self
    }

    pub fn then(mut self, next: impl Into<StepId>) -> Self {
        self.next.push(next.into());
        self
    }

    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn build(self) -> Result<StepDefinition, FlowError> {
        if self.condition_prompt.is_empty() {
            return Err(FlowError::InvalidWorkflow("conditional step requires a prompt".into()));
        }

        Ok(StepDefinition {
            id: self.id,
            kind: StepKind::Conditional {                condition_prompt: self.condition_prompt,
                output_schema: self.output_schema,
                branches: self.branches,
                default_branch: self.default_branch,
            },
            next: self.next,
            description: self.description,
            timeout_ms: self.timeout_ms,
            retry_policy: self.retry_policy,
            meta self.meta,
        })
    }
}

/// Builder for `StepKind::Parallel` steps.
pub struct ParallelStepBuilder {
    id: StepId,
    steps: Vec<StepId>,
    join_output_key: String,
    next: Vec<StepId>,
    description: Option<String>,
    timeout_ms: Option<u64>,
    retry_policy: RetryPolicy,
    meta HashMap<String, Value>,
}

impl ParallelStepBuilder {
    fn new(id: impl Into<String>) -> Self {
        Self {
            id: StepId::new(id),
            steps: Vec::new(),
            join_output_key: "parallel_output".into(),
            next: Vec::new(),
            description: None,
            timeout_ms: None,
            retry_policy: RetryPolicy::default(),
            meta HashMap::new(),
        }
    }

    pub fn add(mut self, step_id: impl Into<StepId>) -> Self {
        self.steps.push(step_id.into());
        self
    }

    pub fn join_key(mut self, key: impl Into<String>) -> Self {
        self.join_output_key = key.into();
        self
    }
    pub fn then(mut self, next: impl Into<StepId>) -> Self {
        self.next.push(next.into());
        self
    }

    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn build(self) -> Result<StepDefinition, FlowError> {
        if self.steps.is_empty() {
            return Err(FlowError::InvalidWorkflow("parallel step requires at least one child".into()));
        }

        Ok(StepDefinition {
            id: self.id,
            kind: StepKind::Parallel {
                steps: self.steps,
                join_output_key: self.join_output_key,
            },
            next: self.next,
            description: self.description,
            timeout_ms: self.timeout_ms,
            retry_policy: self.retry_policy,
            meta self.meta,
        })
    }
}

/// Builder for `StepKind::Transform` steps.
pub struct TransformStepBuilder {
    id: StepId,
    input_key: String,
    transform: TransformKind,
    output_key: String,
    next: Vec<StepId>,
    description: Option<String>,
    timeout_ms: Option<u64>,
    retry_policy: RetryPolicy,    meta HashMap<String, Value>,
}

impl TransformStepBuilder {
    fn new(id: impl Into<String>) -> Self {
        Self {
            id: StepId::new(id),
            input_key: "input".into(),
            transform: TransformKind::SerializeJson,
            output_key: "output".into(),
            next: Vec::new(),
            description: None,
            timeout_ms: None,
            retry_policy: RetryPolicy::default(),
            meta HashMap::new(),
        }
    }

    pub fn input(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    pub fn transform(mut self, t: TransformKind) -> Self {
        self.transform = t;
        self
    }

    pub fn output(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    pub fn then(mut self, next: impl Into<StepId>) -> Self {
        self.next.push(next.into());
        self
    }

    pub fn build(self) -> Result<StepDefinition, FlowError> {
        Ok(StepDefinition {
            id: self.id,
            kind: StepKind::Transform {
                input_key: self.input_key,
                transform: self.transform,
                output_key: self.output_key,
            },
            next: self.next,
            description: self.description,
            timeout_ms: self.timeout_ms,
            retry_policy: self.retry_policy,            meta self.meta,
        })
    }
}
