use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Duration;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::agent::{AgentCapabilities, AgentKind};
use crate::error::NexusError;

// =============================================================================
// Workflow & Step Identification
// =============================================================================

/// Unique identifier for a workflow definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkflowId(Uuid);

impl WorkflowId {
    /// Generates a new random v4 UUID for a workflow.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Returns a nil UUID (all zeros), useful for testing or sentinel values.
    pub fn nil() -> Self {
        Self(Uuid::nil())
    }

    /// Returns a reference to the underlying `Uuid`.
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl fmt::Display for WorkflowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for WorkflowId {
    fn default() -> Self {
        Self::new()
    }
}

/// Human-readable identifier for a step within a workflow.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StepId(String);

impl StepId {
    /// Constructs a `StepId` from a name string.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the underlying string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for StepId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for StepId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

// =============================================================================
// Step Kinds & Transforms
// =============================================================================

/// The operational type of a workflow step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StepKind {
    /// Spawn an agent with a prompt template and collect output.
    Agent {
        /// Type of agent to spawn.
        agent_kind: AgentKind,
        /// Handlebars-style prompt template.
        prompt_template: String,
        /// Required capabilities for the spawned agent.
        capabilities: AgentCapabilities,
        /// Key in the workflow context to store the result.
        output_key: String,
    },

    /// Execute a WASM tool with templated arguments.
    Tool {
        /// Name of the tool to execute.
        tool_name: String,
        /// JSON template for tool arguments.
        arguments_template: Value,
        /// Key in the workflow context to store the result.
        output_key: String,
    },

    /// Route execution based on LLM-evaluated condition.
    Conditional {
        /// Prompt describing the condition to evaluate.
        condition_prompt: String,
        /// JSON Schema for the expected condition output.
        output_schema: Value,
        /// Map of branch values to the next step IDs.
        branches: HashMap<String, StepId>,
        /// Optional default step to execute if no branch matches.
        default_branch: Option<StepId>,
    },

    /// Execute multiple steps concurrently and join results.
    Parallel {
        /// List of step IDs to execute in parallel.
        steps: Vec<StepId>,
        /// Key in the workflow context to store the combined results.
        join_output_key: String,
    },

    /// Apply a deterministic transform to context data.
    Transform {
        /// Key in the context to read input from.
        input_key: String,
        /// The transformation logic to apply.
        transform: TransformKind,
        /// Key in the context to store the output.
        output_key: String,
    },

    /// Pause execution for a fixed duration.
    Wait {
        /// Delay duration in milliseconds.
        duration_ms: u64,
    },

    /// Terminal step marking workflow completion.
    End {
        /// Whether the workflow finished successfully.
        success: bool,
    },
}

/// Built-in data transformation operations for Transform steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "transform", rename_all = "snake_case")]
pub enum TransformKind {
    /// Extract a value from JSON using JSONPath-like syntax.
    JsonExtract {
        /// The path to extract.
        path: String,
    },

    /// Summarize text content (requires model routing).
    TextSummarize,

    /// Join an array of strings with a separator.
    TextJoin {
        /// The separator to use when joining.
        separator: String,
    },

    /// Parse a JSON string into a structured Value.
    ParseJson,
    /// Serialize a Value to a compact JSON string.
    SerializeJson,

    /// Custom transformation logic (code evaluated in sandbox).
    Custom {
        /// The transformation code to execute.
        code: String,
    },
}

// =============================================================================
// Step Definition & Retry Policy
// =============================================================================

/// Configuration for retrying failed step executions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of attempts (1 = no retry).
    pub max_attempts: u32,

    /// Initial backoff delay in milliseconds.
    pub backoff_ms: u64,

    /// Multiplier applied to backoff after each attempt.
    pub backoff_multiplier: f32,

    /// Maximum backoff cap in milliseconds.
    pub max_backoff_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff_ms: 1000,
            backoff_multiplier: 2.0,
            max_backoff_ms: 30000,
        }
    }
}

impl RetryPolicy {
    /// Returns a policy that never retries (single attempt only).
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            ..Default::default()
        }
    }

    /// Returns a policy with the specified maximum attempt count.
    pub fn with_attempts(n: u32) -> Self {
        Self {
            max_attempts: n,
            ..Default::default()
        }
    }

    /// Computes the backoff duration for a given attempt number (0-indexed).
    pub fn backoff_for_attempt(&self, attempt: u32) -> Duration {
        let multiplier = self.backoff_multiplier.powi(attempt as i32);
        let backoff = (self.backoff_ms as f64 * multiplier as f64) as u64;
        Duration::from_millis(backoff.min(self.max_backoff_ms))
    }
}

/// A node in the workflow DAG with execution configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDefinition {
    /// Unique identifier for this step within the workflow.
    pub id: StepId,

    /// The operational behavior of this step.
    pub kind: StepKind,

    /// Outgoing edges: steps to execute after this one completes successfully.
    pub next: Vec<StepId>,

    /// Optional human-readable description for documentation/UI.
    pub description: Option<String>,

    /// Optional hard timeout for step execution in milliseconds.
    pub timeout_ms: Option<u64>,

    /// Retry configuration for transient failures.
    pub retry_policy: RetryPolicy,

    /// Arbitrary metadata for observability, routing hints, or user annotations.
    pub metadata: HashMap<String, Value>,
}

// =============================================================================
// Workflow Definition & Validation
// =============================================================================

/// Complete definition of a workflow DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    /// Unique identifier for this workflow definition.
    pub id: WorkflowId,

    /// Human-readable name for UI and logging.
    pub name: String,
    /// Optional description of workflow purpose and behavior.
    pub description: Option<String>,

    /// Semantic version string for this workflow definition.
    pub version: String,

    /// Map of step ID to step configuration.
    pub steps: HashMap<StepId, StepDefinition>,

    /// Entry point step ID where execution begins.
    pub entry_step: StepId,

    /// Initial context variables available to all steps.
    pub variables: HashMap<String, Value>,

    /// Arbitrary tags for organization and discovery.
    pub tags: Vec<String>,

    /// Timestamp when this definition was created.
    pub created_at: DateTime<Utc>,
}

impl WorkflowDefinition {
    /// Validates the workflow DAG for structural correctness.
    ///
    /// Checks performed:
    /// 1. Entry step exists in the steps map
    /// 2. All `next` references point to existing steps
    /// 3. No cycles exist in the graph (DAG requirement)
    /// 4. At least one `End` step is reachable from every path starting at entry
    pub fn validate(&self) -> Result<(), NexusError> {
        // Check 1: Entry step exists
        if !self.steps.contains_key(&self.entry_step) {
            return Err(NexusError::FlowError(format!(
                "entry step '{}' not found in workflow steps",
                self.entry_step
            )));
        }

        // Check 2: All next references exist
        for (step_id, step) in &self.steps {
            for next_id in &step.next {
                if !self.steps.contains_key(next_id) {
                    return Err(NexusError::FlowError(format!(
                        "step '{}' references unknown next step '{}'",
                        step_id, next_id
                    )));
                }
            }
        }

        // Check 3: Detect cycles using DFS with recursion stack
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        fn has_cycle(
            step_id: &StepId,
            steps: &HashMap<StepId, StepDefinition>,
            visited: &mut HashSet<StepId>,
            rec_stack: &mut HashSet<StepId>,
        ) -> bool {
            visited.insert(step_id.clone());
            rec_stack.insert(step_id.clone());

            if let Some(step) = steps.get(step_id) {
                for next_id in &step.next {
                    if !visited.contains(next_id) {
                        if has_cycle(next_id, steps, visited, rec_stack) {
                            return true;
                        }
                    } else if rec_stack.contains(next_id) {
                        return true;
                    }
                }
            }

            rec_stack.remove(step_id);
            false
        }

        if has_cycle(&self.entry_step, &self.steps, &mut visited, &mut rec_stack) {
            return Err(NexusError::FlowError(
                "workflow contains a cycle; must be a DAG".into(),
            ));
        }

        // Check 4: Every path from entry reaches an End step
        fn reaches_end(
            step_id: &StepId,
            steps: &HashMap<StepId, StepDefinition>,
            memo: &mut HashMap<StepId, bool>,
        ) -> bool {
            if let Some(cached) = memo.get(step_id) {
                return *cached;
            }

            let step = match steps.get(step_id) {
                Some(s) => s,
                None => return false,
            };

            // Terminal check: End step or no outgoing edges
            if matches!(step.kind, StepKind::End { .. }) || step.next.is_empty() {
                let is_end = matches!(step.kind, StepKind::End { .. });
                memo.insert(step_id.clone(), is_end);
                return is_end;
            }

            // Recursive check: at least one path leads to End
            let result = step.next.iter().any(|next| reaches_end(next, steps, memo));
            memo.insert(step_id.clone(), result);
            result
        }

        let mut memo = HashMap::new();
        if !reaches_end(&self.entry_step, &self.steps, &mut memo) {
            return Err(NexusError::FlowError(
                "not all execution paths reach an End step".into(),
            ));
        }

        Ok(())
    }
}

// =============================================================================
// Workflow Runtime Context
// =============================================================================

/// Runtime context holding variables passed between workflow steps.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkflowContext {
    values: HashMap<String, Value>,
}

impl WorkflowContext {
    /// Creates a new empty context.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Gets a reference to a value by key, if present.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Sets or overwrites a value in the context.
    pub fn set(&mut self, key: &str, value: Value) {
        self.values.insert(key.to_string(), value);
    }

    /// Renders a handlebars-style template by replacing {{key}} placeholders
    /// with values from the context. Returns an error if any referenced key is missing.
    pub fn render_template(&self, template: &str) -> Result<String, NexusError> {
        use std::fmt::Write;

        let mut result = String::with_capacity(template.len());
        let mut chars = template.chars().peekable();
        let mut in_placeholder = false;
        let mut placeholder_name = String::new();

        while let Some(ch) = chars.next() {
            if !in_placeholder {
                if ch == '{' && chars.peek() == Some(&'{') {
                    chars.next(); // consume second {
                    in_placeholder = true;
                    placeholder_name.clear();
                } else {
                    result.push(ch);
                }
            } else {
                if ch == '}' && chars.peek() == Some(&'}') {
                    chars.next(); // consume second }
                    in_placeholder = false;

                    let value = self.values.get(placeholder_name.trim()).ok_or_else(|| {
                        NexusError::FlowError(format!(
                            "template variable '{}' not found in context",
                            placeholder_name.trim()
                        ))
                    })?;

                    // Render value: strings as-is, others as JSON
                    match value {
                        Value::String(s) => result.push_str(s),
                        _ => write!(result, "{}", value)
                            .map_err(|e| NexusError::FlowError(format!("template render error: {}", e)))?,
                    }
                } else {
                    placeholder_name.push(ch);
                }
            }
        }

        if in_placeholder {
            return Err(NexusError::FlowError(
                "unclosed template placeholder '{{'".into(),
            ));
        }

        Ok(result)
    }
}

// =============================================================================
// Execution Status Types
// =============================================================================

/// Execution status of an individual workflow step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StepStatus {
    /// Step is queued but not yet started.
    Pending,

    /// Step is currently executing.
    Running {
        /// Timestamp when the step execution began.
        started_at: DateTime<Utc>,
    },

    /// Step completed successfully with output.
    Completed {
        /// Timestamp when the step execution finished.
        finished_at: DateTime<Utc>,
        /// The output data produced by the step.
        output: Value,
    },

    /// Step failed after exhausting retries.
    Failed {
        /// Error message describing the failure.
        error: String,
        /// Total number of attempts made to execute the step.
        attempts: u32,
    },

    /// Step was skipped due to conditional branching.
    Skipped {
        /// Reason why the step was skipped.
        reason: String,
    },
}

impl StepStatus {
    /// Returns `true` if the step has reached a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            StepStatus::Completed { .. } | StepStatus::Failed { .. } | StepStatus::Skipped { .. }
        )
    }
}

/// Overall status of a workflow run instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowRunStatus {
    /// Workflow has been created but not yet started.
    Pending,
    /// Workflow is currently executing.
    Running,
    /// Workflow finished successfully.
    Completed,
    /// Workflow failed during execution.
    Failed,
    /// Workflow execution was cancelled by user.
    Cancelled,
}

/// A running instance of a workflow with per-step state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowRun {
    /// Unique identifier for this run instance.
    pub id: Uuid,

    /// ID of the workflow definition being executed.
    pub workflow_id: WorkflowId,

    /// Per-step execution status tracking.
    pub step_statuses: HashMap<StepId, StepStatus>,

    /// Shared context mutated as steps execute.
    pub context: WorkflowContext,

    /// Timestamp when execution began.
    pub started_at: DateTime<Utc>,

    /// Timestamp when execution completed (None if still running).
    pub finished_at: Option<DateTime<Utc>>,

    /// Overall run status.
    pub status: WorkflowRunStatus,
}

impl WorkflowRun {
    /// Creates a new run instance from a workflow definition.
    pub fn new(workflow: &WorkflowDefinition) -> Self {
        Self {
            id: Uuid::new_v4(),
            workflow_id: workflow.id,
            step_statuses: workflow
                .steps
                .keys()
                .map(|id| (id.clone(), StepStatus::Pending))
                .collect(),
            context: WorkflowContext {
                values: workflow.variables.clone(),
            },
            started_at: Utc::now(),
            finished_at: None,
            status: WorkflowRunStatus::Pending,
        }
    }

    /// Returns `true` if this run has completed (successfully or not).
    pub fn is_finished(&self) -> bool {
        matches!(
            self.status,
            WorkflowRunStatus::Completed | WorkflowRunStatus::Failed | WorkflowRunStatus::Cancelled
        )
    }
}
