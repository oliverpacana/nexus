use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use futures::future::try_join_all;
use handlebars::Handlebars;
use jsonschema::{Draft, JSONSchema};
use serde_json::{json, Value};
use tokio::select;
use tokio::sync::watch;
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use nexus_proto::agent::AgentId;
use nexus_proto::model::{Message, MessageRole, ModelRequest, RoutingPolicy};
use nexus_proto::tool::ToolCall;
use nexus_proto::workflow::{
    RetryPolicy, StepDefinition, StepId, StepKind, StepStatus, TransformKind, WorkflowContext,
    WorkflowDefinition, WorkflowId, WorkflowRun, WorkflowRunStatus,
};
use nexus_kernel::KernelHandle;
use nexus_router::ModelRouter;
use nexus_tools::ToolEngine;
use nexus_mem::MemoryStore;

use crate::checkpoint::CheckpointStore;
use crate::dag::WorkflowDag;
use crate::error::FlowError;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the workflow executor runtime.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of parallel steps allowed concurrently in a workflow.
    pub max_parallel_steps: usize,
    /// Default timeout for step execution in milliseconds if not specified per-step.
    pub step_timeout_ms: u64,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_parallel_steps: 16,
            step_timeout_ms: 300_000, // 5 minutes
        }    }
}

// =============================================================================
// Step Executor
// =============================================================================

/// Executes individual workflow steps by routing to kernel, router, or tools.
pub struct StepExecutor {
    pub kernel: Arc<KernelHandle>,
    pub router: Arc<ModelRouter>,
    pub tool_engine: Arc<ToolEngine>,
    pub memory: Arc<MemoryStore>,
    handlebars: Handlebars<'static>,
}

impl StepExecutor {
    pub fn new(
        kernel: Arc<KernelHandle>,
        router: Arc<ModelRouter>,
        tool_engine: Arc<ToolEngine>,
        memory: Arc<MemoryStore>,
    ) -> Self {
        let mut handlebars = Handlebars::new();
        // Register helpers if needed; default syntax is {{var}}
        handlebars.set_strict_mode(false);
        Self {
            kernel,
            router,
            tool_engine,
            memory,
            handlebars,
        }
    }

    /// Executes a single step and returns its output value.
    #[instrument(skip(self, step, ctx), fields(step_id = %step.id, kind = ?step.kind))]
    pub async fn execute_step(
        &self,
        step: &StepDefinition,
        ctx: &mut WorkflowContext,
        run_id: Uuid,
    ) -> Result<Value, FlowError> {
        match &step.kind {
            StepKind::Agent {
                agent_kind,
                prompt_template,
                capabilities,
                output_key,
            } => {                let prompt = self.render_template(prompt_template, ctx)?;
                
                let request = ModelRequest::builder()
                    .messages(vec![Message::user(prompt)])
                    .system_prompt(step.metadata.get("system_prompt").and_then(|v| v.as_str().map(String::from)))
                    .max_tokens(step.metadata.get("max_tokens").and_then(|v| v.as_u64()).map(|v| v as u32))
                    .temperature(step.metadata.get("temperature").and_then(|v| v.as_f64()).map(|v| v as f32))
                    .routing_policy(RoutingPolicy::default()) // Could be configurable
                    .build()
                    .map_err(|e| FlowError::RequestBuild(e.to_string()))?;

                // Use a synthetic agent ID for the workflow context
                let agent_id = AgentId::new();
                
                let response = self.router.complete(request, agent_id).await
                    .map_err(|e| FlowError::AgentExecution(e.to_string()))?;

                let text = response.message.text_content();
                ctx.set(output_key, Value::String(text));
                
                Ok(serde_json::to_value(&response).unwrap_or(Value::Null))
            }

            StepKind::Tool {
                tool_name,
                arguments_template,
                output_key,
            } => {
                let rendered_args = self.render_json_values(arguments_template, ctx)?;
                
                let call = ToolCall {
                    id: Uuid::new_v4(),
                    tool_name: tool_name.clone(),
                    arguments: rendered_args,
                    agent_id: AgentId::new(),
                    trace_id: run_id,
                };

                let result = self.tool_engine.call(call).await
                    .map_err(|e| FlowError::ToolExecution(e.to_string()))?;

                if result.is_error {
                    return Err(FlowError::ToolExecution(
                        result.error_message.unwrap_or_else(|| "unknown tool error".into())
                    ));
                }

                ctx.set(output_key, result.output.clone());
                Ok(result.output)
            }
            StepKind::Conditional {
                condition_prompt,
                output_schema,
                branches,
                default_branch,
            } => {
                let prompt = self.render_template(condition_prompt, ctx)?;
                let schema_str = serde_json::to_string(output_schema)
                    .unwrap_or_else(|_| "{}".into());
                
                let full_prompt = format!(
                    "{}\n\nRespond ONLY with valid JSON matching this schema:\n{}",
                    prompt, schema_str
                );

                let request = ModelRequest::builder()
                    .messages(vec![Message::user(full_prompt)])
                    .routing_policy(RoutingPolicy::default())
                    .build()
                    .map_err(|e| FlowError::RequestBuild(e.to_string()))?;

                let response = self.router.complete(request, AgentId::new()).await
                    .map_err(|e| FlowError::AgentExecution(e.to_string()))?;

                let text = response.message.text_content();
                
                // Parse JSON from response (handle markdown code blocks if present)
                let cleaned = text.trim().trim_start_matches("```json").trim_end_matches("```").trim();
                let decision: Value = serde_json::from_str(cleaned)
                    .map_err(|e| FlowError::ConditionalParse(format!("failed to parse decision JSON: {}", e)))?;

                // Validate against schema
                let schema = JSONSchema::compile(output_schema)
                    .map_err(|e| FlowError::ConditionalSchema(e.to_string()))?;
                
                if let Err(errors) = schema.validate(&decision) {
                    let first = errors.into_iter().next().map(|e| e.to_string()).unwrap_or_default();
                    return Err(FlowError::ConditionalValidation(first));
                }

                // Determine branch
                let branch_key = decision.get("branch")
                    .or_else(|| decision.get("decision"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| FlowError::ConditionalParse("missing 'branch' or 'decision' key in response".into()))?;

                let next_id = branches.get(branch_key)
                    .or(default_branch)
                    .cloned()                    .ok_or_else(|| FlowError::ConditionalRoute(format!("no branch found for key '{}'", branch_key)))?;

                // Store decision and next step ID in context for the executor to use
                ctx.set("_workflow_branch_decision", Value::String(branch_key.to_string()));
                ctx.set("_workflow_next_step", Value::String(next_id.as_str().to_string()));

                Ok(decision)
            }

            StepKind::Parallel { steps, join_output_key } => {
                // Limit concurrency if needed
                let futures = steps.iter().map(|step_id| async {
                    // Fetch step definition from DAG (caller must ensure it exists)
                    // For simplicity, we assume the executor loop handles fetching
                    // Here we execute recursively
                    unreachable!("Parallel steps should be executed by the main loop to manage context properly")
                });

                // This branch is handled by the main executor loop for proper context management
                // We return a placeholder or implement inline in WorkflowExecutor
                Err(FlowError::Internal("Parallel execution handled by WorkflowExecutor".into()))
            }

            StepKind::Transform {
                input_key,
                transform,
                output_key,
            } => {
                let input = ctx.get(input_key)
                    .ok_or_else(|| FlowError::ContextMissing(input_key.clone()))?;

                let result = match transform {
                    TransformKind::JsonExtract { path } => {
                        // Simple JSON pointer extraction
                        let mut current = input.clone();
                        for part in path.split('/').filter(|p| !p.is_empty()) {
                            current = current.get(part).cloned().unwrap_or(Value::Null);
                        }
                        current
                    }
                    TransformKind::TextSummarize => {
                        let text = input.as_str().unwrap_or("");
                        let prompt = format!("Summarize the following text concisely:\n{}", text);
                        let request = ModelRequest::builder()
                            .messages(vec![Message::user(prompt)])
                            .max_tokens(500)
                            .build().unwrap();
                        let resp = self.router.complete(request, AgentId::new()).await
                            .map_err(|e| FlowError::Transform(e.to_string()))?;
                        Value::String(resp.message.text_content())                    }
                    TransformKind::TextJoin { separator } => {
                        if let Some(arr) = input.as_array() {
                            let joined = arr.iter()
                                .filter_map(|v| v.as_str())
                                .collect::<Vec<_>>()
                                .join(separator);
                            Value::String(joined)
                        } else {
                            Value::Null
                        }
                    }
                    TransformKind::ParseJson => {
                        if let Some(s) = input.as_str() {
                            serde_json::from_str(s).unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        }
                    }
                    TransformKind::SerializeJson => {
                        Value::String(serde_json::to_string(input).unwrap_or_default())
                    }
                    TransformKind::Custom { code } => {
                        // Custom code execution would go here (sandboxed)
                        // For now, return input unchanged with a warning
                        warn!(code_len = code.len(), "custom transform not implemented, returning input");
                        input.clone()
                    }
                };

                ctx.set(output_key, result.clone());
                Ok(result)
            }

            StepKind::Wait { duration_ms } => {
                sleep(Duration::from_millis(*duration_ms)).await;
                Ok(Value::Null)
            }

            StepKind::End { success } => {
                Ok(json!({ "success": success }))
            }
        }
    }

    /// Renders a Handlebars-style template using the workflow context.
    fn render_template(&self, template: &str, ctx: &WorkflowContext) -> Result<String, FlowError> {
        let vars: HashMap<&str, &Value> = ctx.values.iter()
            .filter(|(_, v)| v.is_string() || v.is_number() || v.is_boolean())
            .map(|(k, v)| (k.as_str(), v))            .collect();
            
        self.handlebars.render_template(template, &vars)
            .map_err(|e| FlowError::TemplateRender(e.to_string()))
    }

    /// Recursively renders string values in a JSON structure.
    fn render_json_values(&self, value: &Value, ctx: &WorkflowContext) -> Result<Value, FlowError> {
        match value {
            Value::String(s) => {
                let rendered = self.render_template(s, ctx)?;
                // Try to parse as JSON if it looks like one
                if rendered.starts_with('{') || rendered.starts_with('[') {
                    serde_json::from_str(&rendered).unwrap_or(Value::String(rendered))
                } else {
                    Value::String(rendered)
                }
            }
            Value::Array(arr) => {
                let mut new_arr = Vec::with_capacity(arr.len());
                for item in arr {
                    new_arr.push(self.render_json_values(item, ctx)?);
                }
                Ok(Value::Array(new_arr))
            }
            Value::Object(map) => {
                let mut new_map = serde_json::Map::new();
                for (k, v) in map {
                    new_map.insert(k.clone(), self.render_json_values(v, ctx)?);
                }
                Ok(Value::Object(new_map))
            }
            other => Ok(other.clone()),
        }
    }
}

// =============================================================================
// Workflow Executor
// =============================================================================

/// Executes a complete workflow DAG with checkpointing, retries, and routing.
pub struct WorkflowExecutor {
    pub dag: Arc<WorkflowDag>,
    pub step_executor: Arc<StepExecutor>,
    pub checkpoint: Arc<CheckpointStore>,
    pub config: ExecutorConfig,
}

impl WorkflowExecutor {    /// Runs a workflow from definition to completion.
    ///
    /// Supports initial execution or resumption from a previous checkpoint.
    #[instrument(skip(self, definition, initial_vars), fields(workflow_id = %definition.id))]
    pub async fn run(
        &self,
        definition: WorkflowDefinition,
        initial_vars: HashMap<String, Value>,
        resume_run_id: Option<Uuid>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<WorkflowRun, FlowError> {
        // 1. Initialize or resume run
        let mut run = if let Some(run_id) = resume_run_id {
            info!(%run_id, "resuming workflow run");
            let loaded = self.checkpoint.load_run(run_id).await
                .ok_or_else(|| FlowError::Checkpoint(format!("run {} not found", run_id)))?;
            loaded
        } else {
            info!("starting new workflow run");
            WorkflowRun::new_with_context(definition.id, initial_vars, Utc::now())
        };

        // Update status to running if pending
        if matches!(run.status, WorkflowRunStatus::Pending) {
            run.status = WorkflowRunStatus::Running;
        }
        self.checkpoint.update_run_status(run.id, run.status).await?;
        self.checkpoint.update_run_context(run.id, &run.context).await?;

        // 2. Determine starting step
        let mut current_step_id = if resume_run_id.is_some() {
            // Find first non-terminal step
            run.step_statuses.iter()
                .find(|(_, s)| !s.is_terminal())
                .map(|(id, _)| id.clone())
                .unwrap_or(definition.entry_step.clone())
        } else {
            definition.entry_step.clone()
        };

        // Initialize all steps as Pending if not present
        for step_id in self.dag.graph.node_weights() {
            run.step_statuses.entry(step_id.id.clone()).or_insert(StepStatus::Pending);
        }

        // 3. Execution loop
        let mut is_terminated = false;

        while !is_terminated && !*shutdown_rx.borrow() {
            // Save checkpoint before each step            self.checkpoint.update_run_context(run.id, &run.context).await?;

            // Get current step definition
            let step_def = self.dag.get_step(&current_step_id).ok_or_else(|| {
                FlowError::Dag(format!("step {} not found in DAG", current_step_id))
            })?;

            // Skip if already completed
            if let Some(status) = run.step_statuses.get(&current_step_id) {
                if status.is_terminal() {
                    match self.advance_step(&current_step_id, step_def, &run.context).await? {
                        Some(next_id) => current_step_id = next_id,
                        None => break, // End of workflow
                    }
                    continue;
                }
            }

            // Execute with retry logic
            let result = self.execute_with_retries(step_def, &mut run).await;

            match result {
                Ok(output) => {
                    // Update status to completed
                    let finished_at = Utc::now();
                    let status = StepStatus::Completed {
                        finished_at,
                        output: output.clone(),
                    };
                    run.step_statuses.insert(current_step_id.clone(), status.clone());
                    self.checkpoint.save_step_status(run.id, &current_step_id, &status).await?;

                    // Advance to next step
                    match self.advance_step(&current_step_id, step_def, &run.context).await? {
                        Some(next_id) => current_step_id = next_id,
                        None => is_terminated = true,
                    }
                }
                Err(e) => {
                    error!(error = %e, "step execution failed permanently");
                    let status = StepStatus::Failed {
                        error: e.to_string(),
                        attempts: step_def.retry_policy.max_attempts,
                    };
                    run.step_statuses.insert(current_step_id.clone(), status.clone());
                    self.checkpoint.save_step_status(run.id, &current_step_id, &status).await?;
                    run.status = WorkflowRunStatus::Failed;
                    self.checkpoint.update_run_status(run.id, run.status).await?;
                    return Err(e);
                }            }
        }

        // 4. Finalize
        run.status = if is_terminated {
            WorkflowRunStatus::Completed
        } else {
            WorkflowRunStatus::Cancelled
        };
        run.finished_at = Some(Utc::now());
        self.checkpoint.update_run_status(run.id, run.status).await?;
        self.checkpoint.update_run_context(run.id, &run.context).await?;

        info!(status = ?run.status, "workflow run finished");
        Ok(run)
    }

    /// Executes a single step with retry logic and timeout.
    async fn execute_with_retries(
        &self,
        step: &StepDefinition,
        run: &mut WorkflowRun,
    ) -> Result<Value, FlowError> {
        let policy = &step.retry_policy;
        let mut attempts = 0;
        let mut last_error: Option<FlowError> = None;

        loop {
            attempts += 1;
            let started_at = Utc::now();
            
            // Save running status
            let status = StepStatus::Running { started_at };
            run.step_statuses.insert(step.id.clone(), status.clone());
            self.checkpoint.save_step_status(run.id, &step.id, &status).await?;

            // Determine timeout
            let timeout_ms = step.timeout_ms.unwrap_or(self.config.step_timeout_ms);

            // Execute step
            let exec_result = timeout(Duration::from_millis(timeout_ms), async {
                self.execute_step_logic(&step.id, run).await
            }).await;

            match exec_result {
                Ok(Ok(output)) => {
                    debug!(attempts, "step succeeded");
                    return Ok(output);
                }
                Ok(Err(e)) => {                    last_error = Some(e.clone());
                    if !self.should_retry(&e, &policy, attempts) {
                        return Err(e);
                    }
                    warn!(attempts, error = %e, "retrying step");
                    let backoff = policy.backoff_for_attempt(attempts - 1);
                    sleep(backoff).await;
                }
                Err(_) => {
                    let e = FlowError::Timeout(format!("step {} timed out after {}ms", step.id, timeout_ms));
                    last_error = Some(e.clone());
                    if !self.should_retry(&e, &policy, attempts) {
                        return Err(e);
                    }
                    warn!(attempts, "retrying after timeout");
                    let backoff = policy.backoff_for_attempt(attempts - 1);
                    sleep(backoff).await;
                }
            }
        }
    }

    /// Core logic for executing a step (handles Parallel specially).
    async fn execute_step_logic(
        &self,
        step_id: &StepId,
        run: &mut WorkflowRun,
    ) -> Result<Value, FlowError> {
        let step = self.dag.get_step(step_id).unwrap();
        
        if let StepKind::Parallel { steps, join_output_key } = &step.kind {
            // Execute children concurrently
            let futures: Vec<_> = steps.iter().map(|child_id| async {
                let child_def = self.dag.get_step(child_id).ok_or_else(|| {
                    FlowError::Dag(format!("parallel child {} not found", child_id))
                })?;
                self.step_executor.execute_step(child_def, &mut run.context, run.id).await
            }).collect();

            let outputs = try_join_all(futures).await?;
            let result = Value::Array(outputs);
            run.context.set(join_output_key, result.clone());
            return Ok(result);
        }

        // Handle Conditional routing update from context
        if let StepKind::Conditional { .. } = &step.kind {
            if let Some(Value::String(next_str)) = run.context.get("_workflow_next_step") {
                // The step_executor already set the next step in context
                // We just return the decision value            }
        }

        self.step_executor.execute_step(step, &mut run.context, run.id).await
    }

    /// Determines the next step ID based on current step completion.
    async fn advance_step(
        &self,
        current_id: &StepId,
        step: &StepDefinition,
        ctx: &WorkflowContext,
    ) -> Result<Option<StepId>, FlowError> {
        match &step.kind {
            StepKind::End { .. } => Ok(None),
            StepKind::Conditional { .. } => {
                if let Some(Value::String(next_str)) = ctx.get("_workflow_next_step") {
                    Ok(Some(StepId::new(next_str)))
                } else {
                    Err(FlowError::Dag("conditional step did not set next step in context".into()))
                }
            }
            _ => {
                // Follow first next edge (or handle branching if needed)
                step.next.first().cloned().map_or(Ok(None), Ok)
            }
        }
    }

    /// Checks if an error should trigger a retry based on policy and error type.
    fn should_retry(&self, err: &FlowError, policy: &RetryPolicy, attempts: u32) -> bool {
        if attempts >= policy.max_attempts {
            return false;
        }

        // Never retry capability/security errors
        if matches!(err, FlowError::ToolExecution(_) | FlowError::AgentExecution(_)) {
            // Check if error message indicates permanent failure
            let msg = err.to_string().to_lowercase();
            if msg.contains("capability") || msg.contains("permission") || msg.contains("not found") {
                return false;
            }
        }

        // Timeouts and transient network/provider errors are retryable
        matches!(err, FlowError::Timeout(_)) || err.is_retryable()
    }
}
