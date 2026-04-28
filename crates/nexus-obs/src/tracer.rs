// crates/nexus-obs/src/tracer.rs

use std::collections::HashMap;
use std::sync::OnceLock;

use opentelemetry::global;
use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{self as sdktrace, RandomIdGenerator, Sampler};
use opentelemetry_sdk::Resource;
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use nexus_proto::agent::AgentId;
use uuid::Uuid;

use crate::error::ObsError;

/// The Nexus tracing wrapper, providing a convenient interface for creating spans.
#[derive(Debug, Clone)]
pub struct NexusTracer {
    tracer: opentelemetry::trace::Tracer,
}

static TRACER: OnceLock<NexusTracer> = OnceLock::new();

/// Initializes the global tracing subscriber and OpenTelemetry pipeline.
///
/// # Arguments
/// * `log_level` - Log level filter (e.g., "info", "debug", "trace")
/// * `log_format` - Output format: "pretty", "json", or "compact"
/// * `otlp_endpoint` - Optional OTLP HTTP endpoint (e.g., "http://localhost:4318")
///
/// # Returns
/// * `Ok(NexusTracer)` - If initialization succeeds
/// * `Err(ObsError)` - If subscriber installation or OTLP setup fails
pub fn init_tracing(
    log_level: &str,
    log_format: &str,
    otlp_endpoint: Option<&str>,
) -> Result<NexusTracer, ObsError> {
    // Set global propagator for context propagation
    global::set_text_map_propagator(TraceContextPropagator::new());

    // Build the tracing subscriber with EnvFilter and formatting
    let filter = EnvFilter::try_new(log_level)
        .map_err(|e| ObsError::Tracing(format!("invalid log level: {}", e)))?;
    let fmt_layer = match log_format {
        "pretty" => tracing_subscriber::fmt::layer()
            .pretty()
            .with_span_events(FmtSpan::CLOSE),
        "json" => tracing_subscriber::fmt::layer()
            .json()
            .with_span_events(FmtSpan::CLOSE),
        "compact" => tracing_subscriber::fmt::layer()
            .compact()
            .with_span_events(FmtSpan::CLOSE),
        other => {
            return Err(ObsError::Tracing(format!(
                "unsupported log format: {}",
                other
            )));
        }
    };

    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer);

    // Optionally add OpenTelemetry layer
    if let Some(endpoint) = otlp_endpoint {
        let otel_layer = init_otlp_tracer(endpoint)?;
        subscriber.with(otel_layer).try_init().map_err(|e| {
            ObsError::Tracing(format!("failed to init tracing subscriber: {}", e))
        })?;
    } else {
        subscriber.try_init().map_err(|e| {
            ObsError::Tracing(format!("failed to init tracing subscriber: {}", e))
        })?;
    }

    // Create and store the NexusTracer
    let tracer = NexusTracer {
        tracer: global::tracer("nexus-obs"),
    };

    if let Err(e) = TRACER.set(tracer.clone()) {
        return Err(ObsError::Tracing(format!(
            "failed to set global tracer: {:?}",
            e
        )));
    }

    Ok(tracer)
}
/// Initializes the OpenTelemetry tracer with an OTLP HTTP exporter.
fn init_otlp_tracer(
    endpoint: &str,
) -> Result<opentelemetry::trace::Tracer, ObsError> {
    let exporter = opentelemetry_otlp::new_exporter()
        .http()
        .with_endpoint(endpoint)
        .with_timeout(std::time::Duration::from_secs(10))
        .build_span_exporter()
        .map_err(|e| ObsError::Otlp(format!("failed to build OTLP exporter: {}", e)))?;

    let provider = sdktrace::TracerProvider::builder()
        .with_sampler(Sampler::AlwaysOn)
        .with_id_generator(RandomIdGenerator::default())
        .with_max_events_per_span(64)
        .with_max_attributes_per_span(128)
        .with_resource(Resource::new(vec![
            SERVICE_NAME.string("nexus-runtime"),
            SERVICE_VERSION.string(env!("CARGO_PKG_VERSION")),
        ]))
        .with_batch_exporter(exporter)
        .build();

    let tracer = provider.tracer("nexus-obs");

    // Set as global provider for other crates to use
    global::set_tracer_provider(provider);

    Ok(tracer)
}

/// Returns the global `NexusTracer` instance, if initialized.
pub fn get_tracer() -> Option<NexusTracer> {
    TRACER.get().cloned()
}

// =============================================================================
// Span Helper Functions
// =============================================================================

/// Creates a span for an agent operation.
pub fn agent_span(tracer: &NexusTracer, agent_id: AgentId, operation: &str) -> Span {
    tracer
        .tracer
        .span_builder(format!("agent.{}", operation))
        .with_attributes(vec![
            opentelemetry::KeyValue::new("nexus.agent.id", agent_id.to_string()),
            opentelemetry::KeyValue::new("nexus.agent.operation", operation.to_string()),
        ])
        .start(&tracer.tracer)}

/// Creates a span for a model request.
pub fn model_request_span(tracer: &NexusTracer, model: &str, provider: &str) -> Span {
    tracer
        .tracer
        .span_builder("model.request")
        .with_attributes(vec![
            opentelemetry::KeyValue::new("nexus.model.name", model.to_string()),
            opentelemetry::KeyValue::new("nexus.model.provider", provider.to_string()),
        ])
        .start(&tracer.tracer)
}

/// Creates a span for a tool call.
pub fn tool_call_span(tracer: &NexusTracer, tool_name: &str) -> Span {
    tracer
        .tracer
        .span_builder("tool.call")
        .with_attributes(vec![opentelemetry::KeyValue::new(
            "nexus.tool.name",
            tool_name.to_string(),
        )])
        .start(&tracer.tracer)
}

/// Creates a span for a workflow execution.
pub fn workflow_span(tracer: &NexusTracer, workflow_name: &str, run_id: Uuid) -> Span {
    tracer
        .tracer
        .span_builder("workflow.run")
        .with_attributes(vec![
            opentelemetry::KeyValue::new("nexus.workflow.name", workflow_name.to_string()),
            opentelemetry::KeyValue::new("nexus.workflow.run_id", run_id.to_string()),
        ])
        .start(&tracer.tracer)
}

// =============================================================================
// TracingEvent — Structured Events for Event Bus
// =============================================================================

/// Structured events for the observability event bus.
#[derive(Debug, Clone)]
pub enum TracingEvent {
    /// Generic agent lifecycle or state change event.
    AgentEvent {
        agent_id: AgentId,
        event: String,
        attributes: HashMap<String, String>,    },
    /// Model request initiated.
    ModelRequest {
        agent_id: AgentId,
        model: String,
        tokens: u32,
    },
    /// Model response received.
    ModelResponse {
        agent_id: AgentId,
        model: String,
        tokens: u32,
        latency_ms: u64,
    },
    /// Tool call completed.
    ToolCall {
        agent_id: AgentId,
        tool: String,
        success: bool,
        duration_ms: u64,
    },
    /// Workflow step status update.
    WorkflowStep {
        run_id: Uuid,
        step_id: String,
        status: String,
    },
}

impl TracingEvent {
    /// Emits this event as a tracing span or log record.
    pub fn emit(&self, tracer: &NexusTracer) {
        match self {
            TracingEvent::AgentEvent {
                agent_id,
                event,
                attributes,
            } => {
                let mut span = agent_span(tracer, *agent_id, event);
                for (k, v) in attributes {
                    span.set_attribute(opentelemetry::KeyValue::new(k.clone(), v.clone()));
                }
            }
            TracingEvent::ModelRequest {
                agent_id,
                model,
                tokens,
            } => {
                let span = model_request_span(tracer, model, "unknown");
                span.set_attribute(opentelemetry::KeyValue::new(                    "nexus.agent.id",
                    agent_id.to_string(),
                ));
                span.set_attribute(opentelemetry::KeyValue::new(
                    "nexus.model.input_tokens",
                    *tokens as i64,
                ));
            }
            TracingEvent::ModelResponse {
                agent_id,
                model,
                tokens,
                latency_ms,
            } => {
                let span = model_request_span(tracer, model, "unknown");
                span.set_attribute(opentelemetry::KeyValue::new(
                    "nexus.agent.id",
                    agent_id.to_string(),
                ));
                span.set_attribute(opentelemetry::KeyValue::new(
                    "nexus.model.output_tokens",
                    *tokens as i64,
                ));
                span.set_attribute(opentelemetry::KeyValue::new(
                    "nexus.model.latency_ms",
                    *latency_ms as i64,
                ));
                span.end();
            }
            TracingEvent::ToolCall {
                agent_id,
                tool,
                success,
                duration_ms,
            } => {
                let span = tool_call_span(tracer, tool);
                span.set_attribute(opentelemetry::KeyValue::new(
                    "nexus.agent.id",
                    agent_id.to_string(),
                ));
                span.set_attribute(opentelemetry::KeyValue::new(
                    "nexus.tool.success",
                    *success,
                ));
                span.set_attribute(opentelemetry::KeyValue::new(
                    "nexus.tool.duration_ms",
                    *duration_ms as i64,
                ));
                span.end();
            }            TracingEvent::WorkflowStep {
                run_id,
                step_id,
                status,
            } => {
                let span = workflow_span(tracer, "workflow", *run_id);
                span.set_attribute(opentelemetry::KeyValue::new(
                    "nexus.workflow.step_id",
                    step_id.clone(),
                ));
                span.set_attribute(opentelemetry::KeyValue::new(
                    "nexus.workflow.step_status",
                    status.clone(),
                ));
                span.end();
            }
        }
    }
}
