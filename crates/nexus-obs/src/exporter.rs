// crates/nexus-obs/src/exporter.rs

use std::sync::Arc;

use opentelemetry::global;
use opentelemetry::metrics::{Meter, MeterProvider as _};
use opentelemetry_sdk::metrics::{
    self as sdkmetrics, PeriodicReader, SdkMeterProvider,
};
use opentelemetry_sdk::runtime::Tokio;
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use opentelemetry_sdk::Resource;

use crate::error::ObsError;

/// Metrics exporter that periodically pushes metrics to an OTLP endpoint.
pub struct MetricsExporter {
    provider: SdkMeterProvider,
}

impl MetricsExporter {
    /// Creates a new metrics exporter with the given OTLP endpoint.
    pub async fn new(otlp_endpoint: &str) -> Result<Self, ObsError> {
        let exporter = opentelemetry_otlp::new_exporter()
            .http()
            .with_endpoint(otlp_endpoint)
            .with_timeout(std::time::Duration::from_secs(10))
            .build_metrics_exporter(Box::new(Tokio))
            .map_err(|e| ObsError::Otlp(format!("failed to build OTLP metrics exporter: {}", e)))?;

        let reader = PeriodicReader::builder(exporter, Tokio).build();

        let provider = SdkMeterProvider::builder()
            .with_resource(Resource::new(vec![
                SERVICE_NAME.string("nexus-runtime"),
                SERVICE_VERSION.string(env!("CARGO_PKG_VERSION")),
            ]))
            .with_reader(reader)
            .build();

        Ok(Self { provider })
    }

    /// Returns the configured meter for creating instruments.
    pub fn meter(&self) -> Meter {
        self.provider.meter("nexus-obs")
    }

    /// Shuts down the exporter, flushing any pending metrics.
    pub async fn shutdown(&self) -> Result<(), ObsError> {        self.provider
            .shutdown()
            .map_err(|e| ObsError::Otlp(format!("failed to shutdown metrics provider: {}", e)))
    }
}

/// Registers the global metrics meter with OTLP export.
///
/// # Arguments
/// * `otlp_endpoint` - OTLP HTTP endpoint for metrics (e.g., "http://localhost:4318")
///
/// # Returns
/// * `Ok(Meter)` - The configured meter for creating instruments
/// * `Err(ObsError)` - If exporter setup fails
pub async fn register_metrics(otlp_endpoint: &str) -> Result<Meter, ObsError> {
    let exporter = MetricsExporter::new(otlp_endpoint).await?;
    let meter = exporter.meter();

    // Set as global provider
    global::set_meter_provider(exporter.provider);

    Ok(meter)
}

/// Convenience function to create common Nexus metrics instruments.
pub struct NexusInstruments {
    pub active_agents: opentelemetry::metrics::UpDownCounter<u64>,
    pub tokens_used: opentelemetry::metrics::Counter<u64>,
    pub requests_per_sec: opentelemetry::metrics::Histogram<f64>,
    pub tool_success_rate: opentelemetry::metrics::UpDownCounter<f64>,
    pub model_latency_p50: opentelemetry::metrics::Histogram<f64>,
    pub model_latency_p99: opentelemetry::metrics::Histogram<f64>,
}

impl NexusInstruments {
    /// Creates a new set of instruments from a meter.
    pub fn new(meter: &Meter) -> Self {
        Self {
            active_agents: meter
                .u64_up_down_counter("nexus.agents.active")
                .with_description("Number of currently active agents")
                .init(),
            tokens_used: meter
                .u64_counter("nexus.tokens.used")
                .with_description("Total tokens consumed by model requests")
                .init(),
            requests_per_sec: meter
                .f64_histogram("nexus.requests.per_second")
                .with_description("Rate of model requests per second")
                .init(),            tool_success_rate: meter
                .f64_up_down_counter("nexus.tools.success_rate")
                .with_description("Success rate of tool invocations")
                .init(),
            model_latency_p50: meter
                .f64_histogram("nexus.model.latency.p50")
                .with_unit("ms")
                .with_description("50th percentile model response latency")
                .init(),
            model_latency_p99: meter
                .f64_histogram("nexus.model.latency.p99")
                .with_unit("ms")
                .with_description("99th percentile model response latency")
                .init(),
        }
    }
}
