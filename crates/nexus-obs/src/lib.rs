// crates/nexus-obs/src/lib.rs

//! # Nexus Observability
//!
//! Observability, tracing, metrics, and TUI dashboard for the Nexus AI agent runtime.
//!
//! `nexus-obs` provides:
//! - **Persistent cost ledger**: SQLite-backed accounting for all LLM API calls
//! - **OpenTelemetry integration**: Distributed tracing and metrics export via OTLP
//! - **Live TUI dashboard**: Real-time terminal UI for monitoring agent mesh health
//! - **Session replay engine**: Debugging superpower to re-run past agent sessions
//! - **Unified observability handle**: Single entry point for recording costs and traces
//!
//! ## Usage
//!
//! ```rust
//! use nexus_obs::{ObsHandle, NexusTracer, PersistentCostLedger};
//! use nexus_router::cost::CostRecord;
//! use nexus_proto::agent::AgentId;
//! use nexus_proto::model::ProviderId;
//!
//! async fn example() {
//!     // Initialize tracing and ledger
//!     let tracer = NexusTracer::init_tracing("info", "pretty", Some("http://localhost:4318")).unwrap();
//!     let ledger = PersistentCostLedger::new("./data/cost.db").await.unwrap();
//!
//!     // Create unified handle
//!     let obs = ObsHandle::new(tracer, ledger);
//!
//!     // Record a cost entry
//!     let record = CostRecord::new(
//!         AgentId::new(),
//!         ProviderId::OpenAI,
//!         "gpt-4o-mini".to_string(),
//!         150,
//!         75,
//!         0.0015,
//!         450,
//!     );
//!     obs.record_cost(record);
//!
//!     // Use tracer for spans
//!     let span = nexus_obs::tracer::agent_span(obs.tracer(), AgentId::new(), "research");
//!     // ... do work ...
//!     span.end();
//! }
//! ```
//!
//! ## Thread Safety
//!//! All public types are `Send + Sync`. The `ObsHandle` uses `Arc` for shared ownership
//! and async tasks for non-blocking cost recording.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

// =============================================================================
// Module Declarations
// =============================================================================

pub mod error;
pub mod ledger;
pub mod tracer;
pub mod exporter;
pub mod replay;
pub mod tui;

// =============================================================================
// Unified Observability Handle
// =============================================================================

use std::sync::Arc;

use nexus_router::cost::CostRecord;

use crate::ledger::PersistentCostLedger;
use crate::tracer::NexusTracer;

/// Unified entry point for observability operations.
///
/// Wraps the tracer and cost ledger for convenient, fire-and-forget recording.
#[derive(Debug, Clone)]
pub struct ObsHandle {
    tracer: Arc<NexusTracer>,
    ledger: Arc<PersistentCostLedger>,
}

impl ObsHandle {
    /// Creates a new observability handle.
    pub fn new(tracer: NexusTracer, ledger: Arc<PersistentCostLedger>) -> Self {
        Self {
            tracer: Arc::new(tracer),
            ledger,
        }
    }
    /// Returns a reference to the tracer.
    pub fn tracer(&self) -> &NexusTracer {
        &self.tracer
    }

    /// Returns a reference to the persistent cost ledger.
    pub fn ledger(&self) -> &PersistentCostLedger {
        &self.ledger
    }

    /// Records a cost entry asynchronously without blocking the caller.
    /// Errors are logged but not propagated.
    pub fn record_cost(&self, rec: CostRecord) {
        let ledger = Arc::clone(&self.ledger);
        tokio::spawn(async move {
            if let Err(e) = ledger.record(rec).await {
                tracing::warn!(error = %e, "failed to record cost entry");
            }
        });
    }
}

// =============================================================================
// Public Re-Exports
// =============================================================================

// Core types
pub use error::ObsError;
pub use ledger::{PersistentCostLedger, DailyCostSummary};
pub use tracer::{NexusTracer, TracingEvent, init_tracing, get_tracer};
pub use exporter::{MetricsExporter, NexusInstruments, register_metrics};
pub use replay::{ReplaySession, ReplayEvent, SessionSummary, ReplayDiff};
pub use tui::{TuiApp, TuiConfig, NexusTheme};

// Convenience prelude
pub mod prelude {
    pub use crate::{ObsHandle, ObsError};
    pub use crate::ledger::PersistentCostLedger;
    pub use crate::tracer::{NexusTracer, TracingEvent, init_tracing};
    pub use crate::exporter::{MetricsExporter, NexusInstruments, register_metrics};
    pub use crate::replay::{ReplaySession, SessionSummary, ReplayDiff};
    pub use crate::tui::{TuiApp, TuiConfig, NexusTheme};
}
