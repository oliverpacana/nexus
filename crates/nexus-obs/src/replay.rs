// crates/nexus-obs/src/replay.rs

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use nexus_mem::episodic::EpisodicStore;
use nexus_proto::memory::{EpisodicEvent, EpisodicEventType};
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use tracing::debug;
use uuid::Uuid;

use crate::error::ObsError;

/// A single event in the context of a replay session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEvent {
    /// The original episodic event.
    pub event: EpisodicEvent,
    /// Timestamp when this event was emitted during replay.
    pub replay_timestamp: DateTime<Utc>,
    /// Milliseconds from the start of the original session to this event.
    pub delta_from_start_ms: u64,
    /// Whether this is the last event in the session.
    pub is_last: bool,
}

/// Summary statistics for a replayed session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: Uuid,
    pub agent_id: nexus_proto::agent::AgentId,
    pub event_count: u64,
    pub duration_ms: u64,
    pub tool_calls: u64,
    pub model_requests: u64,
    pub memory_writes: u64,
    pub first_event: DateTime<Utc>,
    pub last_event: DateTime<Utc>,
}

impl SessionSummary {
    /// Computes summary statistics from a slice of episodic events.
    pub fn from_events(events: &[EpisodicEvent]) -> Self {
        let session_id = events.first().map(|e| e.session_id).unwrap_or(Uuid::nil());
        let agent_id = events.first().map(|e| e.agent_id).unwrap_or(nexus_proto::agent::AgentId::nil());

        let event_count = events.len() as u64;
        let duration_ms = if let (Some(first), Some(last)) = (events.first(), events.last()) {            last.timestamp.signed_duration_since(first.timestamp).num_milliseconds() as u64
        } else {
            0
        };

        let tool_calls = events
            .iter()
            .filter(|e| matches!(e.event_type, EpisodicEventType::ToolCalled | EpisodicEventType::ToolResult))
            .count() as u64;

        let model_requests = events
            .iter()
            .filter(|e| matches!(e.event_type, EpisodicEventType::ModelRequest | EpisodicEventType::ModelResponse))
            .count() as u64;

        let memory_writes = events
            .iter()
            .filter(|e| matches!(e.event_type, EpisodicEventType::MemoryWrite))
            .count() as u64;

        let first_event = events.first().map(|e| e.timestamp).unwrap_or(Utc::now());
        let last_event = events.last().map(|e| e.timestamp).unwrap_or(Utc::now());

        Self {
            session_id,
            agent_id,
            event_count,
            duration_ms,
            tool_calls,
            model_requests,
            memory_writes,
            first_event,
            last_event,
        }
    }
}

/// A session replay engine that re-executes a past agent session from its episodic log.
pub struct ReplaySession {
    pub session_id: Uuid,
    pub agent_id: nexus_proto::agent::AgentId,
    pub events: Vec<EpisodicEvent>,
    pub position: usize,
    pub replay_speed: f32,
    session_start: Option<DateTime<Utc>>,
}

impl ReplaySession {
    /// Loads all events for a session from the episodic store, sorted by sequence.
    pub async fn load(        episodic: &EpisodicStore,
        session_id: Uuid,
    ) -> Result<Self, ObsError> {
        let events = episodic
            .get_session(session_id)
            .await
            .map_err(|e| ObsError::Replay(format!("failed to load session: {}", e)))?;

        if events.is_empty() {
            return Err(ObsError::Replay("session not found or empty".into()));
        }

        let agent_id = events[0].agent_id;
        let session_start = events.first().map(|e| e.timestamp);

        // Sort by sequence to ensure chronological order
        let mut sorted = events;
        sorted.sort_by_key(|e| e.sequence);

        Ok(Self {
            session_id,
            agent_id,
            events: sorted,
            position: 0,
            replay_speed: 1.0,
            session_start,
        })
    }

    /// Replays the next event in the session, applying timing delay based on replay_speed.
    /// Returns `None` when all events have been replayed.
    pub async fn replay_next(&mut self) -> Option<ReplayEvent> {
        if self.position >= self.events.len() {
            return None;
        }

        let event = &self.events[self.position];
        let is_last = self.position == self.events.len() - 1;

        // Calculate delay based on original timing and replay speed
        if let Some(start) = self.session_start {
            let original_delta = event.timestamp.signed_duration_since(start).num_milliseconds() as u64;
            let previous_delta = if self.position > 0 {
                self.events[self.position - 1]
                    .timestamp
                    .signed_duration_since(start)
                    .num_milliseconds() as u64
            } else {
                0
            };
            let step_delta = original_delta.saturating_sub(previous_delta);
            let delay_ms = if self.replay_speed > 0.0 {
                (step_delta as f32 / self.replay_speed) as u64
            } else {
                0 // Instant replay
            };

            if delay_ms > 0 {
                sleep(Duration::from_millis(delay_ms)).await;
            }
        }

        let replay_event = ReplayEvent {
            event: event.clone(),
            replay_timestamp: Utc::now(),
            delta_from_start_ms: self.session_start
                .map(|start| event.timestamp.signed_duration_since(start).num_milliseconds() as u64)
                .unwrap_or(0),
            is_last,
        };

        self.position += 1;
        Some(replay_event)
    }

    /// Replays all remaining events in the session, collecting them into a vector.
    pub async fn replay_all(&mut self) -> Vec<ReplayEvent> {
        let mut events = Vec::new();
        while let Some(evt) = self.replay_next().await {
            events.push(evt);
        }
        events
    }

    /// Seeks to a specific sequence number in the session.
    pub fn seek(&mut self, sequence: u64) {
        if let Some(idx) = self.events.iter().position(|e| e.sequence == sequence) {
            self.position = idx;
        }
    }

    /// Returns `true` if all events have been replayed.
    pub fn is_done(&self) -> bool {
        self.position >= self.events.len()
    }

    /// Returns the replay progress as a fraction between 0.0 and 1.0.
    pub fn progress(&self) -> f32 {
        if self.events.is_empty() {            1.0
        } else {
            self.position as f32 / self.events.len() as f32
        }
    }

    /// Returns summary statistics for the session.
    pub fn summary(&self) -> SessionSummary {
        SessionSummary::from_events(&self.events)
    }

    /// Sets the replay speed multiplier (1.0 = realtime, 2.0 = 2x, 0.0 = instant).
    pub fn set_speed(&mut self, speed: f32) {
        self.replay_speed = speed.max(0.0);
    }
}

/// Compares two sessions to identify differences in agent behavior.
#[derive(Debug, Clone)]
pub struct ReplayDiff {
    /// Events present in session A but not in B (by sequence).
    pub events_in_a_only: Vec<EpisodicEvent>,
    /// Events present in session B but not in A (by sequence).
    pub events_in_b_only: Vec<EpisodicEvent>,
    /// Events at the same sequence with different payloads.
    pub different_outcomes: Vec<(EpisodicEvent, EpisodicEvent)>,
    /// Duration of session A in milliseconds.
    pub a_duration_ms: u64,
    /// Duration of session B in milliseconds.
    pub b_duration_ms: u64,
}

impl ReplayDiff {
    /// Compares two slices of episodic events by sequence number and payload.
    pub fn diff(session_a: &[EpisodicEvent], session_b: &[EpisodicEvent]) -> Self {
        let mut events_in_a_only = Vec::new();
        let mut events_in_b_only = Vec::new();
        let mut different_outcomes = Vec::new();

        // Index events by sequence for quick lookup
        let map_a: HashMap<u64, &EpisodicEvent> = session_a.iter().map(|e| (e.sequence, e)).collect();
        let map_b: HashMap<u64, &EpisodicEvent> = session_b.iter().map(|e| (e.sequence, e)).collect();

        // Find events only in A or different
        for (seq, event_a) in &map_a {
            match map_b.get(seq) {
                Some(event_b) => {
                    // Compare event type and payload structurally
                    if event_a.event_type != event_b.event_type || event_a.payload != event_b.payload {
                        different_outcomes.push(((*event_a).clone(), (*event_b).clone()));                    }
                }
                None => {
                    events_in_a_only.push((*event_a).clone());
                }
            }
        }

        // Find events only in B
        for (seq, event_b) in &map_b {
            if !map_a.contains_key(seq) {
                events_in_b_only.push((*event_b).clone());
            }
        }

        // Calculate durations
        let a_duration = if let (Some(first), Some(last)) = (session_a.first(), session_a.last()) {
            last.timestamp.signed_duration_since(first.timestamp).num_milliseconds() as u64
        } else {
            0
        };

        let b_duration = if let (Some(first), Some(last)) = (session_b.first(), session_b.last()) {
            last.timestamp.signed_duration_since(first.timestamp).num_milliseconds() as u64
        } else {
            0
        };

        Self {
            events_in_a_only,
            events_in_b_only,
            different_outcomes,
            a_duration_ms: a_duration,
            b_duration_ms: b_duration,
        }
    }

    /// Returns `true` if the two sessions are identical in structure and content.
    pub fn is_identical(&self) -> bool {
        self.events_in_a_only.is_empty()
            && self.events_in_b_only.is_empty()
            && self.different_outcomes.is_empty()
    }
}
