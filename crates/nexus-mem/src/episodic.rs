use std::sync::Arc;

use chrono::{DateTime, Utc};
use libsql::{Database, Connection, Row, params};
use nexus_proto::agent::AgentId;
use nexus_proto::memory::{EpisodicEvent, EpisodicEventType};
use serde_json::Value;
use tokio::sync::Mutex;
use tracing::{debug, error, instrument, warn};
use uuid::Uuid;

use crate::embeddings::MemoryError;

pub type Result<T> = std::result::Result<T, MemoryError>;

// =============================================================================
// EpisodicStore — SQLite-Backed Event Log (L2 Memory)
// =============================================================================

/// L2 episodic memory: an append-only, chronological event log stored in SQLite.
///
/// # Design
/// - Uses `libsql` for async-native SQLite access with embedded/WASM support
/// - Append-only design enables reliable replay and debugging
/// - Per-agent trimming enforces storage bounds while preserving recent history
/// - Session grouping allows logical organization of agent runs
///
/// # Thread Safety
/// - `libsql::Connection` is `Send + Sync` and safe for concurrent use
/// - All public methods are `async` and use proper await points
/// - No internal mutable state requires additional locking
pub struct EpisodicStore {
    /// The libsql database handle.
    db: Database,

    /// Connection for executing queries.
    conn: Connection,

    /// Maximum number of events to retain per agent (older events trimmed).
    max_events_per_agent: usize,
}

impl EpisodicStore {
    /// Creates or opens an episodic memory store at the given path.
    ///
    /// # Arguments
    /// * `db_path` - Filesystem path to the SQLite database (or `:memory:` for testing)
    /// * `max_events_per_agent` - Maximum events to retain per agent before trimming
    ///
    /// # Returns    /// * `Ok(EpisodicStore)` - If database opened/created and migrations succeeded
    /// * `Err(MemoryError)` - If database access or migration failed
    #[instrument(skip(db_path), fields(path = %db_path))]
    pub async fn new(db_path: &str, max_events_per_agent: usize) -> Result<Self> {
        debug!("opening episodic memory database");

        let db = Database::open(db_path)
            .map_err(|e| MemoryError::ProviderError(format!("failed to open database: {}", e)))?;

        let conn = db.connect()
            .map_err(|e| MemoryError::ProviderError(format!("failed to connect to database: {}", e)))?;

        let store = Self {
            db,
            conn,
            max_events_per_agent,
        };

        // Run migrations
        store.run_migrations().await?;

        debug!(max_events = max_events_per_agent, "episodic memory initialized");
        Ok(store)
    }

    /// Runs inline schema migrations to ensure tables exist.
    async fn run_migrations(&self) -> Result<()> {
        debug!("running episodic memory migrations");

        // Events table: append-only log of agent lifecycle events
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sequence INTEGER NOT NULL
            )",
            (),
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to create events table: {}", e)))?;

        // Index for efficient per-agent history retrieval
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id, sequence)",
            (),
        )        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to create agent index: {}", e)))?;

        // Index for session-based queries
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, sequence)",
            (),
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to create session index: {}", e)))?;

        // Sessions table: metadata about agent execution runs
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                metadata TEXT
            )",
            (),
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to create sessions table: {}", e)))?;

        // Index for session lookups by agent
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id, started_at)",
            (),
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to create sessions index: {}", e)))?;

        debug!("migrations completed");
        Ok(())
    }

    /// Appends an event to the episodic log.
    ///
    /// If the agent already has `max_events_per_agent` events, the oldest event
    /// is deleted first to maintain the size bound.
    ///
    /// # Arguments
    /// * `event` - The `EpisodicEvent` to record
    ///
    /// # Errors
    /// - `MemoryError::ProviderError` if database write fails
    #[instrument(skip(self, event), fields(agent_id = %event.agent_id, event_type = ?event.event_type))]
    pub async fn append(&self, event: EpisodicEvent) -> Result<()> {
        let (id, agent_id, session_id, event_type, payload, timestamp, sequence) =            event_to_row(&event);

        // Begin transaction for atomic append + trim
        let tx = self.conn.transaction()
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to begin transaction: {}", e)))?;

        // Insert the new event
        tx.execute(
            "INSERT INTO events (id, agent_id, session_id, event_type, payload, timestamp, sequence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![id, agent_id, session_id, event_type, payload, timestamp, sequence],
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to insert event: {}", e)))?;

        // Trim if exceeding limit
        let count: i64 = tx.query_row(
            "SELECT COUNT(*) FROM events WHERE agent_id = ?1",
            params![agent_id],
            |row| row.get(0),
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to count events: {}", e)))?;

        if count as usize > self.max_events_per_agent {
            let to_delete = count as usize - self.max_events_per_agent + 1;
            let deleted = tx.execute(
                "DELETE FROM events WHERE agent_id = ?1 AND id IN (
                    SELECT id FROM events 
                    WHERE agent_id = ?1 
                    ORDER BY sequence ASC 
                    LIMIT ?2
                )",
                params![agent_id, to_delete],
            )
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to trim events: {}", e)))?;

            debug!(agent_id = %event.agent_id, deleted, "trimmed old episodic events");
        }

        tx.commit()
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to commit transaction: {}", e)))?;

        debug!(agent_id = %event.agent_id, sequence = event.sequence, "event appended");
        Ok(())
    }
    /// Retrieves the most recent events for an agent, with optional pagination.
    ///
    /// # Arguments
    /// * `agent_id` - The agent whose history to retrieve
    /// * `limit` - Maximum number of events to return
    /// * `before_sequence` - If provided, only return events with sequence < this value
    ///
    /// # Returns
    /// Events in descending sequence order (newest first).
    #[instrument(skip(self), fields(agent_id = %agent_id, limit))]
    pub async fn get_agent_history(
        &self,
        agent_id: AgentId,
        limit: usize,
        before_sequence: Option<u64>,
    ) -> Result<Vec<EpisodicEvent>> {
        let agent_id_str = agent_id.to_string();

        let query = if let Some(before) = before_sequence {
            "SELECT id, agent_id, session_id, event_type, payload, timestamp, sequence 
             FROM events 
             WHERE agent_id = ?1 AND sequence < ?2 
             ORDER BY sequence DESC 
             LIMIT ?3"
        } else {
            "SELECT id, agent_id, session_id, event_type, payload, timestamp, sequence 
             FROM events 
             WHERE agent_id = ?1 
             ORDER BY sequence DESC 
             LIMIT ?2"
        };

        let params = if let Some(before) = before_sequence {
            params![agent_id_str, before as i64, limit as i64]
        } else {
            params![agent_id_str, limit as i64]
        };

        let mut rows = self.conn.query(query, params)
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to query history: {}", e)))?;

        let mut events = Vec::with_capacity(limit);
        while let Some(row) = rows.next().await.map_err(|e| {
            MemoryError::ProviderError(format!("failed to fetch row: {}", e))
        })? {
            events.push(row_to_event(row)?);
        }

        // Return in chronological order (oldest first) for easier consumption        events.reverse();

        debug!(agent_id = %agent_id, count = events.len(), "retrieved agent history");
        Ok(events)
    }

    /// Retrieves all events belonging to a specific session.
    #[instrument(skip(self), fields(session_id = %session_id))]
    pub async fn get_session(&self, session_id: Uuid) -> Result<Vec<EpisodicEvent>> {
        let session_id_str = session_id.to_string();

        let mut rows = self.conn.query(
            "SELECT id, agent_id, session_id, event_type, payload, timestamp, sequence 
             FROM events 
             WHERE session_id = ?1 
             ORDER BY sequence ASC",
            params![session_id_str],
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to query session: {}", e)))?;

        let mut events = Vec::new();
        while let Some(row) = rows.next().await.map_err(|e| {
            MemoryError::ProviderError(format!("failed to fetch row: {}", e))
        })? {
            events.push(row_to_event(row)?);
        }

        debug!(session_id = %session_id, count = events.len(), "retrieved session events");
        Ok(events)
    }

    /// Returns the total number of events recorded for an agent.
    #[instrument(skip(self), fields(agent_id = %agent_id))]
    pub async fn count_events(&self, agent_id: AgentId) -> Result<u64> {
        let agent_id_str = agent_id.to_string();

        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM events WHERE agent_id = ?1",
            params![agent_id_str],
            |row| row.get(0),
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to count events: {}", e)))?;

        Ok(count as u64)
    }

    /// Starts a new session for an agent.
    ///    /// # Arguments
    /// * `agent_id` - The agent starting the session
    /// * `metadata` - Optional JSON metadata about the session (e.g., task description)
    ///
    /// # Returns
    /// The unique session ID (UUID v4).
    #[instrument(skip(self, metadata), fields(agent_id = %agent_id))]
    pub async fn start_session(
        &self,
        agent_id: AgentId,
        metadata: Option<Value>,
    ) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        let agent_id_str = agent_id.to_string();
        let session_id_str = session_id.to_string();
        let started_at = Utc::now().to_rfc3339();
        let metadata_json = metadata.map(|m| m.to_string()).unwrap_or_else(|| "null".to_string());

        self.conn.execute(
            "INSERT INTO sessions (id, agent_id, started_at, ended_at, metadata)
             VALUES (?1, ?2, ?3, NULL, ?4)",
            params![session_id_str, agent_id_str, started_at, metadata_json],
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to start session: {}", e)))?;

        debug!(agent_id = %agent_id, session_id = %session_id, "session started");
        Ok(session_id)
    }

    /// Marks a session as ended.
    #[instrument(skip(self), fields(session_id = %session_id))]
    pub async fn end_session(&self, session_id: Uuid) -> Result<()> {
        let session_id_str = session_id.to_string();
        let ended_at = Utc::now().to_rfc3339();

        let rows = self.conn.execute(
            "UPDATE sessions SET ended_at = ?1 WHERE id = ?2",
            params![ended_at, session_id_str],
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to end session: {}", e)))?;

        if rows == 0 {
            warn!(session_id = %session_id, "attempted to end unknown session");
            return Err(MemoryError::ProviderError("session not found".into()));
        }

        debug!(session_id = %session_id, "session ended");
        Ok(())    }

    /// Deletes all events for an agent.
    ///
    /// # Returns
    /// The number of events deleted.
    #[instrument(skip(self), fields(agent_id = %agent_id))]
    pub async fn delete_agent_history(&self, agent_id: AgentId) -> Result<u64> {
        let agent_id_str = agent_id.to_string();

        let deleted = self.conn.execute(
            "DELETE FROM events WHERE agent_id = ?1",
            params![agent_id_str],
        )
        .await
        .map_err(|e| MemoryError::ProviderError(format!("failed to delete history: {}", e)))?;

        debug!(agent_id = %agent_id, deleted, "deleted agent history");
        Ok(deleted as u64)
    }

    /// Searches events with optional filters.
    ///
    /// # Arguments
    /// * `agent_id` - Filter by agent (None = all agents)
    /// * `event_type` - Filter by event type (None = all types)
    /// * `since` - Only return events at or after this timestamp
    /// * `until` - Only return events at or before this timestamp
    /// * `limit` - Maximum results to return
    ///
    /// # Returns
    /// Events matching all provided filters, in chronological order.
    #[instrument(skip(self), fields(limit))]
    pub async fn search_events(
        &self,
        agent_id: Option<AgentId>,
        event_type: Option<EpisodicEventType>,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<EpisodicEvent>> {
        let mut conditions = Vec::new();
        let mut param_values: Vec<libsql::Value> = Vec::new();

        if let Some(aid) = agent_id {
            conditions.push("agent_id = ?");
            param_values.push(aid.to_string().into());
        }

        if let Some(et) = event_type {            conditions.push("event_type = ?");
            param_values.push(event_type_to_string(&et).into());
        }

        if let Some(s) = since {
            conditions.push("timestamp >= ?");
            param_values.push(s.to_rfc3339().into());
        }

        if let Some(u) = until {
            conditions.push("timestamp <= ?");
            param_values.push(u.to_rfc3339().into());
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let query = format!(
            "SELECT id, agent_id, session_id, event_type, payload, timestamp, sequence 
             FROM events 
             {} 
             ORDER BY sequence ASC 
             LIMIT ?",
            where_clause
        );

        param_values.push((limit as i64).into());

        let mut rows = self.conn.query(&query, param_values.as_slice())
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to search events: {}", e)))?;

        let mut events = Vec::new();
        while let Some(row) = rows.next().await.map_err(|e| {
            MemoryError::ProviderError(format!("failed to fetch row: {}", e))
        })? {
            events.push(row_to_event(row)?);
        }

        debug!(count = events.len(), ?agent_id, ?event_type, "searched episodic events");
        Ok(events)
    }

    /// Trims an agent's history to keep only the most recent `keep_last` events.
    ///
    /// # Returns
    /// The number of events deleted.    #[instrument(skip(self), fields(agent_id = %agent_id, keep_last))]
    pub async fn trim_agent_history(&self, agent_id: AgentId, keep_last: usize) -> Result<u64> {
        let agent_id_str = agent_id.to_string();

        // Get the sequence number of the event to keep as the new oldest
        let cutoff_seq: Option<i64> = self.conn.query_row(
            "SELECT sequence FROM events 
             WHERE agent_id = ?1 
             ORDER BY sequence DESC 
             LIMIT 1 OFFSET ?2",
            params![agent_id_str, keep_last as i64],
            |row| row.get(0),
        )
        .await
        .ok(); // None if fewer than keep_last events exist

        let deleted = if let Some(cutoff) = cutoff_seq {
            self.conn.execute(
                "DELETE FROM events WHERE agent_id = ?1 AND sequence <= ?2",
                params![agent_id_str, cutoff],
            )
            .await
            .map_err(|e| MemoryError::ProviderError(format!("failed to trim: {}", e)))?
        } else {
            0 // Nothing to delete
        };

        debug!(agent_id = %agent_id, deleted, keep_last, "trimmed agent history");
        Ok(deleted as u64)
    }

    /// Returns the configured maximum events per agent.
    pub fn max_events_per_agent(&self) -> usize {
        self.max_events_per_agent
    }

    /// Returns the underlying libsql connection for advanced queries.
    ///
    /// ⚠️ Use with caution: direct SQL execution bypasses the type-safe API.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
}

// =============================================================================
// Serialization Helpers
// =============================================================================

/// Converts an `EpisodicEvent` into a tuple of values for database insertion.
/// All fields are serialized to strings for SQLite TEXT storage.#[inline]
fn event_to_row(event: &EpisodicEvent) -> (String, String, String, String, String, String, i64) {
    (
        event.id.to_string(),
        event.agent_id.to_string(),
        event.session_id.to_string(),
        event_type_to_string(&event.event_type),
        serde_json::to_string(&event.payload).unwrap_or_else(|_| "null".to_string()),
        event.timestamp.to_rfc3339(),
        event.sequence as i64,
    )
}

/// Converts an `EpisodicEventType` to its canonical string representation.
#[inline]
fn event_type_to_string(event_type: &EpisodicEventType) -> String {
    match event_type {
        EpisodicEventType::AgentStarted => "agent_started",
        EpisodicEventType::AgentFinished => "agent_finished",
        EpisodicEventType::ToolCalled => "tool_called",
        EpisodicEventType::ToolResult => "tool_result",
        EpisodicEventType::ModelRequest => "model_request",
        EpisodicEventType::ModelResponse => "model_response",
        EpisodicEventType::MemoryWrite => "memory_write",
        EpisodicEventType::MemoryRead => "memory_read",
        EpisodicEventType::WorkflowStep => "workflow_step",
        EpisodicEventType::CustomEvent(s) => return format!("custom:{}", s),
    }
    .to_string()
}

/// Parses an `EpisodicEventType` from its string representation.
#[inline]
fn event_type_from_string(s: &str) -> Option<EpisodicEventType> {
    match s {
        "agent_started" => Some(EpisodicEventType::AgentStarted),
        "agent_finished" => Some(EpisodicEventType::AgentFinished),
        "tool_called" => Some(EpisodicEventType::ToolCalled),
        "tool_result" => Some(EpisodicEventType::ToolResult),
        "model_request" => Some(EpisodicEventType::ModelRequest),
        "model_response" => Some(EpisodicEventType::ModelResponse),
        "memory_write" => Some(EpisodicEventType::MemoryWrite),
        "memory_read" => Some(EpisodicEventType::MemoryRead),
        "workflow_step" => Some(EpisodicEventType::WorkflowStep),
        s if s.starts_with("custom:") => Some(EpisodicEventType::CustomEvent(s[7..].to_string())),
        _ => None,
    }
}

/// Converts a `libsql::Row` into an `EpisodicEvent`./// Handles JSON parsing and timestamp conversion.
fn row_to_event(row: Row) -> Result<EpisodicEvent> {
    let id: String = row.get(0)
        .map_err(|e| MemoryError::ProviderError(format!("failed to read id: {}", e)))?;
    let agent_id: String = row.get(1)
        .map_err(|e| MemoryError::ProviderError(format!("failed to read agent_id: {}", e)))?;
    let session_id: String = row.get(2)
        .map_err(|e| MemoryError::ProviderError(format!("failed to read session_id: {}", e)))?;
    let event_type_str: String = row.get(3)
        .map_err(|e| MemoryError::ProviderError(format!("failed to read event_type: {}", e)))?;
    let payload_str: String = row.get(4)
        .map_err(|e| MemoryError::ProviderError(format!("failed to read payload: {}", e)))?;
    let timestamp_str: String = row.get(5)
        .map_err(|e| MemoryError::ProviderError(format!("failed to read timestamp: {}", e)))?;
    let sequence: i64 = row.get(6)
        .map_err(|e| MemoryError::ProviderError(format!("failed to read sequence: {}", e)))?;

    let id = Uuid::parse_str(&id)
        .map_err(|e| MemoryError::ProviderError(format!("invalid event id: {}", e)))?;
    let agent_id = AgentId::from(agent_id.as_str());
    let session_id = Uuid::parse_str(&session_id)
        .map_err(|e| MemoryError::ProviderError(format!("invalid session id: {}", e)))?;
    let event_type = event_type_from_string(&event_type_str)
        .ok_or_else(|| MemoryError::ProviderError(format!("unknown event type: {}", event_type_str)))?;
    let payload: Value = serde_json::from_str(&payload_str)
        .map_err(|e| MemoryError::ProviderError(format!("failed to parse payload JSON: {}", e)))?;
    let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| MemoryError::ProviderError(format!("failed to parse timestamp: {}", e)))?;

    Ok(EpisodicEvent {
        id,
        agent_id,
        event_type,
        payload,
        timestamp,
        session_id,
        sequence: sequence as u64,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn test_agent_id() -> AgentId {
        AgentId::new()
    }

    fn test_event(agent_id: AgentId, session_id: Uuid, sequence: u64) -> EpisodicEvent {
        EpisodicEvent::new(
            agent_id,
            EpisodicEventType::ToolCalled,
            json!({"tool": "test", "args": {"query": "hello"}}),
            session_id,
            sequence,
        )
    }

    #[tokio::test]
    async fn test_episodic_store_basic() {
        let store = EpisodicStore::new(":memory:", 100).await.unwrap();
        let agent_id = test_agent_id();
        let session_id = store.start_session(agent_id, None).await.unwrap();

        // Append an event
        let event = test_event(agent_id, session_id, 0);
        store.append(event.clone()).await.unwrap();

        // Verify count
        assert_eq!(store.count_events(agent_id).await.unwrap(), 1);

        // Retrieve history
        let history = store.get_agent_history(agent_id, 10, None).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].id, event.id);
        assert_eq!(history[0].sequence, 0);

        // End session
        store.end_session(session_id).await.unwrap();
    }

    #[tokio::test]
    async fn test_trim_on_append() {
        let store = EpisodicStore::new(":memory:", 5).await.unwrap();
        let agent_id = test_agent_id();
        let session_id = store.start_session(agent_id, None).await.unwrap();

        // Append 7 events (max is 5, so oldest should be trimmed)
        for seq in 0..7 {
            let event = test_event(agent_id, session_id, seq);
            store.append(event).await.unwrap();
        }

        // Should only have 5 events        assert_eq!(store.count_events(agent_id).await.unwrap(), 5);

        // History should contain sequences 2-6 (0 and 1 trimmed)
        let history = store.get_agent_history(agent_id, 10, None).await.unwrap();
        assert_eq!(history.len(), 5);
        assert_eq!(history[0].sequence, 2);
        assert_eq!(history[4].sequence, 6);
    }

    #[tokio::test]
    async fn test_pagination() {
        let store = EpisodicStore::new(":memory:", 100).await.unwrap();
        let agent_id = test_agent_id();
        let session_id = store.start_session(agent_id, None).await.unwrap();

        // Append 10 events
        for seq in 0..10 {
            store.append(test_event(agent_id, session_id, seq)).await.unwrap();
        }

        // Get first page (limit 3)
        let page1 = store.get_agent_history(agent_id, 3, None).await.unwrap();
        assert_eq!(page1.len(), 3);
        assert_eq!(page1[0].sequence, 0);
        assert_eq!(page1[2].sequence, 2);

        // Get next page using before_sequence
        let last_seq = page1.last().unwrap().sequence;
        let page2 = store.get_agent_history(agent_id, 3, Some(last_seq)).await.unwrap();
        assert_eq!(page2.len(), 3);
        assert_eq!(page2[0].sequence, 3);
        assert_eq!(page2[2].sequence, 5);
    }

    #[tokio::test]
    async fn test_search_filters() {
        let store = EpisodicStore::new(":memory:", 100).await.unwrap();
        let agent1 = test_agent_id();
        let agent2 = test_agent_id();
        let session1 = store.start_session(agent1, None).await.unwrap();
        let session2 = store.start_session(agent2, None).await.unwrap();

        // Append mixed events
        store.append(EpisodicEvent::new(
            agent1,
            EpisodicEventType::ToolCalled,
            json!({}),
            session1,
            0,
        )).await.unwrap();
        store.append(EpisodicEvent::new(
            agent1,
            EpisodicEventType::ModelResponse,
            json!({}),
            session1,
            1,
        )).await.unwrap();

        store.append(EpisodicEvent::new(
            agent2,
            EpisodicEventType::ToolCalled,
            json!({}),
            session2,
            0,
        )).await.unwrap();

        // Filter by agent
        let results = store.search_events(
            Some(agent1),
            None,
            None,
            None,
            10,
        ).await.unwrap();
        assert_eq!(results.len(), 2);

        // Filter by event type
        let results = store.search_events(
            None,
            Some(EpisodicEventType::ToolCalled),
            None,
            None,
            10,
        ).await.unwrap();
        assert_eq!(results.len(), 2);

        // Filter by both
        let results = store.search_events(
            Some(agent1),
            Some(EpisodicEventType::ModelResponse),
            None,
            None,
            10,
        ).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_session_management() {        let store = EpisodicStore::new(":memory:", 100).await.unwrap();
        let agent_id = test_agent_id();

        // Start session with metadata
        let meta = json!({"task": "research", "priority": "high"});
        let session_id = store.start_session(agent_id, Some(meta.clone())).await.unwrap();

        // Append event to session
        store.append(test_event(agent_id, session_id, 0)).await.unwrap();

        // Retrieve session events
        let events = store.get_session(session_id).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].session_id, session_id);

        // End session
        store.end_session(session_id).await.unwrap();

        // Verify session can still be queried after ending
        let events = store.get_session(session_id).await.unwrap();
        assert_eq!(events.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_and_trim() {
        let store = EpisodicStore::new(":memory:", 100).await.unwrap();
        let agent_id = test_agent_id();
        let session_id = store.start_session(agent_id, None).await.unwrap();

        // Append 10 events
        for seq in 0..10 {
            store.append(test_event(agent_id, session_id, seq)).await.unwrap();
        }
        assert_eq!(store.count_events(agent_id).await.unwrap(), 10);

        // Trim to keep last 3
        let deleted = store.trim_agent_history(agent_id, 3).await.unwrap();
        assert_eq!(deleted, 7);
        assert_eq!(store.count_events(agent_id).await.unwrap(), 3);

        // Remaining events should be sequences 7, 8, 9
        let history = store.get_agent_history(agent_id, 10, None).await.unwrap();
        assert_eq!(history[0].sequence, 7);
        assert_eq!(history[2].sequence, 9);

        // Delete all
        let deleted = store.delete_agent_history(agent_id).await.unwrap();
        assert_eq!(deleted, 3);
        assert_eq!(store.count_events(agent_id).await.unwrap(), 0);
    }
    #[test]
    fn test_event_type_serialization() {
        use EpisodicEventType::*;

        assert_eq!(event_type_to_string(&AgentStarted), "agent_started");
        assert_eq!(event_type_to_string(&CustomEvent("foo".into())), "custom:foo");

        assert_eq!(event_type_from_string("agent_started"), Some(AgentStarted));
        assert_eq!(event_type_from_string("custom:bar"), Some(CustomEvent("bar".into())));
        assert_eq!(event_type_from_string("unknown"), None);
    }

    #[test]
    fn test_event_row_roundtrip() {
        let agent_id = test_agent_id();
        let session_id = Uuid::new_v4();
        let event = EpisodicEvent::new(
            agent_id,
            EpisodicEventType::MemoryWrite,
            json!({"key": "test", "value": 42}),
            session_id,
            123,
        );

        let row = event_to_row(&event);
        // Simulate a libsql Row by reconstructing from the tuple
        // (In real tests, libsql would provide the Row)

        // Verify key fields survive roundtrip
        assert_eq!(row.0, event.id.to_string());
        assert_eq!(row.1, event.agent_id.to_string());
        assert_eq!(row.2, event.session_id.to_string());
        assert_eq!(row.3, "memory_write");
        assert!(row.4.contains("test"));
        assert_eq!(row.6, 123);
    }
}
