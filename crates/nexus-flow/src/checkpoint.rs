use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use libsql::{Connection, Database, Row};
use serde_json::Value;
use tokio::time::sleep;
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use nexus_proto::workflow::{
    StepId, StepStatus, WorkflowContext, WorkflowId, WorkflowRun, WorkflowRunStatus,
};

use crate::error::FlowError;

// =============================================================================
// CheckpointStore — SQLite-Backed Workflow State Persistence
// =============================================================================

/// Persists workflow execution state to SQLite for crash recovery and resumption.
/// All operations are parameterized to prevent SQL injection and ensure thread safety.
pub struct CheckpointStore {
    db_path: String,
    conn: Connection,
}

impl CheckpointStore {
    /// Opens or creates the checkpoint database and runs schema migrations.
    #[instrument(skip(db_path), fields(path = %db_path))]
    pub async fn new(db_path: &str) -> Result<Self, FlowError> {
        debug!("initializing checkpoint store");

        let db = Database::open(db_path)
            .await
            .map_err(|e| FlowError::Database(format!("failed to open checkpoint DB: {}", e)))?;

        let conn = db
            .connect()
            .map_err(|e| FlowError::Database(format!("failed to connect to checkpoint DB: {}", e)))?;

        let store = Self {
            db_path: db_path.to_string(),
            conn,
        };

        store.run_migrations().await?;
        debug!("checkpoint migrations completed");
        Ok(store)
    }
    /// Runs inline SQL migrations to ensure tables exist.
    async fn run_migrations(&self) -> Result<(), FlowError> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    workflow_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS step_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    finished_at TEXT,
                    UNIQUE(run_id, step_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_cp_run ON step_checkpoints(run_id);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON workflow_runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_updated ON workflow_runs(updated_at);",
            )
            .await
            .map_err(|e| FlowError::Database(format!("migration failed: {}", e)))
    }

    /// Inserts a new workflow run record.
    #[instrument(skip(self, run), fields(run_id = %run.id))]
    pub async fn create_run(&self, run: &WorkflowRun) -> Result<(), FlowError> {
        let context_json = serde_json::to_string(&run.context.values)
            .map_err(|e| FlowError::Serialization(e.to_string()))?;

        self.conn
            .execute(
                "INSERT INTO workflow_runs (run_id, workflow_id, workflow_name, status, context_json, started_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                libsql::params![
                    run.id.to_string(),
                    run.workflow_id.as_uuid().to_string(),
                    run.workflow_name(), // Note: WorkflowRun doesn't have name directly; we'd pass it or use default                    status_to_string(&run.status),
                    context_json,
                    run.started_at.to_rfc3339(),
                    run.started_at.to_rfc3339(),
                ],
            )
            .await
            .map_err(|e| FlowError::Database(format!("create_run failed: {}", e)))?;

        // Initialize step statuses
        for (step_id, status) in &run.step_statuses {
            self.save_step_status(run.id, step_id, status).await?;
        }

        debug!("workflow run created");
        Ok(())
    }

    /// Updates the status of a workflow run.
    #[instrument(skip(self), fields(run_id = %run_id, status = ?status))]
    pub async fn update_run_status(&self, run_id: Uuid, status: WorkflowRunStatus) -> Result<(), FlowError> {
        self.conn
            .execute(
                "UPDATE workflow_runs SET status = ?1, updated_at = ?2 WHERE run_id = ?3",
                libsql::params![
                    status_to_string(&status),
                    Utc::now().to_rfc3339(),
                    run_id.to_string(),
                ],
            )
            .await
            .map_err(|e| FlowError::Database(format!("update_run_status failed: {}", e)))?;
        Ok(())
    }

    /// Updates the workflow context (shared variables) for a run.
    #[instrument(skip(self, ctx), fields(run_id = %run_id))]
    pub async fn update_run_context(&self, run_id: Uuid, ctx: &WorkflowContext) -> Result<(), FlowError> {
        let context_json = serde_json::to_string(&ctx.values)
            .map_err(|e| FlowError::Serialization(e.to_string()))?;

        self.conn
            .execute(
                "UPDATE workflow_runs SET context_json = ?1, updated_at = ?2 WHERE run_id = ?3",
                libsql::params![context_json, Utc::now().to_rfc3339(), run_id.to_string()],
            )
            .await
            .map_err(|e| FlowError::Database(format!("update_run_context failed: {}", e)))?;
        Ok(())
    }
    /// Saves or updates the status of a single step checkpoint.
    /// Uses upsert semantics (`INSERT OR REPLACE`) to handle retries and state transitions.
    #[instrument(skip(self, status), fields(run_id = %run_id, step_id = %step_id))]
    pub async fn save_step_status(
        &self,
        run_id: Uuid,
        step_id: &StepId,
        status: &StepStatus,
    ) -> Result<(), FlowError> {
        let (status_str, output_json, error, attempts, started_at, finished_at) =
            step_status_to_row(status);

        self.conn
            .execute(
                "INSERT OR REPLACE INTO step_checkpoints 
                 (run_id, step_id, status, output_json, error, attempts, started_at, finished_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                libsql::params![
                    run_id.to_string(),
                    step_id.as_str(),
                    status_str,
                    output_json,
                    error,
                    attempts as i64,
                    started_at,
                    finished_at,
                ],
            )
            .await
            .map_err(|e| FlowError::Database(format!("save_step_status failed: {}", e)))?;
        Ok(())
    }

    /// Loads a complete workflow run including its step statuses.
    pub async fn load_run(&self, run_id: Uuid) -> Result<Option<WorkflowRun>, FlowError> {
        let row = self
            .conn
            .query_row(
                "SELECT run_id, workflow_id, workflow_name, status, context_json, started_at, updated_at 
                 FROM workflow_runs WHERE run_id = ?1",
                libsql::params![run_id.to_string()],
                |r| r,
            )
            .await;

        match row {
            Ok(r) => {
                let run_id_str: String = r.get(0)?;
                let wf_id_str: String = r.get(1)?;                let wf_name: String = r.get(2)?; // Placeholder: real impl would store name
                let status_str: String = r.get(3)?;
                let context_json: String = r.get(4)?;
                let started_at_str: String = r.get(5)?;
                
                let run_id = Uuid::parse_str(&run_id_str).map_err(|e| FlowError::Serialization(e.to_string()))?;
                let workflow_id = WorkflowId::from(Uuid::parse_str(&wf_id_str).map_err(|e| FlowError::Serialization(e.to_string()))?);
                let status = string_to_status(&status_str)?;
                let context_values: HashMap<String, Value> = serde_json::from_str(&context_json)
                    .map_err(|e| FlowError::Serialization(e.to_string()))?;
                let started_at = parse_rfc3339(&started_at_str)?;

                let mut run = WorkflowRun::new_with_context(
                    run_id,
                    workflow_id,
                    context_values,
                    started_at,
                );
                run.status = status;
                // Note: workflow_name isn't in WorkflowRun proto; ignore or store separately if needed.

                // Load step statuses
                let steps = self.load_step_statuses(run_id).await?;
                run.step_statuses = steps;

                Ok(Some(run))
            }
            Err(libsql::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(FlowError::Database(format!("load_run failed: {}", e))),
        }
    }

    /// Loads all step statuses for a given run.
    pub async fn load_step_statuses(
        &self,
        run_id: Uuid,
    ) -> Result<HashMap<StepId, StepStatus>, FlowError> {
        let mut rows = self
            .conn
            .query(
                "SELECT step_id, status, output_json, error, attempts, started_at, finished_at 
                 FROM step_checkpoints WHERE run_id = ?1",
                libsql::params![run_id.to_string()],
            )
            .await
            .map_err(|e| FlowError::Database(format!("load_step_statuses query failed: {}", e)))?;

        let mut statuses = HashMap::new();
        while let Some(row) = rows.next().await.map_err(|e| {
            FlowError::Database(format!("load_step_statuses fetch failed: {}", e))        })? {
            let step_id_str: String = row.get(0)?;
            let step_id = StepId::new(step_id_str);
            let status = row_to_step_status(&row)?;
            statuses.insert(step_id, status);
        }

        Ok(statuses)
    }

    /// Lists workflow runs with optional filtering.
    pub async fn list_runs(
        &self,
        workflow_id: Option<WorkflowId>,
        status: Option<WorkflowRunStatus>,
    ) -> Result<Vec<WorkflowRun>, FlowError> {
        let mut query = String::from(
            "SELECT run_id, workflow_id, workflow_name, status, context_json, started_at, updated_at 
             FROM workflow_runs WHERE 1=1"
        );
        let mut params = Vec::new();

        if let Some(wf_id) = workflow_id {
            query.push_str(" AND workflow_id = ?");
            params.push(libsql::Value::from(wf_id.as_uuid().to_string()));
        }
        if let Some(st) = status {
            query.push_str(" AND status = ?");
            params.push(libsql::Value::from(status_to_string(&st)));
        }
        query.push_str(" ORDER BY started_at DESC");

        let mut rows = self
            .conn
            .query(&query, params.as_slice())
            .await
            .map_err(|e| FlowError::Database(format!("list_runs query failed: {}", e)))?;

        let mut runs = Vec::new();
        while let Some(row) = rows.next().await.map_err(|e| {
            FlowError::Database(format!("list_runs fetch failed: {}", e))
        })? {
            if let Some(run) = parse_run_row(&row)? {
                // Load steps for each run
                let run_id = run.id;
                // Note: In a high-throughput scenario, we'd batch load steps.
                // For simplicity here, we load synchronously per run.
                let steps = self.load_step_statuses(run_id).await?;
                let mut full_run = run;
                full_run.step_statuses = steps;                runs.push(full_run);
            }
        }

        Ok(runs)
    }

    /// Deletes a workflow run and all its step checkpoints.
    pub async fn delete_run(&self, run_id: Uuid) -> Result<bool, FlowError> {
        let run_id_str = run_id.to_string();
        
        self.conn
            .execute("DELETE FROM step_checkpoints WHERE run_id = ?1", libsql::params![&run_id_str])
            .await
            .map_err(|e| FlowError::Database(format!("delete_run steps failed: {}", e)))?;

        let deleted = self.conn
            .execute("DELETE FROM workflow_runs WHERE run_id = ?1", libsql::params![run_id_str])
            .await
            .map_err(|e| FlowError::Database(format!("delete_run main failed: {}", e)))?;

        Ok(deleted > 0)
    }

    /// Removes workflow runs older than the specified duration.
    /// Returns the number of runs deleted.
    pub async fn cleanup_old_runs(&self, older_than: Duration) -> Result<u64, FlowError> {
        let cutoff = Utc::now()
            .checked_sub_signed(chrono::Duration::from_std(older_than).map_err(|e| {
                FlowError::Internal(format!("duration conversion failed: {}", e))
            })?)
            .unwrap_or(Utc::now());

        let cutoff_str = cutoff.to_rfc3339();
        
        // Collect run IDs to delete from steps first (due to FK-like logic, though no strict FK here)
        let mut rows = self
            .conn
            .query(
                "SELECT run_id FROM workflow_runs WHERE updated_at < ?1",
                libsql::params![cutoff_str],
            )
            .await
            .map_err(|e| FlowError::Database(format!("cleanup_old_runs query failed: {}", e)))?;

        let mut ids = Vec::new();
        while let Some(row) = rows.next().await.map_err(|e| {
            FlowError::Database(format!("cleanup_old_runs fetch failed: {}", e))
        })? {
            let id: String = row.get(0)?;            ids.push(id);
        }

        for id in &ids {
            self.conn
                .execute("DELETE FROM step_checkpoints WHERE run_id = ?1", libsql::params![id])
                .await
                .ok(); // Ignore errors for individual step cleanup
        }

        let deleted = self.conn
            .execute("DELETE FROM workflow_runs WHERE updated_at < ?1", libsql::params![cutoff_str])
            .await
            .map_err(|e| FlowError::Database(format!("cleanup_old_runs main delete failed: {}", e)))?;

        debug!(count = deleted, "cleaned up old workflow runs");
        Ok(deleted as u64)
    }
}

// =============================================================================
// Helper: Find Resumable Workflow
// =============================================================================

/// Finds the most recent `Pending` or `Running` workflow run for a given workflow ID.
/// Useful for automatic crash recovery.
pub async fn find_resumable(
    store: &CheckpointStore,
    workflow_id: WorkflowId,
) -> Result<Option<WorkflowRun>, FlowError> {
    let wf_id_str = workflow_id.as_uuid().to_string();
    
    let row = store.conn
        .query_row(
            "SELECT run_id, workflow_id, workflow_name, status, context_json, started_at, updated_at 
             FROM workflow_runs 
             WHERE workflow_id = ?1 AND status IN ('pending', 'running')
             ORDER BY started_at DESC LIMIT 1",
            libsql::params![wf_id_str],
            |r| r,
        )
        .await;

    match row {
        Ok(r) => parse_run_row(&r),
        Err(libsql::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(FlowError::Database(format!("find_resumable failed: {}", e))),
    }
}
// =============================================================================
// Internal Mappers & Helpers
// =============================================================================

fn parse_run_row(row: &Row) -> Result<Option<WorkflowRun>, FlowError> {
    let run_id_str: String = row.get(0)?;
    let wf_id_str: String = row.get(1)?;
    let status_str: String = row.get(3)?;
    let context_json: String = row.get(4)?;
    let started_at_str: String = row.get(5)?;

    let run_id = Uuid::parse_str(&run_id_str).map_err(|e| FlowError::Serialization(e.to_string()))?;
    let workflow_id = WorkflowId::from(Uuid::parse_str(&wf_id_str).map_err(|e| FlowError::Serialization(e.to_string()))?);
    let status = string_to_status(&status_str)?;
    let context_values: HashMap<String, Value> = serde_json::from_str(&context_json)
        .map_err(|e| FlowError::Serialization(e.to_string()))?;
    let started_at = parse_rfc3339(&started_at_str)?;

    let mut run = WorkflowRun::new_with_context(run_id, workflow_id, context_values, started_at);
    run.status = status;
    Ok(Some(run))
}

fn step_status_to_row(status: &StepStatus) -> (String, Option<String>, Option<String>, i64, Option<String>, Option<String>) {
    match status {
        StepStatus::Pending => ("pending".into(), None, None, 0, None, None),
        StepStatus::Running { started_at } => (
            "running".into(),
            None,
            None,
            0,
            Some(started_at.to_rfc3339()),
            None,
        ),
        StepStatus::Completed { finished_at, output } => (
            "completed".into(),
            Some(serde_json::to_string(output).unwrap_or_default()),
            None,
            0,
            None,
            Some(finished_at.to_rfc3339()),
        ),
        StepStatus::Failed { error, attempts } => (
            "failed".into(),
            None,
            Some(error.clone()),
            *attempts as i64,
            None,
            None,
        ),        StepStatus::Skipped { reason } => (
            "skipped".into(),
            None,
            Some(reason.clone()),
            0,
            None,
            None,
        ),
    }
}

fn row_to_step_status(row: &Row) -> Result<StepStatus, FlowError> {
    let status_str: String = row.get(1)?;
    match status_str.as_str() {
        "pending" => Ok(StepStatus::Pending),
        "running" => {
            let started_at_str: Option<String> = row.get(5)?;
            let started_at = started_at_str.map(parse_rfc3339).transpose()?.unwrap_or(Utc::now());
            Ok(StepStatus::Running { started_at })
        }
        "completed" => {
            let finished_at_str: Option<String> = row.get(6)?;
            let output_str: Option<String> = row.get(2)?;
            let finished_at = finished_at_str.map(parse_rfc3339).transpose()?.unwrap_or(Utc::now());
            let output = output_str.map(|s| serde_json::from_str(&s).unwrap_or(Value::Null)).unwrap_or(Value::Null);
            Ok(StepStatus::Completed { finished_at, output })
        }
        "failed" => {
            let error: Option<String> = row.get(3)?;
            let attempts: i64 = row.get(4)?;
            Ok(StepStatus::Failed {
                error: error.unwrap_or_else(|| "unknown error".into()),
                attempts: attempts as u32,
            })
        }
        "skipped" => {
            let reason: Option<String> = row.get(3)?;
            Ok(StepStatus::Skipped {
                reason: reason.unwrap_or_else(|| "skipped".into()),
            })
        }
        other => Err(FlowError::Internal(format!("unknown step status: {}", other))),
    }
}

fn status_to_string(status: &WorkflowRunStatus) -> String {
    match status {
        WorkflowRunStatus::Pending => "pending".into(),
        WorkflowRunStatus::Running => "running".into(),
        WorkflowRunStatus::Completed => "completed".into(),        WorkflowRunStatus::Failed => "failed".into(),
        WorkflowRunStatus::Cancelled => "cancelled".into(),
    }
}

fn string_to_status(s: &str) -> Result<WorkflowRunStatus, FlowError> {
    match s {
        "pending" => Ok(WorkflowRunStatus::Pending),
        "running" => Ok(WorkflowRunStatus::Running),
        "completed" => Ok(WorkflowRunStatus::Completed),
        "failed" => Ok(WorkflowRunStatus::Failed),
        "cancelled" => Ok(WorkflowRunStatus::Cancelled),
        other => Err(FlowError::Internal(format!("unknown workflow run status: {}", other))),
    }
}

fn parse_rfc3339(s: &str) -> Result<DateTime<Utc>, FlowError> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| FlowError::Serialization(format!("invalid timestamp '{}': {}", s, e)))
}
