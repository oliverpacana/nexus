use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, NaiveDate, Utc};
use libsql::{Connection, Row, params};
use nexus_proto::agent::AgentId;
use nexus_proto::model::ProviderId;
use nexus_router::cost::{CostLedger, CostRecord};
use tracing::debug;
use uuid::Uuid;

use crate::error::ObsError;

/// Daily aggregated cost summary.
#[derive(Debug, Clone)]
pub struct DailyCostSummary {
    pub date: NaiveDate,
    pub total_usd: f64,
    pub total_calls: u64,
    pub total_tokens: u64,
}

/// Persistent SQLite-backed cost ledger that stays in sync with the in-memory ledger.
///
/// # Thread Safety
/// - `Send + Sync`
/// - SQLite connections from `libsql` are safe for concurrent async access
/// - In-memory ledger is wrapped in `Arc` for shared ownership
pub struct PersistentCostLedger {
    conn: Connection,
    in_memory: Arc<CostLedger>,
}

impl PersistentCostLedger {
    /// Creates a new persistent cost ledger and runs schema migrations.
    #[tracing::instrument(skip(db_path), fields(path = %db_path))]
    pub async fn new(db_path: &str) -> Result<Self, ObsError> {
        debug!("initializing persistent cost ledger");

        let db = libsql::Database::open(db_path)
            .await
            .map_err(|e| ObsError::Database(format!("failed to open ledger DB: {}", e)))?;

        let conn = db
            .connect()
            .map_err(|e| ObsError::Database(format!("failed to connect to ledger DB: {}", e)))?;
        let ledger = Self {
            conn,
            in_memory: Arc::new(CostLedger::new()),
        };

        ledger.run_migrations().await?;
        debug!("ledger migrations completed");
        Ok(ledger)
    }

    /// Runs inline SQL migrations to ensure the cost_records table exists.
    async fn run_migrations(&self) -> Result<(), ObsError> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS cost_records (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    workflow_run_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    actual_latency_ms INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cost_agent ON cost_records(agent_id);
                CREATE INDEX IF NOT EXISTS idx_cost_ts ON cost_records(timestamp);
                CREATE INDEX IF NOT EXISTS idx_cost_model ON cost_records(model);",
            )
            .await
            .map_err(|e| ObsError::Database(format!("migration failed: {}", e)))
    }

    /// Records a cost entry in both SQLite and the in-memory ledger.
    #[tracing::instrument(skip(self, rec), fields(id = %rec.id))]
    pub async fn record(&self, rec: CostRecord) -> Result<(), ObsError> {
        // Note: workflow_run_id is nullable in schema; CostRecord doesn't have it by default
        let workflow_run_id: Option<String> = None;

        self.conn
            .execute(
                "INSERT OR IGNORE INTO cost_records (id, agent_id, workflow_run_id, provider, model, input_tokens, output_tokens, estimated_cost_usd, actual_latency_ms, timestamp)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    rec.id.to_string(),
                    rec.agent_id.to_string(),
                    workflow_run_id,
                    rec.provider.to_string(),                    rec.model,
                    rec.input_tokens as i64,
                    rec.output_tokens as i64,
                    rec.estimated_cost_usd,
                    rec.actual_latency_ms as i64,
                    rec.timestamp.to_rfc3339(),
                ],
            )
            .await
            .map_err(|e| ObsError::Database(format!("record insert failed: {}", e)))?;

        // Sync to in-memory ledger
        self.in_memory
            .record(rec)
            .await
            .map_err(|e| ObsError::Budget(e.to_string()))?;

        Ok(())
    }

    /// Returns the total estimated cost across all recorded calls.
    pub async fn total_spent(&self) -> Result<f64, ObsError> {
        let sum: Option<f64> = self
            .conn
            .query_row("SELECT SUM(estimated_cost_usd) FROM cost_records", (), |r| r.get(0))
            .await
            .map_err(|e| ObsError::Database(e.to_string()))?;
        Ok(sum.unwrap_or(0.0))
    }

    /// Returns total spending since a specific timestamp.
    pub async fn spent_in_period(&self, since: DateTime<Utc>) -> Result<f64, ObsError> {
        let sum: Option<f64> = self
            .conn
            .query_row(
                "SELECT SUM(estimated_cost_usd) FROM cost_records WHERE timestamp >= ?1",
                params![since.to_rfc3339()],
                |r| r.get(0),
            )
            .await
            .map_err(|e| ObsError::Database(e.to_string()))?;
        Ok(sum.unwrap_or(0.0))
    }

    /// Returns aggregated spending by model name.
    pub async fn by_model(&self, since: Option<DateTime<Utc>>) -> Result<HashMap<String, f64>, ObsError> {
        self.query_grouped("model", since).await
    }

    /// Returns aggregated spending by agent ID.    pub async fn by_agent(&self, since: Option<DateTime<Utc>>) -> Result<HashMap<String, f64>, ObsError> {
        self.query_grouped("agent_id", since).await
    }

    /// Generic helper for grouped aggregation queries.
    async fn query_grouped(&self, column: &str, since: Option<DateTime<Utc>>) -> Result<HashMap<String, f64>, ObsError> {
        let query = if since.is_some() {
            format!("SELECT {col}, SUM(estimated_cost_usd) FROM cost_records WHERE timestamp >= ?1 GROUP BY {col}", col = column)
        } else {
            format!("SELECT {col}, SUM(estimated_cost_usd) FROM cost_records GROUP BY {col}", col = column)
        };

        let params_slice: &[libsql::Value] = if let Some(s) = since {
            &[s.to_rfc3339().into()]
        } else {
            &[]
        };

        let mut rows = self
            .conn
            .query(&query, params_slice)
            .await
            .map_err(|e| ObsError::Database(e.to_string()))?;

        let mut result = HashMap::new();
        while let Some(row) = rows.next().await.map_err(|e| ObsError::Database(e.to_string()))? {
            let key: String = row.get(0)?;
            let val: f64 = row.get(1)?;
            result.insert(key, val);
        }
        Ok(result)
    }

    /// Fetches the most recent cost records.
    pub async fn recent_records(&self, limit: usize) -> Result<Vec<CostRecord>, ObsError> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, agent_id, provider, model, input_tokens, output_tokens, estimated_cost_usd, actual_latency_ms, timestamp 
                 FROM cost_records ORDER BY timestamp DESC LIMIT ?1",
                params![limit as i64],
            )
            .await
            .map_err(|e| ObsError::Database(e.to_string()))?;

        let mut records = Vec::new();
        while let Some(row) = rows.next().await.map_err(|e| ObsError::Database(e.to_string()))? {
            records.push(row_to_cost_record(row)?);
        }
        Ok(records)    }

    /// Exports cost records to a CSV file.
    /// Format: id,agent_id,provider,model,input_tokens,output_tokens,cost_usd,latency_ms,timestamp
    pub async fn export_csv(
        &self,
        path: &Path,
        since: Option<DateTime<Utc>>,
    ) -> Result<usize, ObsError> {
        let query = if since.is_some() {
            "SELECT id, agent_id, provider, model, input_tokens, output_tokens, estimated_cost_usd, actual_latency_ms, timestamp FROM cost_records WHERE timestamp >= ?1 ORDER BY timestamp ASC"
        } else {
            "SELECT id, agent_id, provider, model, input_tokens, output_tokens, estimated_cost_usd, actual_latency_ms, timestamp FROM cost_records ORDER BY timestamp ASC"
        };

        let params_slice: &[libsql::Value] = if let Some(s) = since {
            &[s.to_rfc3339().into()]
        } else {
            &[]
        };

        let mut rows = self
            .conn
            .query(query, params_slice)
            .await
            .map_err(|e| ObsError::Database(e.to_string()))?;

        let mut file = File::create(path)
            .map_err(|e| ObsError::Io(format!("failed to create CSV file: {}", e)))?;

        // Write header
        writeln!(file, "id,agent_id,provider,model,input_tokens,output_tokens,cost_usd,latency_ms,timestamp")
            .map_err(|e| ObsError::Io(e.to_string()))?;

        let mut count = 0;
        while let Some(row) = rows.next().await.map_err(|e| ObsError::Database(e.to_string()))? {
            let id: String = row.get(0)?;
            let agent_id: String = row.get(1)?;
            let provider: String = row.get(2)?;
            let model: String = row.get(3)?;
            let input_tokens: i64 = row.get(4)?;
            let output_tokens: i64 = row.get(5)?;
            let cost_usd: f64 = row.get(6)?;
            let latency_ms: i64 = row.get(7)?;
            let timestamp: String = row.get(8)?;

            writeln!(
                file,
                "{},{},{},{},{},{},{:.6},{},{}",
                csv_escape(&id),                csv_escape(&agent_id),
                csv_escape(&provider),
                csv_escape(&model),
                input_tokens,
                output_tokens,
                cost_usd,
                latency_ms,
                csv_escape(&timestamp)
            )
            .map_err(|e| ObsError::Io(e.to_string()))?;
            count += 1;
        }

        file.flush().map_err(|e| ObsError::Io(e.to_string()))?;
        debug!(count, path = %path.display(), "cost ledger exported to CSV");
        Ok(count)
    }

    /// Returns daily cost summaries for the past `days` days.
    pub async fn daily_summary(&self, days: u32) -> Result<Vec<DailyCostSummary>, ObsError> {
        let cutoff = Utc::now()
            .checked_sub_signed(chrono::Duration::days(days as i64))
            .unwrap_or(Utc::now());

        let mut rows = self
            .conn
            .query(
                "SELECT DATE(timestamp), SUM(estimated_cost_usd), COUNT(*), SUM(input_tokens + output_tokens)
                 FROM cost_records WHERE timestamp >= ?1 GROUP BY DATE(timestamp) ORDER BY DATE(timestamp)",
                params![cutoff.to_rfc3339()],
            )
            .await
            .map_err(|e| ObsError::Database(e.to_string()))?;

        let mut summaries = Vec::new();
        while let Some(row) = rows.next().await.map_err(|e| ObsError::Database(e.to_string()))? {
            let date_str: String = row.get(0)?;
            let date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                .map_err(|e| ObsError::Database(format!("invalid date in DB: {}", e)))?;
            
            let total_usd: f64 = row.get(1)?;
            let total_calls: i64 = row.get(2)?;
            let total_tokens: i64 = row.get(3)?;

            summaries.push(DailyCostSummary {
                date,
                total_usd,
                total_calls: total_calls as u64,
                total_tokens: total_tokens as u64,
            });        }
        Ok(summaries)
    }

    /// Returns a reference to the synchronized in-memory ledger.
    pub fn memory_ledger(&self) -> Arc<CostLedger> {
        Arc::clone(&self.in_memory)
    }
}

/// Helper to convert a database row back to a `CostRecord`.
fn row_to_cost_record(row: Row) -> Result<CostRecord, ObsError> {
    let id_str: String = row.get(0)?;
    let agent_id_str: String = row.get(1)?;
    let provider_str: String = row.get(2)?;
    let model: String = row.get(3)?;
    let input_tokens: i64 = row.get(4)?;
    let output_tokens: i64 = row.get(5)?;
    let cost_usd: f64 = row.get(6)?;
    let latency_ms: i64 = row.get(7)?;
    let ts_str: String = row.get(8)?;

    let id = Uuid::parse_str(&id_str)
        .map_err(|e| ObsError::Database(format!("invalid UUID in ledger: {}", e)))?;
    
    let agent_id = AgentId::from(agent_id_str.as_str());
    
    // Parse ProviderId from string (basic matching; extend as needed)
    let provider = match provider_str.as_str() {
        "openai" => ProviderId::OpenAI,
        "anthropic" => ProviderId::Anthropic,
        "groq" => ProviderId::Groq,
        "mistral" => ProviderId::Mistral,
        "local" => ProviderId::Local,
        _ => ProviderId::Custom(provider_str),
    };

    let timestamp = DateTime::parse_from_rfc3339(&ts_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ObsError::Database(format!("invalid timestamp in ledger: {}", e)))?;

    Ok(CostRecord::new(
        agent_id,
        provider,
        model,
        input_tokens as u32,
        output_tokens as u32,
        cost_usd,
        latency_ms as u64,
    ))}

/// Escapes a string for safe inclusion in RFC 4180 CSV.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
