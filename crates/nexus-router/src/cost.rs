use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use nexus_proto::agent::AgentId;
use nexus_proto::model::ProviderId;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, instrument};
use uuid::Uuid;

use crate::error::RouterError;

// =============================================================================
// CostRecord — Single LLM Call Cost Tracking
// =============================================================================

/// Record of a single LLM API call with cost and performance metrics.
/// Used for cost accounting, budgeting, and observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecord {
    /// Unique identifier for this cost record.
    pub id: Uuid,

    /// Agent that initiated the request.
    pub agent_id: AgentId,

    /// Provider that served the request.
    pub provider: ProviderId,

    /// Model that generated the response.
    pub model: String,

    /// Number of tokens in the prompt/input.
    pub input_tokens: u32,

    /// Number of tokens in the completion/output.
    pub output_tokens: u32,

    /// Estimated cost in USD for this call.
    pub estimated_cost_usd: f64,

    /// Actual round-trip latency in milliseconds.
    pub actual_latency_ms: u64,

    /// Timestamp when the request was completed.
    pub timestamp: DateTime<Utc>,
}
impl CostRecord {
    /// Creates a new cost record.
    pub fn new(
        agent_id: AgentId,
        provider: ProviderId,
        model: String,
        input_tokens: u32,
        output_tokens: u32,
        estimated_cost_usd: f64,
        actual_latency_ms: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            agent_id,
            provider,
            model,
            input_tokens,
            output_tokens,
            estimated_cost_usd,
            actual_latency_ms,
            timestamp: Utc::now(),
        }
    }

    /// Returns total tokens for this call.
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

// =============================================================================
// BudgetPeriod — Spending Limit Timeframe
// =============================================================================

/// Time period for a cost budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BudgetPeriod {
    /// Budget applies to a single agent session (resets on agent restart).
    Session,
    /// Budget resets at midnight UTC each day.
    Daily,
    /// Budget resets on the 1st of each month at midnight UTC.
    Monthly,
    /// Budget is cumulative and never resets.
    Total,
}

impl BudgetPeriod {
    /// Returns a human-readable label for this period.
    pub fn label(&self) -> &'static str {        match self {
            BudgetPeriod::Session => "session",
            BudgetPeriod::Daily => "daily",
            BudgetPeriod::Monthly => "monthly",
            BudgetPeriod::Total => "total",
        }
    }
}

// =============================================================================
// CostBudget — Per-Agent Spending Limit
// =============================================================================

/// Spending limit configuration for an agent or session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBudget {
    /// Maximum allowed spending in USD for the budget period.
    pub limit_usd: f64,

    /// Amount already spent in the current period.
    pub spent_usd: f64,

    /// Time period this budget applies to.
    pub period: BudgetPeriod,
}

impl CostBudget {
    /// Creates a new budget with the given limit and period.
    pub fn new(limit_usd: f64, period: BudgetPeriod) -> Self {
        Self {
            limit_usd,
            spent_usd: 0.0,
            period,
        }
    }

    /// Returns the remaining budget in USD.
    pub fn remaining(&self) -> f64 {
        (self.limit_usd - self.spent_usd).max(0.0)
    }

    /// Returns `true` if the budget has been exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.spent_usd >= self.limit_usd
    }

    /// Attempts to spend the given amount, returning an error if it would exceed the limit.
    pub fn spend(&mut self, amount: f64) -> Result<(), RouterError> {
        if self.spent_usd + amount > self.limit_usd {
            return Err(RouterError::BudgetExceeded {                limit: self.limit_usd,
                spent: self.spent_usd,
                requested: amount,
            });
        }
        self.spent_usd += amount;
        Ok(())
    }

    /// Resets the spent amount to zero (for period rollover or manual reset).
    pub fn reset(&mut self) {
        self.spent_usd = 0.0;
    }
}

// =============================================================================
// CostSummary — Aggregated Cost Analytics
// =============================================================================

/// Summary statistics for cost tracking and reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSummary {
    /// Total spending across all agents and models.
    pub total_usd: f64,

    /// Total number of API calls recorded.
    pub total_calls: u64,

    /// Spending breakdown by provider.
    pub by_provider: HashMap<String, f64>,

    /// Spending breakdown by model.
    pub by_model: HashMap<String, f64>,

    /// Spending breakdown by agent.
    pub by_agent: HashMap<String, f64>,

    /// Average cost per API call.
    pub avg_cost_per_call: f64,

    /// Average latency across all calls in milliseconds.
    pub avg_latency_ms: f64,
}

impl Default for CostSummary {
    fn default() -> Self {
        Self {
            total_usd: 0.0,
            total_calls: 0,
            by_provider: HashMap::new(),            by_model: HashMap::new(),
            by_agent: HashMap::new(),
            avg_cost_per_call: 0.0,
            avg_latency_ms: 0.0,
        }
    }
}

// =============================================================================
// CostLedger — In-Memory Cost Tracking
// =============================================================================

/// In-memory ledger for tracking LLM API costs and budgets.
///
/// # Design
/// - Records are stored in a `Vec` protected by `RwLock` for concurrent access
/// - Budgets are tracked per-agent via `DashMap` for lock-free lookups
/// - Summary statistics are computed on-demand from records (no caching)
/// - For persistent cost tracking, use `nexus-obs` which writes to SQLite
///
/// # Thread Safety
/// - `Send + Sync`; safe for concurrent access from many Tokio tasks
/// - Read operations use shared locks; writes use exclusive locks
/// - Budget checks are atomic via `DashMap` entry API
pub struct CostLedger {
    /// Append-only log of cost records.
    records: Arc<RwLock<Vec<CostRecord>>>,

    /// Per-agent budget tracking.
    budgets: DashMap<AgentId, CostBudget>,

    /// Cached total spending (updated on each record for fast reads).
    total_spent: Arc<RwLock<f64>>,
}

impl Default for CostLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl CostLedger {
    /// Creates a new empty cost ledger.
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(Vec::new())),
            budgets: DashMap::new(),
            total_spent: Arc::new(RwLock::new(0.0)),
        }
    }
    /// Records a new cost entry and updates aggregates.
    ///
    /// # Arguments
    /// * `rec` - The `CostRecord` to store
    ///
    /// # Returns
    /// * `Ok(())` - If record stored successfully and budget (if any) not exceeded
    /// * `Err(RouterError::BudgetExceeded)` - If agent's budget would be exceeded
    #[instrument(skip(self, rec), fields(agent = %rec.agent_id, model = %rec.model, cost = rec.estimated_cost_usd))]
    pub async fn record(&self, rec: CostRecord) -> Result<(), RouterError> {
        let cost = rec.estimated_cost_usd;

        // Check budget if one is set for this agent
        if let Some(mut budget) = self.budgets.get_mut(&rec.agent_id) {
            budget.spend(cost)?;
        }

        // Store the record
        {
            let mut records = self.records.write().await;
            records.push(rec);
        }

        // Update total spent cache
        {
            let mut total = self.total_spent.write().await;
            *total += cost;
        }

        debug!(cost, "cost record stored");
        Ok(())
    }

    /// Sets or updates the cost budget for an agent.
    pub fn set_budget(&self, agent: AgentId, budget: CostBudget) {
        self.budgets.insert(agent, budget);
        debug!(agent = %agent, limit = budget.limit_usd, period = ?budget.period, "budget set");
    }

    /// Removes the budget for an agent.
    pub fn remove_budget(&self, agent: AgentId) {
        self.budgets.remove(&agent);
        debug!(agent = %agent, "budget removed");
    }

    /// Returns the total spending across all agents.
    pub async fn total_spent(&self) -> f64 {
        *self.total_spent.read().await
    }
    /// Returns total spending by a specific agent.
    pub async fn spent_by_agent(&self, agent: AgentId) -> f64 {
        self.records
            .read()
            .await
            .iter()
            .filter(|r| r.agent_id == agent)
            .map(|r| r.estimated_cost_usd)
            .sum()
    }

    /// Returns total spending on a specific model.
    pub async fn spent_by_model(&self, model: &str) -> f64 {
        self.records
            .read()
            .await
            .iter()
            .filter(|r| r.model == model)
            .map(|r| r.estimated_cost_usd)
            .sum()
    }

    /// Returns total spending via a specific provider.
    pub async fn spent_by_provider(&self, provider: ProviderId) -> f64 {
        self.records
            .read()
            .await
            .iter()
            .filter(|r| r.provider == provider)
            .map(|r| r.estimated_cost_usd)
            .sum()
    }

    /// Returns the most recent cost records for an agent.
    pub async fn records_for_agent(&self, agent: AgentId, limit: usize) -> Vec<CostRecord> {
        self.records
            .read()
            .await
            .iter()
            .filter(|r| r.agent_id == agent)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Returns all records within a time range.
    pub async fn records_in_range(
        &self,        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<CostRecord> {
        self.records
            .read()
            .await
            .iter()
            .filter(|r| r.timestamp >= start && r.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Computes aggregated cost summary statistics.
    pub async fn summary(&self) -> CostSummary {
        let records = self.records.read().await;
        let total_calls = records.len() as u64;

        if total_calls == 0 {
            return CostSummary::default();
        }

        let total_usd: f64 = records.iter().map(|r| r.estimated_cost_usd).sum();
        let total_latency: u64 = records.iter().map(|r| r.actual_latency_ms).sum();

        let mut by_provider: HashMap<String, f64> = HashMap::new();
        let mut by_model: HashMap<String, f64> = HashMap::new();
        let mut by_agent: HashMap<String, f64> = HashMap::new();

        for rec in records.iter() {
            *by_provider.entry(rec.provider.to_string()).or_insert(0.0) += rec.estimated_cost_usd;
            *by_model.entry(rec.model.clone()).or_insert(0.0) += rec.estimated_cost_usd;
            *by_agent.entry(rec.agent_id.to_string()).or_insert(0.0) += rec.estimated_cost_usd;
        }

        CostSummary {
            total_usd,
            total_calls,
            by_provider,
            by_model,
            by_agent,
            avg_cost_per_call: total_usd / total_calls as f64,
            avg_latency_ms: total_latency as f64 / total_calls as f64,
        }
    }

    /// Clears all records and resets totals (use with caution).
    pub async fn clear(&self) {
        self.records.write().await.clear();
        *self.total_spent.write().await = 0.0;
        debug!("cost ledger cleared");    }

    /// Returns the number of recorded cost entries.
    pub async fn record_count(&self) -> usize {
        self.records.read().await.len()
    }

    /// Returns a reference to the budgets map for advanced operations.
    pub fn budgets(&self) -> &DashMap<AgentId, CostBudget> {
        &self.budgets
    }
}

// =============================================================================
// Token Counting — tiktoken-rs Integration
// =============================================================================

/// Counts tokens in text using the appropriate encoding for the given model.
///
/// # Arguments
/// * `text` - The input text to tokenize
/// * `model` - Model identifier (e.g., "gpt-4o", "claude-3-sonnet", "llama-3.1-70b")
///
/// # Returns
/// Estimated token count as u32.
///
/// # Fallback Behavior
/// - OpenAI models: Use model-specific encoding (cl100k_base, p50k_base, etc.)
/// - Non-OpenAI models: Fall back to cl100k_base (most common for modern models)
/// - Unknown encodings: Fall back to character-based estimate (4 chars ≈ 1 token)
#[instrument(skip(text), fields(text_len = text.len(), model))]
pub fn count_tokens(text: &str, model: &str) -> u32 {
    // Map model names to tiktoken encoding names
    let encoding_name = model_to_encoding(model);

    // Try to get the encoding from tiktoken-rs
    match tiktoken_rs::get_bpe_from_model(encoding_name) {
        Ok(bpe) => {
            // Use the BPE encoder to count tokens
            bpe.encode_with_special_tokens(text).len() as u32
        }
        Err(_) => {
            // Fallback: try cl100k_base directly (most common for modern models)
            if let Ok(bpe) = tiktoken_rs::get_bpe_from_model("cl100k_base") {
                return bpe.encode_with_special_tokens(text).len() as u32;
            }

            // Last resort: character-based heuristic
            // Rough estimate: 4 characters ≈ 1 token for English text
            debug!(model, "using character-based token estimate");            (text.len() / 4) as u32
        }
    }
}

/// Maps model identifiers to tiktoken encoding names.
fn model_to_encoding(model: &str) -> &str {
    match model {
        // GPT-4 and GPT-3.5 Turbo use cl100k_base
        m if m.starts_with("gpt-4") || m.starts_with("gpt-3.5-turbo") => "cl100k_base",
        
        // Older models use different encodings
        "text-davinci-003" | "text-davinci-002" | "text-curie-001" 
        | "text-babbage-001" | "text-ada-001" => "p50k_base",
        
        "code-davinci-002" | "code-cushman-001" | "code-davinci-001" => "p50k_base",
        
        "text-embedding-ada-002" | "text-embedding-3-small" | "text-embedding-3-large" => "cl100k_base",
        
        // Claude models: use cl100k_base as best approximation
        m if m.starts_with("claude-") => "cl100k_base",
        
        // Llama/Mistral/Groq models: use cl100k_base
        m if m.contains("llama") || m.contains("mistral") || m.contains("mixtral") => "cl100k_base",
        
        // Gemma and other models
        m if m.contains("gemma") => "cl100k_base",
        
        // Default fallback
        _ => "cl100k_base",
    }
}

/// Estimates tokens for a ModelRequest (prompt + expected completion).
///
/// # Arguments
/// * `messages` - Vector of messages to encode
/// * `model` - Target model for encoding
/// * `expected_output_tokens` - Estimated completion length
///
/// # Returns
/// Total estimated tokens (input + output).
pub fn estimate_request_tokens(
    messages: &[nexus_proto::model::Message],
    model: &str,
    expected_output_tokens: u32,
) -> u32 {
    // Concatenate all message text for estimation
    let text: String = messages
        .iter()        .map(|m| m.text_content())
        .collect::<Vec<_>>()
        .join("\n\n");

    let input_tokens = count_tokens(&text, model);
    input_tokens + expected_output_tokens
}

/// Calculates estimated cost for a request given model pricing.
///
/// # Arguments
/// * `model` - Model identifier
/// * `input_tokens` - Prompt token count
/// * `output_tokens` - Expected completion token count
/// * `input_rate_per_1k` - Cost per 1K input tokens in USD
/// * `output_rate_per_1k` - Cost per 1K output tokens in USD
///
/// # Returns
/// Estimated cost in USD.
pub fn calculate_cost(
    input_tokens: u32,
    output_tokens: u32,
    input_rate_per_1k: f64,
    output_rate_per_1k: f64,
) -> f64 {
    // Rates are per 1K tokens
    let input_cost = (input_tokens as f64 * input_rate_per_1k) / 1000.0;
    let output_cost = (output_tokens as f64 * output_rate_per_1k) / 1000.0;
    input_cost + output_cost
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_proto::model::Message;

    #[test]
    fn test_cost_record_creation() {
        let rec = CostRecord::new(
            AgentId::new(),
            ProviderId::OpenAI,
            "gpt-4o-mini".to_string(),
            150,
            75,
            0.0015,
            450,        );

        assert_eq!(rec.total_tokens(), 225);
        assert!(!rec.id.as_uuid().is_nil());
        assert!(rec.timestamp <= Utc::now());
    }

    #[test]
    fn test_budget_spending() {
        let mut budget = CostBudget::new(10.0, BudgetPeriod::Daily);
        
        assert_eq!(budget.remaining(), 10.0);
        assert!(!budget.is_exhausted());
        
        budget.spend(3.5).unwrap();
        assert_eq!(budget.remaining(), 6.5);
        
        budget.spend(6.0).unwrap();
        assert_eq!(budget.remaining(), 0.5);
        
        // This should fail
        assert!(budget.spend(1.0).is_err());
        assert!(budget.is_exhausted());
    }

    #[test]
    fn test_budget_reset() {
        let mut budget = CostBudget::new(5.0, BudgetPeriod::Session);
        budget.spend(4.0).unwrap();
        assert_eq!(budget.remaining(), 1.0);
        
        budget.reset();
        assert_eq!(budget.remaining(), 5.0);
        assert!(!budget.is_exhausted());
    }

    #[tokio::test]
    async fn test_cost_ledger_basic() {
        let ledger = CostLedger::new();
        let agent = AgentId::new();
        
        // Record some costs
        let rec1 = CostRecord::new(
            agent,
            ProviderId::OpenAI,
            "gpt-4o".to_string(),
            100,
            50,
            0.00125,
            300,        );
        let rec2 = CostRecord::new(
            agent,
            ProviderId::Groq,
            "llama-3.1-70b".to_string(),
            200,
            100,
            0.00018,
            50,
        );
        
        ledger.record(rec1.clone()).await.unwrap();
        ledger.record(rec2.clone()).await.unwrap();
        
        // Verify totals
        assert_eq!(ledger.total_spent().await, 0.00143);
        assert_eq!(ledger.record_count().await, 2);
        
        // Verify per-agent spending
        assert_eq!(ledger.spent_by_agent(agent).await, 0.00143);
        
        // Verify per-model spending
        assert!((ledger.spent_by_model("gpt-4o").await - 0.00125).abs() < 0.00001);
        
        // Verify summary
        let summary = ledger.summary().await;
        assert_eq!(summary.total_calls, 2);
        assert!((summary.total_usd - 0.00143).abs() < 0.00001);
        assert!((summary.avg_cost_per_call - 0.000715).abs() < 0.00001);
        assert!((summary.avg_latency_ms - 175.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_ledger_budget_enforcement() {
        let ledger = CostLedger::new();
        let agent = AgentId::new();
        
        // Set a tight budget
        ledger.set_budget(agent, CostBudget::new(0.001, BudgetPeriod::Session));
        
        // This should succeed
        let rec = CostRecord::new(
            agent,
            ProviderId::OpenAI,
            "gpt-4o-mini".to_string(),
            50,
            25,
            0.00009,
            200,
        );        assert!(ledger.record(rec).await.is_ok());
        
        // This should exceed budget
        let rec2 = CostRecord::new(
            agent,
            ProviderId::OpenAI,
            "gpt-4o".to_string(),
            100,
            50,
            0.00125,
            300,
        );
        assert!(matches!(
            ledger.record(rec2).await,
            Err(RouterError::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn test_token_counting_openai() {
        // Simple English text
        let text = "Hello, world! How are you today?";
        
        // GPT-4 uses cl100k_base
        let tokens = count_tokens(text, "gpt-4o");
        assert!(tokens > 0 && tokens < 20); // Reasonable range
        
        // Same text with different model should give similar count
        let tokens2 = count_tokens(text, "gpt-3.5-turbo");
        assert_eq!(tokens, tokens2); // Same encoding
    }

    #[test]
    fn test_token_counting_fallback() {
        // Unknown model should fall back to cl100k_base
        let text = "The quick brown fox jumps over the lazy dog.";
        let tokens = count_tokens(text, "unknown-model-123");
        assert!(tokens > 0);
        
        // Very long text should not panic
        let long_text = "a".repeat(10000);
        let tokens = count_tokens(&long_text, "gpt-4");
        assert!(tokens > 1000);
    }

    #[test]
    fn test_estimate_request_tokens() {
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("What is the capital of France?"),        ];
        
        let total = estimate_request_tokens(&messages, "gpt-4o", 50);
        assert!(total > 50); // Input + expected output
    }

    #[test]
    fn test_calculate_cost() {
        // gpt-4o pricing: $0.005 input, $0.015 output per 1K tokens
        let cost = calculate_cost(1000, 500, 0.005, 0.015);
        assert!((cost - 0.0125).abs() < 0.00001); // (1000*0.005 + 500*0.015)/1000
        
        // Zero tokens = zero cost
        assert_eq!(calculate_cost(0, 0, 0.005, 0.015), 0.0);
    }

    #[test]
    fn test_model_to_encoding() {
        assert_eq!(model_to_encoding("gpt-4o"), "cl100k_base");
        assert_eq!(model_to_encoding("gpt-3.5-turbo"), "cl100k_base");
        assert_eq!(model_to_encoding("text-davinci-003"), "p50k_base");
        assert_eq!(model_to_encoding("claude-3-5-sonnet-20241022"), "cl100k_base");
        assert_eq!(model_to_encoding("llama-3.1-70b-versatile"), "cl100k_base");
        assert_eq!(model_to_encoding("unknown"), "cl100k_base");
    }

    #[tokio::test]
    async fn test_records_in_range() {
        let ledger = CostLedger::new();
        let agent = AgentId::new();
        
        // Record with known timestamp
        let mut rec = CostRecord::new(
            agent,
            ProviderId::OpenAI,
            "gpt-4o".to_string(),
            100,
            50,
            0.00125,
            300,
        );
        let test_time = Utc::now();
        rec.timestamp = test_time;
        
        ledger.record(rec).await.unwrap();
        
        // Query with range that includes the record
        let start = test_time - chrono::Duration::minutes(1);
        let end = test_time + chrono::Duration::minutes(1);
        let results = ledger.records_in_range(start, end).await;        assert_eq!(results.len(), 1);
        
        // Query with range that excludes the record
        let start = test_time + chrono::Duration::hours(1);
        let end = start + chrono::Duration::hours(1);
        let results = ledger.records_in_range(start, end).await;
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_budget_period_labels() {
        assert_eq!(BudgetPeriod::Session.label(), "session");
        assert_eq!(BudgetPeriod::Daily.label(), "daily");
        assert_eq!(BudgetPeriod::Monthly.label(), "monthly");
        assert_eq!(BudgetPeriod::Total.label(), "total");
    }
}
