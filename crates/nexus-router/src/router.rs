// crates/nexus-router/src/router.rs

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use dashmap::DashMap;
use futures::stream::{Stream, StreamExt};
use nexus_proto::agent::AgentId;
use nexus_proto::model::{ModelRequest, ModelResponse, ProviderId, RoutingPolicy, Token};
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, instrument, warn};

use crate::cost::{calculate_cost, count_tokens, CostLedger};
use crate::error::RouterError;
use crate::policy::ProviderHealth;
use crate::providers::{ModelProvider, ProviderRegistry};

// =============================================================================
// Router Configuration
// =============================================================================

/// Configuration for the model routing engine.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Default routing policy when request doesn't specify one.
    pub default_policy: RoutingPolicy,

    /// Maximum time to wait for a model response before timing out.
    pub request_timeout_secs: u64,

    /// Maximum retry attempts for transient provider failures.
    pub max_retries: u32,

    /// Initial backoff delay in milliseconds for exponential retry.
    pub retry_backoff_ms: u64,

    /// Buffer size for streaming token channels.
    pub stream_buffer_size: usize,

    /// Whether to enforce agent cost budgets before executing requests.
    pub budget_enforcement: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            default_policy: RoutingPolicy::CostOptimized {                max_latency_ms: 5000,
            },
            request_timeout_secs: 120,
            max_retries: 3,
            retry_backoff_ms: 500,
            stream_buffer_size: 32,
            budget_enforcement: true,
        }
    }
}

// =============================================================================
// ModelRouter — Top-Level Routing Engine
// =============================================================================

/// The central model routing engine for the Nexus runtime.
///
/// # Responsibilities
/// - Provider selection based on routing policy, cost, latency, and capabilities
/// - Request execution with timeout, retry, and budget enforcement
/// - Streaming response handling with cost tracking
/// - Health monitoring and caching for provider selection
///
/// # Thread Safety
/// - `Send + Sync`; safe for concurrent access from many Tokio tasks
/// - Provider registry and cost ledger use `Arc` for shared ownership
/// - Health cache uses `DashMap` for lock-free concurrent reads/writes
pub struct ModelRouter {
    /// Registry of available model providers.
    pub providers: Arc<ProviderRegistry>,

    /// Cost tracking and budget enforcement ledger.
    pub cost_ledger: Arc<CostLedger>,

    /// Runtime configuration.
    pub config: RouterConfig,

    /// Cached health status of providers (provider_id → (health, last_check)).
    health_cache: DashMap<ProviderId, (ProviderHealth, Instant)>,

    /// Cancellation token for background health refresh task.
    _cancel: CancellationToken,
}

impl ModelRouter {
    /// Creates a new model router with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Routing configuration including policies and limits
    /// * `providers` - Registry of available model providers    /// * `cost_ledger` - Ledger for cost tracking and budget enforcement
    ///
    /// # Returns
    /// A new `ModelRouter` instance with health monitoring started.
    #[instrument(skip(providers, cost_ledger))]
    pub async fn new(
        config: RouterConfig,
        providers: Arc<ProviderRegistry>,
        cost_ledger: Arc<CostLedger>,
    ) -> Self {
        let router = Self {
            providers,
            cost_ledger,
            config,
            health_cache: DashMap::new(),
            _cancel: CancellationToken::new(),
        };

        // Initialize health cache with initial checks
        router.refresh_health().await;

        // Spawn background health refresh task
        let router_clone = router.clone_shallow();
        let cancel = router_clone._cancel.clone();
        tokio::spawn(async move {
            router_clone.health_refresh_loop(cancel).await;
        });

        info!("model router initialized with {} providers", router.providers.available_ids().len());
        router
    }

    /// Creates a shallow clone for background task use.
    fn clone_shallow(&self) -> Self {
        Self {
            providers: Arc::clone(&self.providers),
            cost_ledger: Arc::clone(&self.cost_ledger),
            config: self.config.clone(),
            health_cache: DashMap::new(), // Fresh cache for background task
            _cancel: self._cancel.clone(),
        }
    }

    /// Selects the best provider for a request based on routing policy.
    ///
    /// # Routing Logic by Policy
    /// - `CostOptimized`: Filter healthy providers by max_latency, sort by cost, return cheapest
    /// - `LatencyOptimized`: Filter by max_cost, sort by cached latency, return fastest
    /// - `CapabilityFirst`: Filter by context_tokens and vision requirements
    /// - `LocalFirst`: Try Local provider first; fallback to cheapest cloud if Down    /// - `Pinned`: Return specific provider; error if not found or Down
    ///
    /// # Returns
    /// * `Ok(Arc<dyn ModelProvider>)` - Selected provider ready to handle request
    /// * `Err(RouterError)` - If no suitable provider found or all are Down
    #[instrument(skip(self, request), fields(policy = ?request.routing_policy))]
    pub async fn select_provider(
        &self,
        request: &ModelRequest,
    ) -> Result<Arc<dyn ModelProvider + Send + Sync>, RouterError> {
        let policy = request.routing_policy.clone();
        let providers = self.providers.all();

        match policy {
            RoutingPolicy::CostOptimized { max_latency_ms } => {
                self.select_cost_optimized(&providers, max_latency_ms).await
            }
            RoutingPolicy::LatencyOptimized { max_cost_per_1k_tokens } => {
                self.select_latency_optimized(&providers, max_cost_per_1k_tokens).await
            }
            RoutingPolicy::CapabilityFirst {
                required_context_tokens,
                requires_vision,
            } => self.select_capability_first(&providers, required_context_tokens, requires_vision),
            RoutingPolicy::LocalFirst { cloud_fallback } => {
                self.select_local_first(&providers, cloud_fallback).await
            }
            RoutingPolicy::Pinned(model_id) => self.select_pinned(&providers, &model_id),
        }
    }

    /// Cost-optimized selection: cheapest provider within latency budget.
    async fn select_cost_optimized(
        &self,
        providers: &[Arc<dyn ModelProvider + Send + Sync>],
        max_latency_ms: u64,
    ) -> Result<Arc<dyn ModelProvider + Send + Sync>, RouterError> {
        let mut candidates = Vec::new();

        for provider in providers {
            // Check health cache
            let health = self.get_cached_health(&provider.provider_id());
            if !health.is_operational() {
                continue;
            }

            // Filter by latency constraint (use cached latency if available)
            if let ProviderHealth::Degraded { latency_ms } = health {
                if latency_ms > max_latency_ms {
                    continue;                }
            }

            // Estimate cost for a typical request (100 input, 50 output tokens)
            let model = provider.available_models().first()
                .cloned()
                .unwrap_or_else(|| "default".to_string());
            let cost = provider.estimate_cost(&model, 100, 50);

            candidates.push((provider.clone(), cost));
        }

        if candidates.is_empty() {
            return Err(RouterError::NoAvailableProviders);
        }

        // Sort by cost ascending
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        debug!(
            selected = %candidates[0].0.provider_id(),
            cost = candidates[0].1,
            "selected cost-optimized provider"
        );
        Ok(candidates[0].0.clone())
    }

    /// Latency-optimized selection: fastest provider within cost budget.
    async fn select_latency_optimized(
        &self,
        providers: &[Arc<dyn ModelProvider + Send + Sync>],
        max_cost_per_1k: f64,
    ) -> Result<Arc<dyn ModelProvider + Send + Sync>, RouterError> {
        let mut candidates = Vec::new();

        for provider in providers {
            let health = self.get_cached_health(&provider.provider_id());
            if !health.is_operational() {
                continue;
            }

            // Get latency from health cache
            let latency = match health {
                ProviderHealth::Healthy => 100, // Assume good latency for healthy
                ProviderHealth::Degraded { latency_ms } => latency_ms,
                ProviderHealth::Down { .. } => continue,
            };

            // Estimate cost and filter by budget
            let model = provider.available_models().first()                .cloned()
                .unwrap_or_else(|| "default".to_string());
            let cost = provider.estimate_cost(&model, 100, 50);

            // Convert per-1K cost to per-request estimate for comparison
            let estimated_request_cost = cost * 0.15; // ~150 tokens typical request
            if estimated_request_cost > max_cost_per_1k {
                continue;
            }

            candidates.push((provider.clone(), latency));
        }

        if candidates.is_empty() {
            return Err(RouterError::NoAvailableProviders);
        }

        // Sort by latency ascending
        candidates.sort_by_key(|(_, lat)| *lat);

        debug!(
            selected = %candidates[0].0.provider_id(),
            latency_ms = candidates[0].1,
            "selected latency-optimized provider"
        );
        Ok(candidates[0].0.clone())
    }

    /// Capability-first selection: filter by context size and vision support.
    fn select_capability_first(
        &self,
        providers: &[Arc<dyn ModelProvider + Send + Sync>],
        required_context: Option<u32>,
        requires_vision: bool,
    ) -> Result<Arc<dyn ModelProvider + Send + Sync>, RouterError> {
        for provider in providers {
            let health = self.get_cached_health(&provider.provider_id());
            if !health.is_operational() {
                continue;
            }

            // Check context window requirement
            if let Some(required) = required_context {
                let model = provider.available_models().first()
                    .cloned()
                    .unwrap_or_else(|| "default".to_string());
                if provider.max_context_tokens(&model) < required as usize {
                    continue;
                }
            }
            // Vision support: check if provider supports it (simplified check)
            if requires_vision && !provider.supports_tool_calls() {
                // Most vision-capable models also support tool calls; use as proxy
                // In production, add explicit vision capability check
                continue;
            }

            debug!(
                selected = %provider.provider_id(),
                "selected capability-first provider"
            );
            return Ok(provider.clone());
        }

        Err(RouterError::NoAvailableProviders)
    }

    /// Local-first selection: prefer local provider, fallback to cloud.
    async fn select_local_first(
        &self,
        providers: &[Arc<dyn ModelProvider + Send + Sync>],
        cloud_fallback: bool,
    ) -> Result<Arc<dyn ModelProvider + Send + Sync>, RouterError> {
        // Try to find Local provider first
        if let Some(local) = self.providers.get(&ProviderId::Local) {
            let health = self.get_cached_health(&ProviderId::Local);
            if health.is_operational() {
                debug!("selected local provider (LocalFirst policy)");
                return Ok(local);
            }
        }

        // Fallback to cloud if enabled
        if cloud_fallback {
            // Use cost-optimized selection for cloud fallback
            return self.select_cost_optimized(providers, 10_000).await;
        }

        Err(RouterError::ProviderUnavailable("local provider down and fallback disabled".into()))
    }

    /// Pinned selection: return specific provider by model ID.
    fn select_pinned(
        &self,
        providers: &[Arc<dyn ModelProvider + Send + Sync>],
        model_id: &nexus_proto::model::ModelId,
    ) -> Result<Arc<dyn ModelProvider + Send + Sync>, RouterError> {
        let provider = self.providers.get(&model_id.provider)
            .ok_or_else(|| RouterError::ProviderNotFound(model_id.provider.to_string()))?;
        let health = self.get_cached_health(&model_id.provider);
        if !health.is_operational() {
            return Err(RouterError::ProviderUnavailable(format!(
                "pinned provider {} is {}",
                model_id.provider,
                match health {
                    ProviderHealth::Down { ref error, .. } => error,
                    _ => "unhealthy",
                }
            )));
        }

        debug!(
            selected = %model_id.provider,
            model = %model_id.model,
            "selected pinned provider"
        );
        Ok(provider)
    }

    /// Gets cached health status for a provider, refreshing if stale.
    fn get_cached_health(&self, provider_id: &ProviderId) -> ProviderHealth {
        const CACHE_TTL_SECS: u64 = 30;

        if let Some(entry) = self.health_cache.get(provider_id) {
            let (health, timestamp) = entry.value();
            if timestamp.elapsed().as_secs() < CACHE_TTL_SECS {
                return health.clone();
            }
        }

        // Cache miss or stale: return default Healthy (will be refreshed by background task)
        ProviderHealth::Healthy
    }

    /// Updates health cache for a provider.
    fn update_health_cache(&self, provider_id: ProviderId, health: ProviderHealth) {
        self.health_cache
            .insert(provider_id, (health, Instant::now()));
    }

    /// Background task: periodically refresh provider health status.
    async fn health_refresh_loop(self, cancel: CancellationToken) {
        let interval = Duration::from_secs(30);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    debug!("health refresh loop cancelled");                    break;
                }
                _ = tokio::time::sleep(interval) => {
                    self.refresh_health().await;
                }
            }
        }
    }

    /// Refreshes health status for all registered providers.
    pub async fn refresh_health(&self) {
        let providers = self.providers.all();
        let mut tasks = Vec::new();

        for provider in providers {
            let provider_id = provider.provider_id();
            let health_check = provider.health_check();

            tasks.push(tokio::spawn(async move {
                let start = Instant::now();
                match health_check.await {
                    Ok(_) => {
                        let latency_ms = start.elapsed().as_millis() as u64;
                        let health = if latency_ms < 1000 {
                            ProviderHealth::Healthy
                        } else {
                            ProviderHealth::Degraded { latency_ms }
                        };
                        Ok((provider_id, health))
                    }
                    Err(e) => {
                        Err((provider_id, ProviderHealth::Down {
                            since: Utc::now(),
                            error: e.to_string(),
                        }))
                    }
                }
            }));
        }

        for task in tasks {
            if let Ok(result) = task.await {
                match result {
                    Ok((id, health)) => self.update_health_cache(id, health),
                    Err((id, health)) => self.update_health_cache(id, health),
                }
            }
        }

        debug!("health refresh completed for {} providers", self.health_cache.len());    }

    /// Executes a complete, non-streaming model request.
    ///
    /// # Flow
    /// 1. Select provider based on routing policy
    /// 2. Estimate cost and check budget if enforcement enabled
    /// 3. Execute with timeout
    /// 4. Retry on transient failures with exponential backoff
    /// 5. Record cost and metrics on success
    ///
    /// # Arguments
    /// * `request` - The model request to execute
    /// * `calling_agent` - Agent ID for cost tracking and budget enforcement
    ///
    /// # Returns
    /// * `Ok(ModelResponse)` - Successful model response
    /// * `Err(RouterError)` - If selection, execution, or budget check fails
    #[instrument(skip(self, request), fields(agent = %calling_agent, model = ?request.model))]
    pub async fn complete(
        &self,
        mut request: ModelRequest,
        calling_agent: AgentId,
    ) -> Result<ModelResponse, RouterError> {
        let start = Instant::now();
        let mut last_error: Option<RouterError> = None;

        for attempt in 0..=self.config.max_retries {
            // 1. Select provider
            let provider = match self.select_provider(&request).await {
                Ok(p) => p,
                Err(e) => return Err(e),
            };

            // 2. Estimate cost and check budget
            if self.config.budget_enforcement {
                if let Err(e) = self.check_budget(&request, &provider, calling_agent).await {
                    return Err(e);
                }
            }

            // 3. Execute with timeout
            let exec_result = timeout(
                Duration::from_secs(self.config.request_timeout_secs),
                provider.complete(&request),
            )
            .await;

            match exec_result {
                Ok(Ok(response)) => {                    // 4. Record cost and metrics
                    let latency_ms = start.elapsed().as_millis() as u64;
                    self.record_cost(&response, calling_agent, latency_ms).await;

                    debug!(
                        provider = %provider.provider_id(),
                        model = %response.model.model,
                        latency_ms,
                        tokens = response.usage.total_tokens,
                        "model request completed"
                    );
                    return Ok(response);
                }
                Ok(Err(e)) => {
                    last_error = Some(e.clone());
                    if should_retry(&e) && attempt < self.config.max_retries {
                        let backoff = self.backoff_duration(attempt);
                        warn!(
                            attempt,
                            error = %e,
                            backoff_ms = backoff.as_millis(),
                            "retrying model request"
                        );
                        sleep(backoff).await;
                        continue;
                    }
                    return Err(e);
                }
                Err(_) => {
                    // Timeout
                    last_error = Some(RouterError::Timeout {
                        operation: "model complete".into(),
                        duration_ms: self.config.request_timeout_secs * 1000,
                    });
                    if attempt < self.config.max_retries {
                        let backoff = self.backoff_duration(attempt);
                        warn!(
                            attempt,
                            backoff_ms = backoff.as_millis(),
                            "retrying after timeout"
                        );
                        sleep(backoff).await;
                        continue;
                    }
                    return Err(RouterError::Timeout {
                        operation: "model complete".into(),
                        duration_ms: self.config.request_timeout_secs * 1000,
                    });
                }
            }        }

        Err(last_error.unwrap_or_else(|| RouterError::ProviderError("request failed after retries".into())))
    }

    /// Executes a streaming model request.
    ///
    /// # Flow
    /// 1. Select provider based on routing policy
    /// 2. Check budget if enforcement enabled
    /// 3. Get streaming response from provider
    /// 4. Wrap stream to record final cost when complete
    ///
    /// # Returns
    /// A stream of `Token` items, or error if setup fails.
    #[instrument(skip(self, request), fields(agent = %calling_agent))]
    pub async fn stream(
        &self,
        mut request: ModelRequest,
        calling_agent: AgentId,
    ) -> Result<impl Stream<Item = Result<Token, RouterError>> + Send + Unpin, RouterError> {
        // 1. Select provider
        let provider = self.select_provider(&request).await?;

        // 2. Check budget (estimate based on typical streaming request)
        if self.config.budget_enforcement {
            // Streaming: estimate conservatively
            let estimated_input = request.messages.iter().map(|m| count_tokens(&m.text_content(), &request.model.as_ref().map(|m| m.model.clone()).unwrap_or_default())).sum::<u32>();
            let estimated_output = 200; // Conservative estimate for streaming
            let estimated_cost = provider.estimate_cost(
                &request.model.as_ref().map(|m| m.model.clone()).unwrap_or_default(),
                estimated_input,
                estimated_output,
            );

            if let Some(mut budget) = self.cost_ledger.budgets().get_mut(&calling_agent) {
                if budget.remaining() < estimated_cost {
                    return Err(RouterError::BudgetExceeded {
                        limit: budget.limit_usd,
                        spent: budget.spent_usd,
                        requested: estimated_cost,
                    });
                }
            }
        }

        // 3. Get streaming response
        let stream = provider.stream(&request).await?;

        // 4. Wrap stream to record cost on completion        let cost_ledger = Arc::clone(&self.cost_ledger);
        let provider_id = provider.provider_id();
        let model = request.model.as_ref().map(|m| m.model.clone()).unwrap_or_default();
        let agent_id = calling_agent;

        let wrapped = stream.then(move |result| {
            let cost_ledger = Arc::clone(&cost_ledger);
            let provider_id = provider_id.clone();
            let model = model.clone();
            let agent_id = calling_agent;

            async move {
                // Pass through tokens, but track final token for cost recording
                match result {
                    Ok(token) if token.is_final => {
                        // Final token: record cost (simplified; real impl would track cumulative tokens)
                        let _ = cost_ledger.record(crate::cost::CostRecord::new(
                            agent_id,
                            provider_id,
                            model,
                            100, // Would track actual input tokens
                            50,  // Would track actual output tokens
                            0.001, // Would calculate actual cost
                            0,     // Would track actual latency
                        )).await;
                        Ok(token)
                    }
                    other => other,
                }
            }
        });

        Ok(wrapped)
    }

    /// Checks if agent has sufficient budget for the estimated request cost.
    async fn check_budget(
        &self,
        request: &ModelRequest,
        provider: &Arc<dyn ModelProvider + Send + Sync>,
        agent: AgentId,
    ) -> Result<(), RouterError> {
        // Estimate token counts
        let model_str = request.model.as_ref().map(|m| m.model.as_str()).unwrap_or("default");
        let input_tokens: u32 = request.messages.iter().map(|m| count_tokens(&m.text_content(), model_str)).sum();
        let output_tokens = request.max_tokens.unwrap_or(500);

        // Estimate cost
        let estimated_cost = provider.estimate_cost(model_str, input_tokens, output_tokens);
        // Check budget if one exists for this agent
        if let Some(mut budget) = self.cost_ledger.budgets().get_mut(&agent) {
            if budget.remaining() < estimated_cost {
                return Err(RouterError::BudgetExceeded {
                    limit: budget.limit_usd,
                    spent: budget.spent_usd,
                    requested: estimated_cost,
                });
            }
        }

        Ok(())
    }

    /// Records cost and metrics for a completed request.
    async fn record_cost(
        &self,
        response: &ModelResponse,
        agent: AgentId,
        latency_ms: u64,
    ) {
        let record = crate::cost::CostRecord::new(
            agent,
            response.model.provider.clone(),
            response.model.model.clone(),
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.estimated_cost_usd,
            latency_ms,
        );

        if let Err(e) = self.cost_ledger.record(record).await {
            warn!(error = %e, "failed to record cost");
        }
    }

    /// Computes exponential backoff duration with jitter.
    fn backoff_duration(&self, attempt: u32) -> Duration {
        let multiplier = 2u64.saturating_pow(attempt);
        let backoff = self.config.retry_backoff_ms.saturating_mul(multiplier);
        // Add jitter: ±25% to avoid thundering herd
        let jitter = (backoff as f64 * 0.25 * rand::random::<f64>()) as u64;
        Duration::from_millis(backoff.saturating_add(jitter).saturating_sub(jitter / 2))
    }

    /// Returns current health summary for all providers.
    pub fn health_summary(&self) -> HashMap<ProviderId, ProviderHealth> {
        self.health_cache
            .iter()
            .map(|entry| {                let (provider_id, (health, _)) = entry.pair();
                (provider_id.clone(), health.clone())
            })
            .collect()
    }

    /// Returns the router configuration.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }
}

// Helper: determines if an error is retryable.
fn should_retry(err: &RouterError) -> bool {
    matches!(
        err,
        RouterError::ProviderUnavailable(_)
            | RouterError::Timeout { .. }
            | RouterError::ProviderError(msg) if msg.contains("5") || msg.contains("429")
    )
}

// Manual Clone for ModelRouter (shallow clone for background tasks)
impl Clone for ModelRouter {
    fn clone(&self) -> Self {
        self.clone_shallow()
    }
}
