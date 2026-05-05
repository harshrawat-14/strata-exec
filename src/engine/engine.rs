use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use ethers::types::Address;
use serde_json::json;
use tokio::time::interval;

use super::state::EngineState;
use crate::analytics::metrics::ExecutionMetrics;
use crate::api::AppState;
use crate::blockchain::client::BlockchainClient;
use crate::engine::scheduler::{ExecutionMode, Scheduler};
use crate::market::liquidity::LiquiditySnapshot;
use crate::market::volatility::VolatilityEstimator;

/// Slippage-prediction constants (mirrors `scheduler::predict_slippage`).
const SLIPPAGE_ALPHA: f64 = 0.01;
const SLIPPAGE_BETA: f64 = 0.005;

/// Central orchestrator for the execution pipeline.
///
/// On each new block the engine:
/// 1. Fetches on-chain reserves for the configured pair.
/// 2. Builds a [`LiquiditySnapshot`] and derives the mid-price.
/// 3. Feeds block-over-block returns into the [`VolatilityEstimator`].
/// 4. Delegates chunk-sizing to the [`Scheduler`].
///
/// This is a **read-only** decision engine — no transactions are sent.
pub struct Engine {
    client: Arc<dyn BlockchainClient>,
    state: EngineState,
    last_seen_block: u64,
    poll_interval: Duration,
    scheduler: Scheduler,
    vol_estimator: VolatilityEstimator,
    pair_address: Address,
    last_price: Option<f64>,
    app_state: Arc<AppState>,
    /// ERC-20 decimal places for token0 of the pair.
    decimals_token0: u8,
    /// ERC-20 decimal places for token1 of the pair.
    decimals_token1: u8,
    /// Execution-quality tracker (initialised on first valid price).
    metrics: Option<ExecutionMetrics>,
    /// Active execution mode — used for the performance label.
    mode: ExecutionMode,
}

impl Engine {
    pub fn new(
        client: Arc<dyn BlockchainClient>,
        poll_interval: Duration,
        scheduler: Scheduler,
        vol_estimator: VolatilityEstimator,
        pair_address: Address,
        app_state: Arc<AppState>,
        mode: ExecutionMode,
    ) -> Self {
        Self {
            client,
            state: EngineState::Idle,
            last_seen_block: 0,
            poll_interval,
            scheduler,
            vol_estimator,
            pair_address,
            last_price: None,
            app_state,
            decimals_token0: 18,
            decimals_token1: 18,
            metrics: None,
            mode,
        }
    }

    /// Returns the current engine state.
    pub fn state(&self) -> &EngineState {
        &self.state
    }

    /// Returns the last observed block number.
    pub fn last_seen_block(&self) -> u64 {
        self.last_seen_block
    }

    /// Start the block-polling loop. Runs indefinitely.
    ///
    /// Uses `tokio::time::interval` for drift-free cadence — the tick
    /// accounts for processing time so polls stay evenly spaced.
    pub async fn run(&mut self) -> anyhow::Result<()> {
        self.transition(EngineState::Running);

        // ── Fetch token decimals once at startup ───────────────────
        let token0_addr = self.client.get_token0(self.pair_address).await?;
        let token1_addr = self.client.get_token1(self.pair_address).await?;

        self.decimals_token0 = self.client.get_erc20_decimals(token0_addr).await?;
        self.decimals_token1 = self.client.get_erc20_decimals(token1_addr).await?;

        tracing::info!(
            token0 = ?token0_addr,
            token1 = ?token1_addr,
            dec0 = self.decimals_token0,
            dec1 = self.decimals_token1,
            "pair token decimals resolved",
        );

        let mut ticker = interval(self.poll_interval);

        loop {
            ticker.tick().await;

            // Respect the control-plane is_running flag.
            if !self.app_state.is_running.load(Ordering::Relaxed) {
                continue;
            }

            let block_number = self.client.get_block_number().await?;

            if block_number > self.last_seen_block {
                self.last_seen_block = block_number;
                self.on_new_block(block_number).await?;
            }
        }
    }

    /// Centralised state transition — single choke-point for future
    /// logging, validation, or event emission on state changes.
    fn transition(&mut self, new_state: EngineState) {
        tracing::debug!(
            from = ?self.state,
            to = ?new_state,
            "engine state transition",
        );
        self.state = new_state;
    }

    /// Called when a new block is observed.
    ///
    /// Fetches reserves, derives mid-price, updates volatility, and
    /// forwards the decision to the scheduler.  Blocks with invalid
    /// reserves (zero / negative) are skipped with a warning.
    async fn on_new_block(&mut self, block_number: u64) -> anyhow::Result<()> {
        // ── 1. Fetch reserves ──────────────────────────────────────
        let (r0, r1) = self
            .client
            .get_uniswap_v2_reserves(self.pair_address)
            .await?;

        // ── 2. Build liquidity snapshot (raw reserves — no scaling) ─
        let liq = LiquiditySnapshot::new(r0 as f64, r1 as f64);

        // ── 3. Derive decimal-normalised mid-price ─────────────────
        //   price = (reserve1 / 10^dec1) / (reserve0 / 10^dec0)
        let norm0 = r0 as f64 / 10f64.powi(self.decimals_token0 as i32);
        let norm1 = r1 as f64 / 10f64.powi(self.decimals_token1 as i32);

        if norm0 <= 0.0 || norm1 <= 0.0 {
            tracing::warn!(block_number, "skipping block — invalid reserves");
            return Ok(());
        }

        let price = norm1 / norm0;

        // ── 4. Feed return into volatility estimator ───────────────
        if let Some(prev) = self.last_price {
            let ret = (price - prev) / prev;
            self.vol_estimator.update(ret);
        }

        // ── 5. Store current price ─────────────────────────────────
        self.last_price = Some(price);

        // ── 5b. Lazy-init execution metrics on first valid price ──
        if self.metrics.is_none() {
            let inventory = self.scheduler.remaining_notional();
            self.metrics = Some(ExecutionMetrics::new(price, inventory));
        }

        // ── 6. Extract current volatility ──────────────────────────
        let volatility = self.vol_estimator.current().unwrap_or(0.0);

        // ── 7. Scheduler decision (detect risk denials) ────────────
        let remaining_before = self.scheduler.remaining_notional();

        self.scheduler
            .on_block(volatility, &liq)
            .map_err(|e| anyhow::anyhow!("scheduler error: {e}"))?;

        let remaining_after = self.scheduler.remaining_notional();

        // ── 8. Update shared AppState counters ─────────────────────
        self.app_state
            .blocks_processed
            .fetch_add(1, Ordering::Relaxed);

        if (remaining_before - remaining_after).abs() > 1e-12 {
            // A chunk was successfully booked.
            self.app_state
                .chunks_executed
                .fetch_add(1, Ordering::Relaxed);

            // ── 8b. Record trade in execution metrics ──────────────
            let chunk = remaining_before - remaining_after;
            let liq_depth = liq.depth_metric().max(1e-8);
            // predicted_slippage_frac is a dimensionless fraction (e.g. 0.0015 = 15 bps).
            // record_trade expects a price-delta, so multiply by price to convert.
            let predicted_slippage_frac =
                SLIPPAGE_ALPHA * (chunk / liq_depth) + SLIPPAGE_BETA * volatility;
            let slippage_cost = -predicted_slippage_frac * price; // sell depresses price

            if let Some(ref mut m) = self.metrics {
                // Live engine: transient/permanent impact fractions and spread proxy
                // are not yet wired here. Pass zeros so decompose() is available but
                // all IS appears as timing_cost until Phase 2 Track B wires this up.
                // TODO Phase 2 Track B — wire up real impact fractions and LOB spread.
                m.record_trade(chunk, price, slippage_cost, 0.0, 0.0, 0.0, 0.0);
            }
        } else if !self.scheduler.is_completed() {
            // Neither full nor half chunk was booked → risk denial.
            self.app_state.risk_denials.fetch_add(1, Ordering::Relaxed);
        }

        {
            let mut vol = self.app_state.last_volatility.write().await;
            *vol = volatility;
        }
        {
            let mut rem = self.app_state.remaining_notional.write().await;
            *rem = remaining_after;
        }

        // ── 8c. Publish performance snapshot ───────────────────────
        if let Some(ref m) = self.metrics {
            let mode_label = match self.mode {
                ExecutionMode::Twap => "twap",
                ExecutionMode::Heuristic => "heuristic",
                ExecutionMode::Optimal => "optimal",
                ExecutionMode::AdaptiveOptimal => "adaptive_optimal",
            };
            let perf = json!({
                "mode": mode_label,
                "total_executed": m.total_executed(),
                "vwap": m.vwap(),
                "implementation_shortfall": m.implementation_shortfall(),
                "slippage_percent": m.slippage_percent(),
                "mean_cost": m.mean_cost(),
                "variance": m.cost_variance(),
                "ac_objective": m.ac_objective(1.0),
            });
            let mut perf_lock = self.app_state.performance.write().await;
            *perf_lock = perf;
        }

        // ── 9. Structured log ──────────────────────────────────────
        tracing::info!(
            block_number,
            price,
            volatility,
            remaining_notional = remaining_after,
            completed = self.scheduler.is_completed(),
            "block processed",
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    use async_trait::async_trait;
    use ethers::types::{Address, U256};

    use crate::engine::risk::RiskManager;
    use crate::engine::scheduler::ExecutionMode;

    /// Mock client that returns an atomically incrementing block number.
    struct MockBlockchainClient {
        counter: AtomicU64,
        reserves: (u128, u128),
        decimals_token0: u8,
        decimals_token1: u8,
    }

    impl MockBlockchainClient {
        fn new(reserves: (u128, u128)) -> Self {
            Self {
                counter: AtomicU64::new(0),
                reserves,
                decimals_token0: 18,
                decimals_token1: 18,
            }
        }

        fn with_decimals(reserves: (u128, u128), dec0: u8, dec1: u8) -> Self {
            Self {
                counter: AtomicU64::new(0),
                reserves,
                decimals_token0: dec0,
                decimals_token1: dec1,
            }
        }
    }

    #[async_trait]
    impl BlockchainClient for MockBlockchainClient {
        async fn get_block_number(&self) -> anyhow::Result<u64> {
            Ok(self.counter.fetch_add(1, Ordering::SeqCst) + 1)
        }

        async fn get_gas_price(&self) -> anyhow::Result<U256> {
            Ok(U256::zero())
        }

        async fn get_uniswap_v2_reserves(&self, _pair: Address) -> anyhow::Result<(u128, u128)> {
            Ok(self.reserves)
        }

        async fn get_token0(&self, _pair: Address) -> anyhow::Result<Address> {
            // Deterministic dummy address for token0.
            Ok("0x0000000000000000000000000000000000000001".parse()?)
        }

        async fn get_token1(&self, _pair: Address) -> anyhow::Result<Address> {
            // Deterministic dummy address for token1.
            Ok("0x0000000000000000000000000000000000000002".parse()?)
        }

        async fn get_erc20_decimals(&self, token: Address) -> anyhow::Result<u8> {
            let token0_addr: Address = "0x0000000000000000000000000000000000000001".parse()?;
            if token == token0_addr {
                Ok(self.decimals_token0)
            } else {
                Ok(self.decimals_token1)
            }
        }
    }

    /// Helper: build a scheduler with standard parameters.
    fn default_scheduler() -> Scheduler {
        let risk = RiskManager::new(1000.0, 0.05, 0.2);
        Scheduler::new(
            risk,
            1000.0,
            100.0,
            0.02,
            500.0,
            ExecutionMode::Heuristic,
            0.0,
            0.0,
            None,
        )
    }

    /// Helper: build a dummy AppState for tests.
    fn test_app_state() -> Arc<AppState> {
        Arc::new(AppState::new(1000.0))
    }

    /// Deterministic test using tokio virtual time — completes in ~0ms wall-clock.
    /// Mock returns (0, 0) reserves so blocks are skipped (no crash).
    #[tokio::test]
    async fn engine_tracks_new_blocks() {
        tokio::time::pause();

        let client = Arc::new(MockBlockchainClient::new((0, 0)));
        let mut engine = Engine::new(
            client,
            Duration::from_millis(100),
            default_scheduler(),
            VolatilityEstimator::new(20),
            Address::zero(),
            test_app_state(),
            ExecutionMode::Heuristic,
        );

        assert_eq!(engine.last_seen_block(), 0);
        assert_eq!(*engine.state(), EngineState::Idle);

        // Virtual time: 500ms timeout with 100ms interval → at least 5 ticks.
        let result = tokio::time::timeout(Duration::from_millis(500), engine.run()).await;

        // The engine loop is infinite, so we expect a timeout.
        assert!(result.is_err(), "engine should still be running at timeout");

        // With 100ms interval over 500ms, expect at least 5 block updates.
        assert!(
            engine.last_seen_block() >= 5,
            "expected at least 5 blocks, got {}",
            engine.last_seen_block(),
        );
    }

    /// End-to-end: valid reserves → price → vol → scheduler pipeline.
    #[tokio::test]
    async fn on_new_block_feeds_scheduler() {
        tokio::time::pause();

        // Reserves (1_000_000, 2_000_000) → mid_price = 2.0
        let client = Arc::new(MockBlockchainClient::new((1_000_000, 2_000_000)));
        let app_state = test_app_state();

        let mut engine = Engine::new(
            client,
            Duration::from_millis(100),
            default_scheduler(),
            VolatilityEstimator::new(20),
            Address::zero(),
            app_state.clone(),
            ExecutionMode::Heuristic,
        );

        // Run for 500ms — should process at least 5 blocks with valid reserves.
        let _ = tokio::time::timeout(Duration::from_millis(500), engine.run()).await;

        // Price should be stored after the first block.
        assert!(
            engine.last_price.is_some(),
            "last_price should be set after processing blocks",
        );

        // Scheduler should have executed some notional (reserves are valid).
        assert!(
            engine.scheduler.remaining_notional() < 1000.0,
            "scheduler should have executed some notional, remaining {}",
            engine.scheduler.remaining_notional(),
        );

        // AppState counters should have been updated.
        assert!(
            app_state.blocks_processed.load(Ordering::Relaxed) > 0,
            "blocks_processed should be > 0",
        );
    }

    /// Validate that decimal normalization produces the correct price.
    ///
    /// Scenario: USDC/WETH pair
    ///   token0 = USDC (6 decimals), reserve0  = 2_000_000_000  (2 000 USDC)
    ///   token1 = WETH (18 decimals), reserve1 = 1_000_000_000_000_000_000 (1 WETH)
    ///
    /// Expected normalised price = (1.0 WETH) / (2000.0 USDC) = 0.0005
    #[tokio::test]
    async fn decimal_normalization_corrects_price() {
        tokio::time::pause();

        let reserve0: u128 = 2_000_000_000; // 2 000 USDC  (6 dec)
        let reserve1: u128 = 1_000_000_000_000_000_000; // 1 WETH (18 dec)

        let client = Arc::new(MockBlockchainClient::with_decimals(
            (reserve0, reserve1),
            6,  // USDC decimals
            18, // WETH decimals
        ));

        let app_state = test_app_state();

        let mut engine = Engine::new(
            client,
            Duration::from_millis(100),
            default_scheduler(),
            VolatilityEstimator::new(20),
            Address::zero(),
            app_state,
            ExecutionMode::Heuristic,
        );

        // Run for just long enough to process one block.
        let _ = tokio::time::timeout(Duration::from_millis(250), engine.run()).await;

        let price = engine
            .last_price
            .expect("price should be set after one block");

        // norm0 = 2_000_000_000 / 10^6  = 2000.0
        // norm1 = 1e18         / 10^18 = 1.0
        // price = 1.0 / 2000.0 = 0.0005
        let expected = 0.0005_f64;
        assert!(
            (price - expected).abs() < 1e-12,
            "expected price {expected}, got {price}",
        );
    }
}
