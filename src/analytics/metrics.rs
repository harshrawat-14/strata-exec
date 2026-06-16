/// Full cost decomposition for a completed execution session.
///
/// The five components partition total IS exactly:
///   spread + temporary_impact + permanent_impact + timing_cost + opportunity_cost == total
///
/// This is a conservation law: if it fails, something is double-counted or missing.
/// All values are in the same dollar units as `ExecutionMetrics::implementation_shortfall()`.
#[derive(Debug, Clone)]
pub struct CostDecomposition {
    /// Crossing half the bid-ask spread on each trade.
    /// Currently a proxy: qty × 0.5 × σ × √dt per trade.
    /// TODO Phase 2 Track B — replace with real LOB spread.
    pub spread_cost: f64,

    /// Transient (temporary) price impact: qty × decayed impact fraction at trade time.
    pub temporary_impact: f64,

    /// Permanent price impact: qty × cumulative price-adjustment fraction at trade time.
    pub permanent_impact: f64,

    /// Residual timing cost: price drift while waiting, net of all impact and spread.
    /// Can be negative when price moves in the trader's favour between slices.
    pub timing_cost: f64,

    /// Opportunity cost from unexecuted inventory.
    /// Zero while force_complete=true ensures full execution.
    /// Becomes relevant when partial execution is allowed.
    pub opportunity_cost: f64,

    /// Sum of all five components. Must equal `ExecutionMetrics::implementation_shortfall()`.
    pub total: f64,

    /// `total` as a percentage of (arrival_price × total_qty).
    pub total_as_pct: f64,
}

/// Execution-quality accounting for arrival-price benchmarking.
///
/// Tracks VWAP, implementation shortfall, and cumulative slippage
/// across all trades executed during a session.
pub struct ExecutionMetrics {
    arrival_price: f64,
    initial_inventory: f64,
    initial_notional: f64,
    total_executed: f64,
    vwap_numerator: f64,
    cumulative_cost: f64,
    cost_sq_accumulator: f64,
    accumulated_risk_penalty: f64,
    trade_count: usize,
    /// Cost attributed to forced liquidation at end of horizon (Option B).
    /// Zero under Option A (shared horizon). Non-zero only when force_complete=true
    /// and residual inventory was liquidated at the final step.
    pub forced_liquidation_cost: f64,

    // ── Cost decomposition accumulators ──────────────────────────────
    // Each accumulates the dollar contribution of its component across all trades.
    spread_cost_acc: f64,
    temporary_impact_acc: f64,
    permanent_impact_acc: f64,
    timing_cost_acc: f64,

    // ── VPIN observation (Phase 3 Track A) ───────────────────────────
    /// VPIN estimate sampled at the time of each recorded trade.
    /// Populated by `record_vpin`; length == `trade_count` after a full run.
    pub vpin_at_execution: Vec<f64>,
    /// Per-trade realised IS cost (`qty × (exec_price − arrival_price)`).
    /// Populated by `record_trade`; used for conditional IS analysis by VPIN.
    per_trade_cost: Vec<f64>,

    // ── Regime tracking (Phase 3 Track B) ────────────────────────────
    /// Regime label recorded at the time of each trade (Phase 3 Track B).
    /// Populated by `record_regime`; aligned with `per_trade_cost` by index.
    pub regime_at_execution: Vec<String>,
}

impl ExecutionMetrics {
    /// Create a new metrics tracker.
    ///
    /// * `arrival_price`     — benchmark price at session start.
    /// * `initial_inventory` — total quantity to be executed (e.g. `total_notional`).
    pub fn new(arrival_price: f64, initial_inventory: f64) -> Self {
        Self {
            arrival_price,
            initial_inventory,
            initial_notional: arrival_price * initial_inventory,
            total_executed: 0.0,
            vwap_numerator: 0.0,
            cumulative_cost: 0.0,
            cost_sq_accumulator: 0.0,
            accumulated_risk_penalty: 0.0,
            trade_count: 0,
            forced_liquidation_cost: 0.0,
            spread_cost_acc: 0.0,
            temporary_impact_acc: 0.0,
            permanent_impact_acc: 0.0,
            timing_cost_acc: 0.0,
            vpin_at_execution: Vec::new(),
            per_trade_cost: Vec::new(),
            regime_at_execution: Vec::new(),
        }
    }

    /// Record a single trade execution and decompose its cost into five components.
    ///
    /// # Parameters
    /// * `quantity`                  — notional size of the trade.
    /// * `market_price`              — mid-price at the time of execution (after permanent
    ///                                 impact has already been applied by the caller).
    /// * `slippage_cost`             — total adverse price delta in price units:
    ///                                 `exec_price = market_price + slippage_cost`.
    ///                                 For a sell this is typically ≤ 0.
    /// * `transient_impact_fraction` — lingering fractional impact from prior trades at
    ///                                 trade time (from `TransientImpactTracker::current_impact`).
    /// * `delta_permanent_fraction`  — marginal permanent-impact increment added by this
    ///                                 trade only: `gamma * (traded / total_notional)`.
    ///                                 Must NOT be the cumulative accumulator — that would
    ///                                 double-discount because `market_price` is already
    ///                                 depressed by all prior permanent impact.
    /// * `raw_price`                 — raw simulator price before any permanent-impact
    ///                                 depression. Used so permanent attribution is
    ///                                 `qty × delta × raw_price`, not `qty × delta × (raw × (1-adj))`.
    /// * `half_spread_proxy`         — half-spread estimate in price units.
    ///                                 Proxy formula: `0.5 × σ × √dt × market_price`.
    ///                                 TODO Phase 2 Track B — replace with real LOB spread.
    pub fn record_trade(
        &mut self,
        quantity: f64,
        market_price: f64,
        slippage_cost: f64,
        transient_impact_fraction: f64,
        delta_permanent_fraction: f64,
        raw_price: f64,
        half_spread_proxy: f64,
    ) {
        let exec_price = market_price + slippage_cost;
        let realized_cost = quantity * (exec_price - self.arrival_price);

        self.total_executed += quantity;
        self.vwap_numerator += quantity * exec_price;
        self.cumulative_cost += realized_cost;
        self.cost_sq_accumulator += realized_cost * realized_cost;
        self.per_trade_cost.push(realized_cost);
        self.trade_count += 1;

        // ── Cost decomposition ────────────────────────────────────────────
        // All components are in dollar cost units (positive = adverse for a sell).

        // 1. Spread cost proxy.
        //    TODO Phase 2 Track B — replace with real LOB spread.
        let spread = quantity * half_spread_proxy;

        // 2. Temporary impact: the fraction of slippage driven by transient market impact.
        //    transient_impact_fraction is a fractional price depression (dimensionless).
        let temporary = quantity * transient_impact_fraction * market_price;

        // 3. Permanent impact: marginal price depression caused by this trade only.
        //    Uses raw_price (undepressed) so the attribution is qty × delta × P₀,
        //    not qty × delta × P₀ × (1 - cumulative_adj) which would double-discount.
        let permanent = quantity * delta_permanent_fraction * raw_price;

        // 4. Timing cost: residual — what moved against the trader while waiting, net of
        //    all known impact components and the spread proxy.
        //    timing = total_IS_this_trade - spread - temporary - permanent
        let timing = realized_cost - spread - temporary - permanent;

        self.spread_cost_acc += spread;
        self.temporary_impact_acc += temporary;
        self.permanent_impact_acc += permanent;
        self.timing_cost_acc += timing;
    }

    /// Decompose total implementation shortfall into its five constituent cost components.
    ///
    /// Returns `None` if no trades have been recorded.
    ///
    /// # Conservation law
    /// `spread + temporary_impact + permanent_impact + timing_cost + opportunity_cost == total`
    /// is asserted within floating-point tolerance (1e-8 × |total|, minimum 1e-10).
    /// A violation means a component is double-counted or missing — treat it as a bug.
    pub fn decompose(&self) -> Option<CostDecomposition> {
        if self.trade_count == 0 {
            return None;
        }

        let total = self.cumulative_cost;

        // Opportunity cost represents the cost of forced liquidation at the horizon end.
        let opportunity_cost = self.forced_liquidation_cost;

        let total_as_pct = if self.initial_notional.abs() > 1e-15 {
            total / self.initial_notional * 100.0
        } else {
            0.0
        };

        let decomp = CostDecomposition {
            spread_cost: self.spread_cost_acc,
            temporary_impact: self.temporary_impact_acc,
            permanent_impact: self.permanent_impact_acc,
            timing_cost: self.timing_cost_acc,
            opportunity_cost,
            total,
            total_as_pct,
        };

        // Conservation-law assertion.
        let reconstructed = decomp.spread_cost
            + decomp.temporary_impact
            + decomp.permanent_impact
            + decomp.timing_cost
            + decomp.opportunity_cost;
        let tol = (total.abs() * 1e-8).max(1e-10);
        debug_assert!(
            (reconstructed - total).abs() <= tol,
            "CostDecomposition conservation law violated: \
             reconstructed={reconstructed:.10e} total={total:.10e} diff={:.10e}",
            (reconstructed - total).abs(),
        );

        Some(decomp)
    }

    /// Record a forced liquidation at horizon end (Option B only).
    ///
    /// Adds `cost` to both `cumulative_cost` (so IS includes the forced trade)
    /// and `forced_liquidation_cost` (for separate reporting).
    pub fn record_forced_liquidation(&mut self, residual_qty: f64, last_price: f64) {
        let cost = residual_qty * (last_price - self.arrival_price);
        self.cumulative_cost += cost;
        self.forced_liquidation_cost += cost;
        self.total_executed += residual_qty;
        self.trade_count += 1;
    }

    /// Record the theoretical risk penalty for a single time-step.
    ///
    /// J = E[Cost] + λ ∑ (inventory² · σ² · dt)
    ///
    /// * `inventory` - Remaining notional position.
    /// * `sigma`     - Annualized volatility.
    /// * `dt`        - Time step duration in years.
    pub fn record_risk_step(&mut self, inventory: f64, sigma: f64, dt: f64) {
        let dollar_inventory = inventory * self.arrival_price;
        self.accumulated_risk_penalty += dollar_inventory * dollar_inventory * sigma * sigma * dt;
    }

    /// Total notional executed so far.
    pub fn total_executed(&self) -> f64 {
        self.total_executed
    }

    /// Arrival (benchmark) price used for shortfall calculation.
    pub fn arrival_price(&self) -> f64 {
        self.arrival_price
    }

    /// Volume-weighted average price across all recorded trades.
    ///
    /// Returns `None` if no trades have been recorded yet.
    pub fn vwap(&self) -> Option<f64> {
        if self.total_executed <= 0.0 {
            return None;
        }
        Some(self.vwap_numerator / self.total_executed)
    }

    /// Implementation shortfall: total cost versus the arrival price.
    pub fn implementation_shortfall(&self) -> f64 {
        self.cumulative_cost
    }

    /// Slippage as a percentage of total execution value at arrival price.
    ///
    /// Returns `None` if no trades have been recorded.
    pub fn slippage_percent(&self) -> Option<f64> {
        let notional_at_arrival = self.total_executed * self.arrival_price;
        if notional_at_arrival.abs() < 1e-15 {
            return None;
        }
        Some((self.cumulative_cost / notional_at_arrival) * 100.0)
    }

    /// Mean per-trade realised cost.
    ///
    /// Returns `None` if no trades have been recorded.
    pub fn mean_cost(&self) -> Option<f64> {
        if self.trade_count == 0 {
            return None;
        }
        Some(self.cumulative_cost / self.trade_count as f64)
    }

    /// Population variance of per-trade realised costs.
    ///
    /// Uses the identity `Var = E[X²] − (E[X])²`.
    /// Returns `None` if no trades have been recorded.
    pub fn cost_variance(&self) -> Option<f64> {
        if self.trade_count == 0 {
            return None;
        }
        let n = self.trade_count as f64;
        let mean = self.cumulative_cost / n;
        Some((self.cost_sq_accumulator / n) - mean * mean)
    }

    /// Portfolio-level Almgren–Chriss objective:
    /// `E[Cost] + λ · ∑(inventory² · σ² · dt)`.
    ///
    /// This uses the true theoretical discrete risk term, preventing
    /// slice-count dependency and directly tracking the risk profile
    /// along the trajectory.
    ///
    /// Returns `None` if no trades have been recorded.
    pub fn ac_objective(&self, lambda: f64) -> Option<f64> {
        if self.trade_count == 0 {
            return None;
        }
        Some(self.cumulative_cost + lambda * self.accumulated_risk_penalty)
    }

    /// Average execution price across all recorded trades (VWAP).
    ///
    /// Returns `None` if no trades have been recorded.
    pub fn avg_exec_price(&self) -> Option<f64> {
        self.vwap()
    }

    /// Number of trades recorded so far.
    pub fn trade_count(&self) -> usize {
        self.trade_count
    }

    // ── VPIN observation (Phase 3 Track A) ───────────────────────────

    /// Record the VPIN estimate at the time of the most recently recorded trade.
    ///
    /// Call this immediately after `record_trade` so the indices stay aligned.
    /// If called more times than `record_trade`, the excess entries are ignored
    /// in conditional analysis (length mismatch guard is applied at query time).
    pub fn record_vpin(&mut self, vpin: f64) {
        self.vpin_at_execution.push(vpin);
    }

    /// Compute conditional IS % split by VPIN threshold.
    ///
    /// Returns `(low_vpin_is_pct, high_vpin_is_pct)` where each is the total
    /// realised cost for trades in that VPIN regime divided by `initial_notional`.
    /// Returns `None` for a bucket if no trades fall into it.
    pub fn conditional_shortfall_percent(
        &self,
        threshold: f64,
    ) -> (Option<f64>, Option<f64>) {
        if self.vpin_at_execution.is_empty() || self.initial_notional.abs() < 1e-15 {
            return (None, None);
        }
        let n = self.vpin_at_execution.len().min(self.per_trade_cost.len());
        let (mut low_cost, mut high_cost) = (0.0_f64, 0.0_f64);
        let (mut low_count, mut high_count) = (0usize, 0usize);

        for i in 0..n {
            if self.vpin_at_execution[i] > threshold {
                high_cost += self.per_trade_cost[i];
                high_count += 1;
            } else {
                low_cost += self.per_trade_cost[i];
                low_count += 1;
            }
        }

        let low_pct = if low_count > 0 {
            Some(low_cost / self.initial_notional * 100.0)
        } else {
            None
        };
        let high_pct = if high_count > 0 {
            Some(high_cost / self.initial_notional * 100.0)
        } else {
            None
        };
        (low_pct, high_pct)
    }

    /// Fraction of recorded trades executed when VPIN exceeded `threshold`.
    pub fn high_vpin_trade_fraction(&self, threshold: f64) -> f64 {
        let n = self.vpin_at_execution.len();
        if n == 0 {
            return 0.0;
        }
        let high = self.vpin_at_execution.iter().filter(|&&v| v > threshold).count();
        high as f64 / n as f64
    }

    // ── Regime tracking (Phase 3 Track B) ────────────────────────────

    /// Record the market regime label at the time of the most recently recorded trade.
    ///
    /// Call immediately after `record_trade` so indices align with `per_trade_cost`.
    pub fn record_regime(&mut self, label: &str) {
        self.regime_at_execution.push(label.to_string());
    }

    /// Compute the distribution of regime labels across all recorded trades.
    ///
    /// Returns a vector of `(label, count, fraction)` tuples sorted by count descending.
    /// Returns an empty vector when no regimes have been recorded.
    pub fn regime_distribution(&self) -> Vec<(String, usize, f64)> {
        if self.regime_at_execution.is_empty() {
            return Vec::new();
        }
        let total = self.regime_at_execution.len();
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for label in &self.regime_at_execution {
            *counts.entry(label.as_str()).or_insert(0) += 1;
        }
        let mut result: Vec<(String, usize, f64)> = counts
            .into_iter()
            .map(|(label, count)| (label.to_string(), count, count as f64 / total as f64))
            .collect();
        result.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        result
    }

    /// Compute mean IS% broken down by regime label.
    ///
    /// Returns `Vec<(label, mean_is_pct, fill_count)>` sorted by fill_count descending.
    /// Requires `regime_at_execution` and `per_trade_cost` to be co-populated.
    pub fn is_by_regime(&self) -> Vec<(String, f64, usize)> {
        let n = self.regime_at_execution.len().min(self.per_trade_cost.len());
        if n == 0 || self.initial_notional.abs() < 1e-15 {
            return Vec::new();
        }
        let mut cost_map: std::collections::HashMap<&str, (f64, usize)> =
            std::collections::HashMap::new();
        for i in 0..n {
            let label = self.regime_at_execution[i].as_str();
            let e = cost_map.entry(label).or_insert((0.0, 0));
            e.0 += self.per_trade_cost[i];
            e.1 += 1;
        }
        let notional = self.initial_notional;
        let mut result: Vec<(String, f64, usize)> = cost_map
            .into_iter()
            .map(|(label, (cost, count))| {
                let is_pct = cost / notional * 100.0;
                (label.to_string(), is_pct, count)
            })
            .collect();
        result.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        result
    }

    // ── Percentage-of-notional metrics ──────────────────────────────

    /// Implementation shortfall as a percentage of initial notional.
    ///
    /// Returns `None` if `initial_notional` is zero.
    pub fn shortfall_percent(&self) -> Option<f64> {
        if self.initial_notional.abs() < 1e-15 {
            return None;
        }
        Some(self.cumulative_cost / self.initial_notional * 100.0)
    }

    /// Mean per-trade cost as a percentage of initial notional.
    ///
    /// Returns `None` if no trades have been recorded or initial notional is zero.
    pub fn mean_cost_percent(&self) -> Option<f64> {
        let mean = self.mean_cost()?;
        if self.initial_notional.abs() < 1e-15 {
            return None;
        }
        Some(mean / self.initial_notional * 100.0)
    }

    /// Accumulated risk penalty expressed in percent² units.
    ///
    /// `risk_penalty / (initial_notional²) * 100.0`
    ///
    /// Returns `None` if no trades or initial notional is zero.
    pub fn risk_penalty_percent(&self) -> Option<f64> {
        if self.trade_count == 0 || self.initial_notional.abs() < 1e-15 {
            return None;
        }
        Some(
            self.accumulated_risk_penalty / (self.initial_notional * self.initial_notional) * 100.0,
        )
    }

    /// Portfolio-level AC objective as a percentage of initial notional.
    ///
    /// Uses unit-consistent formulation:
    /// `shortfall_pct + λ · (variance / initial_notional² · 100)`
    ///
    /// Both terms are in %-of-notional, so the sum is meaningful.
    ///
    /// Returns `None` if no trades or initial notional is zero.
    pub fn ac_objective_percent(&self, lambda: f64) -> Option<f64> {
        let shortfall_pct = self.shortfall_percent()?;
        let risk_pct = self.risk_penalty_percent()?;
        Some(shortfall_pct + lambda * risk_pct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vwap_single_trade() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // exec_price = 100.0 + 0.5 = 100.5
        m.record_trade(10.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0);

        let vwap = m.vwap().expect("should have vwap");
        assert!(
            (vwap - 100.5).abs() < 1e-10,
            "expected vwap 100.5, got {vwap}",
        );
    }

    #[test]
    fn vwap_multiple_trades() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // Trade 1: exec_price = 100 + 0.5 = 100.5, qty = 10
        m.record_trade(10.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0);
        // Trade 2: exec_price = 102 + 1.0 = 103.0, qty = 5
        m.record_trade(5.0, 102.0, 1.0, 0.0, 0.0, 0.0, 0.0);

        // vwap = (10*100.5 + 5*103.0) / 15 = (1005 + 515) / 15 = 101.333...
        let vwap = m.vwap().expect("should have vwap");
        let expected = (10.0 * 100.5 + 5.0 * 103.0) / 15.0;
        assert!(
            (vwap - expected).abs() < 1e-10,
            "expected vwap {expected}, got {vwap}",
        );
    }

    #[test]
    fn vwap_none_when_no_trades() {
        let m = ExecutionMetrics::new(100.0, 1000.0);
        assert!(m.vwap().is_none());
    }

    #[test]
    fn shortfall_accumulation() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // Trade 1: exec = 100.5, shortfall = 10 * (100.5 - 100) = 5.0
        m.record_trade(10.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0);
        // Trade 2: exec = 103.0, shortfall = 5 * (103.0 - 100) = 15.0
        m.record_trade(5.0, 102.0, 1.0, 0.0, 0.0, 0.0, 0.0);

        let expected = 5.0 + 15.0;
        assert!(
            (m.implementation_shortfall() - expected).abs() < 1e-10,
            "expected shortfall {expected}, got {}",
            m.implementation_shortfall(),
        );
    }

    #[test]
    fn slippage_percent_correct() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // exec = 100.5, qty = 10
        m.record_trade(10.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0);

        // shortfall = 10 * 0.5 = 5.0
        // notional_at_arrival = 10 * 100 = 1000
        // pct = 5 / 1000 * 100 = 0.5%
        let pct = m.slippage_percent().expect("should have pct");
        assert!((pct - 0.5).abs() < 1e-10, "expected 0.5%, got {pct}%",);
    }

    #[test]
    fn slippage_percent_none_when_no_trades() {
        let m = ExecutionMetrics::new(100.0, 1000.0);
        assert!(m.slippage_percent().is_none());
    }

    #[test]
    fn risk_penalty_accumulation() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);

        // arrival_price = 100.0
        // Step 1: inv = 1000.0 (dollar_inv = 100,000), sigma = 0.20, dt = 0.01
        m.record_risk_step(1000.0, 0.20, 0.01);
        // Penalty = 10^10 * 0.04 * 0.01 = 4_000_000.0

        // Step 2: inv = 500.0 (dollar_inv = 50,000), sigma = 0.20, dt = 0.01
        m.record_risk_step(500.0, 0.20, 0.01);
        // Penalty = 2.5 * 10^9 * 0.04 * 0.01 = 1_000_000.0

        assert!(
            (m.accumulated_risk_penalty - 5_000_000.0).abs() < 1e-10,
            "expected total risk 5_000_000.0, got {}",
            m.accumulated_risk_penalty
        );
    }

    #[test]
    fn ac_objective_discrete_calculation() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);

        // Trade 1: cost = 5.0
        m.record_trade(10.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0);
        // Total shortfall = 5.0

        // Risk step: 4_000_000.0
        m.record_risk_step(1000.0, 0.20, 0.01);

        let lambda = 0.5;
        // ac_objective = 5.0 + 0.5 * 4_000_000.0 = 2_000_005.0
        let ac = m.ac_objective(lambda).expect("should have ac_objective");
        assert!(
            (ac - 2_000_005.0).abs() < 1e-10,
            "expected ac_objective 2000005.0, got {ac}",
        );
    }

    #[test]
    fn shortfall_percent_correct() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // initial_notional = 100 * 1000 = 100_000
        // Trade: exec = 100.5, cost = 10*0.5 = 5.0
        m.record_trade(10.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0);

        // shortfall_pct = 5.0 / 100_000 * 100 = 0.005
        let pct = m.shortfall_percent().expect("should have pct");
        assert!((pct - 0.005).abs() < 1e-10, "expected 0.005%, got {pct}%",);
    }

    #[test]
    fn mean_cost_percent_correct() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // initial_notional = 100_000
        // Trade 1: cost = 5.0, Trade 2: cost = 15.0 → mean = 10
        m.record_trade(10.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0);
        m.record_trade(5.0, 102.0, 1.0, 0.0, 0.0, 0.0, 0.0);

        // mean_cost_pct = 10 / 100_000 * 100 = 0.01
        let pct = m.mean_cost_percent().expect("should have pct");
        assert!((pct - 0.01).abs() < 1e-10, "expected 0.01%, got {pct}%",);
    }

    // ── New tests ────────────────────────────────────────────────────────────

    /// WHAT: A sell filled below the arrival price produces a negative shortfall %.
    /// WHY: Sign-convention regression test for Fix 1. Before the fix, adverse sells
    ///      may have been recorded as positive cost, masking the true direction of loss.
    #[test]
    fn shortfall_is_negative_for_adverse_sell_execution() {
        // arrival_price = 100.0, initial_inventory = 1000.0
        // initial_notional = 100_000.0
        let mut m = ExecutionMetrics::new(100.0, 1000.0);

        // Sell filled at exec_price = 98.0 (below arrival).
        // slippage_cost = exec_price - market_price = 98.0 - 100.0 = -2.0
        // realized_cost = 100 * (98.0 - 100.0) = -200.0  ← negative (loss vs arrival)
        m.record_trade(100.0, 100.0, -2.0, 0.0, 0.0, 0.0, 0.0);

        let pct = m.shortfall_percent().expect("must have shortfall_percent");
        assert!(
            pct < 0.0,
            "adverse sell must produce negative shortfall %, got {pct:.6}%",
        );
    }

    /// WHAT: IS with gamma=0.01 (permanent impact) is more negative than IS with gamma=0.
    /// WHY: Direct regression test for Fix 1's direction — permanent impact must worsen
    ///      the sell shortfall, not improve it. gamma=0 is the clean baseline.
    #[test]
    fn shortfall_worsens_with_larger_permanent_impact() {
        // Simulate the price seen by each strategy after permanent impact accumulates.
        // gamma=0 → execution at market price (no permanent depression).
        // gamma=0.01 → each trade of 100/1000 notional adds 0.01*(100/1000)=0.001 depression.
        let arrival = 100.0;
        let inventory = 1000.0;
        let trade_size = 100.0;  // 10 equal trades of 100
        let market_price = 100.0;

        // gamma=0 baseline: execution at market_price with no impact.
        let mut m_no_impact = ExecutionMetrics::new(arrival, inventory);
        for _ in 0..10 {
            // exec_price = market_price (zero slippage, zero permanent impact)
            m_no_impact.record_trade(trade_size, market_price, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let is_no_impact = m_no_impact.shortfall_percent().expect("must have IS");

        // gamma=0.01: effective price drops with each trade.
        let gamma = 0.01;
        let mut m_with_impact = ExecutionMetrics::new(arrival, inventory);
        let mut price_adjustment = 0.0_f64;
        for _ in 0..10 {
            let delta = gamma * (trade_size / inventory);
            price_adjustment += delta;
            let effective_price = market_price * (1.0 - price_adjustment);
            // delta_permanent_fraction = marginal increment this trade; raw_price = market_price.
            m_with_impact.record_trade(trade_size, effective_price, 0.0, 0.0, delta, market_price, 0.0);
        }
        let is_with_impact = m_with_impact.shortfall_percent().expect("must have IS");

        assert!(
            is_with_impact < is_no_impact,
            "IS with permanent impact ({is_with_impact:.6}%) must be worse (more negative) \
             than without ({is_no_impact:.6}%)",
        );
    }

    #[test]
    fn risk_penalty_percent_scaling() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // initial_notional = 100_000
        m.record_trade(10.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0); // Ensure trade_count > 0

        // accumulate risk = 25.0
        m.accumulated_risk_penalty = 25.0;

        // risk_pct = 25 / (100_000^2) * 100 = 2.5e-7
        let pct = m.risk_penalty_percent().expect("should have pct");
        let expected = 25.0 / (100_000.0 * 100_000.0) * 100.0;
        assert!(
            (pct - expected).abs() < 1e-15,
            "expected {expected}, got {pct}",
        );
    }

    // ── Cost decomposition tests ─────────────────────────────────────────────

    /// Spread + temporary + permanent + timing + opportunity == total IS.
    /// This is the conservation law. Uses explicit non-zero values for all
    /// three impact parameters to exercise every accumulator path.
    #[test]
    fn cost_decomposition_sums_to_total_is() {
        // arrival = 100, inventory = 1000
        let mut m = ExecutionMetrics::new(100.0, 1000.0);

        // Trade 1: market=100, slip=-1 (exec=99), transient=0.005, delta_perm=0.002, raw=100, half_spread=0.10
        // realized_cost = 50 * (99 - 100) = -50
        m.record_trade(50.0, 100.0, -1.0, 0.005, 0.002, 100.0, 0.10);

        // Trade 2: market=99, slip=-0.5 (exec=98.5), transient=0.003, delta_perm=0.004, raw=100, half_spread=0.09
        // realized_cost = 30 * (98.5 - 100) = -45
        m.record_trade(30.0, 99.0, -0.5, 0.003, 0.004, 100.0, 0.09);

        let d = m.decompose().expect("must have decomposition");
        let reconstructed = d.spread_cost + d.temporary_impact + d.permanent_impact
            + d.timing_cost + d.opportunity_cost;
        let tol = (d.total.abs() * 1e-8).max(1e-10);
        assert!(
            (reconstructed - d.total).abs() <= tol,
            "conservation law violated: reconstructed={reconstructed:.10e} total={:.10e}",
            d.total,
        );
        // total_as_pct sanity: total / (100 * 1000) * 100
        let expected_pct = d.total / 100_000.0 * 100.0;
        assert!(
            (d.total_as_pct - expected_pct).abs() < 1e-10,
            "total_as_pct mismatch: {:.10e} vs {:.10e}",
            d.total_as_pct, expected_pct,
        );
    }

    /// Temporary impact component is negative (adverse cost) when selling into impact.
    /// A sell with positive transient_impact_fraction means price was already depressed —
    /// the component is qty × transient_frac × market_price, which is positive cost.
    /// For our sign convention (adverse sell = negative realized_cost), temporary_impact
    /// is a positive dollar value representing adverse execution cost.
    #[test]
    fn temporary_impact_is_positive_for_sell_with_transient_impact() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // transient_impact_fraction = 0.01 (1% lingering impact), market_price = 100
        // temporary = qty × 0.01 × 100 = 50 × 1.0 = 50.0 (positive cost for seller)
        m.record_trade(50.0, 100.0, -1.0, 0.01, 0.0, 0.0, 0.0);
        let d = m.decompose().expect("must have decomposition");
        assert!(
            d.temporary_impact > 0.0,
            "temporary_impact must be positive for a sell with transient impact, got {}",
            d.temporary_impact,
        );
    }

    /// Timing cost is zero when price does not move (arrival == execution price, zero impact,
    /// zero spread). All IS is zero, and every component including timing_cost is zero.
    #[test]
    fn timing_cost_is_zero_when_price_does_not_move() {
        // arrival = execution price, no impact, no spread → IS = 0, timing_cost = 0
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // exec_price = 100 + 0 = 100 == arrival_price → realized_cost = 0
        // spread=0, temporary=0, permanent=0 → timing = 0 - 0 - 0 - 0 = 0
        m.record_trade(100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let d = m.decompose().expect("must have decomposition");
        assert!(
            d.timing_cost.abs() < 1e-12,
            "timing_cost must be zero when price does not move, got {}",
            d.timing_cost,
        );
        assert!(
            d.total.abs() < 1e-12,
            "total IS must be zero when price does not move, got {}",
            d.total,
        );
    }

    /// Permanent impact component accumulates monotonically as sells are recorded.
    ///
    /// `permanent_impact_acc = Σ qty × delta_perm × raw_price` — a positive dollar
    /// value representing the cumulative price depression cost. It grows with each
    /// successive sell because each trade adds a positive marginal permanent increment.
    #[test]
    fn permanent_impact_component_grows_monotonically_during_sell() {
        let mut m = ExecutionMetrics::new(100.0, 10_000.0);
        let gamma = 0.005;
        let inventory = 10_000.0;
        let trade_size = 1_000.0;
        let raw_price = 100.0_f64;
        let mut price_adj = 0.0_f64;
        let mut prev_permanent = 0.0_f64;

        for step in 0..5usize {
            let delta = gamma * (trade_size / inventory);
            price_adj += delta;
            let effective_price = raw_price * (1.0 - price_adj);
            m.record_trade(trade_size, effective_price, 0.0, 0.0, delta, raw_price, 0.0);

            let d = m.decompose().expect("must have decomposition after each trade");

            if step > 0 {
                // Each new trade contributes qty × price_adj × price > 0, so the
                // cumulative permanent_impact grows strictly after step 0.
                assert!(
                    d.permanent_impact >= prev_permanent,
                    "permanent_impact must be non-decreasing across sells: \
                     step={step} prev={prev_permanent:.6} current={:.6}",
                    d.permanent_impact,
                );
            }
            prev_permanent = d.permanent_impact;
        }
        // After all trades, permanent_impact must be strictly positive.
        assert!(
            prev_permanent > 0.0,
            "cumulative permanent_impact must be positive after all sells, got {prev_permanent}",
        );
    }

    // ── Regime tracking tests (Phase 3 Track B) ──────────────────────────────

    /// Sum of all regime fill counts must equal total recorded trade count.
    #[test]
    fn regime_distribution_sums_to_total_fill_count() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        for _ in 0..5 {
            m.record_trade(10.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            m.record_regime("normal/normal");
        }
        for _ in 0..3 {
            m.record_trade(10.0, 100.0, -0.5, 0.0, 0.0, 0.0, 0.0);
            m.record_regime("high-vol/thin");
        }
        let dist = m.regime_distribution();
        let total: usize = dist.iter().map(|(_, c, _)| c).sum();
        assert_eq!(total, 8, "regime distribution must sum to total trade count");
        // Fractions must sum to 1.0
        let frac_sum: f64 = dist.iter().map(|(_, _, f)| f).sum();
        assert!((frac_sum - 1.0).abs() < 1e-12, "fractions must sum to 1.0, got {frac_sum}");
    }

    /// IS by regime: trades in high-vol/thin regime with negative slippage must produce
    /// negative IS%; normal regime with zero slippage must produce zero IS%.
    #[test]
    fn is_by_regime_computed_correctly() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // 4 normal/normal trades at arrival → zero IS
        for _ in 0..4 {
            m.record_trade(10.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            m.record_regime("normal/normal");
        }
        // 2 high-vol/thin trades at exec=99 → IS = 2×10×(99−100) = −20
        for _ in 0..2 {
            m.record_trade(10.0, 100.0, -1.0, 0.0, 0.0, 0.0, 0.0);
            m.record_regime("high-vol/thin");
        }
        let by_regime = m.is_by_regime();
        let normal = by_regime.iter().find(|(l, _, _)| l == "normal/normal");
        let hv_thin = by_regime.iter().find(|(l, _, _)| l == "high-vol/thin");

        let (_, normal_is, _) = normal.expect("normal/normal must be present");
        let (_, hv_is, _) = hv_thin.expect("high-vol/thin must be present");

        assert!(
            normal_is.abs() < 1e-12,
            "normal/normal IS% must be zero, got {normal_is}"
        );
        assert!(
            *hv_is < 0.0,
            "high-vol/thin IS% must be negative (adverse sell), got {hv_is}"
        );
    }

    /// Empty regime_history returns an empty distribution.
    #[test]
    fn empty_regime_history_returns_empty_distribution() {
        let m = ExecutionMetrics::new(100.0, 1000.0);
        assert!(m.regime_distribution().is_empty());
        assert!(m.is_by_regime().is_empty());
    }

    /// Spread cost proxy scales with volatility.
    /// For two identical trades differing only in half_spread_proxy,
    /// the higher-vol trade must have a larger (more negative) spread cost.
    #[test]
    fn spread_cost_proxy_scales_with_volatility() {
        // Low-vol run: sigma=0.01, dt=1/500 → half_spread ≈ 0.5×0.01×√(1/500)×100 ≈ 0.0224
        let dt = 1.0 / 500.0_f64;
        let sigma_low = 0.01_f64;
        let sigma_high = 0.40_f64;
        let price = 100.0_f64;
        let qty = 1000.0_f64;

        let hs_low = 0.5 * sigma_low * dt.sqrt() * price;
        let hs_high = 0.5 * sigma_high * dt.sqrt() * price;

        let mut m_low = ExecutionMetrics::new(price, qty);
        // slip = -hs_low so exec = price - hs_low (otherwise IS includes spread price move)
        m_low.record_trade(qty, price, -hs_low, 0.0, 0.0, 0.0, hs_low);

        let mut m_high = ExecutionMetrics::new(price, qty);
        m_high.record_trade(qty, price, -hs_high, 0.0, 0.0, 0.0, hs_high);

        let d_low  = m_low.decompose().expect("must have decomposition (low vol)");
        let d_high = m_high.decompose().expect("must have decomposition (high vol)");

        // spread_cost = qty × half_spread; higher vol → larger spread cost (positive value)
        assert!(
            d_high.spread_cost > d_low.spread_cost,
            "spread_cost must be larger under higher volatility: low={} high={}",
            d_low.spread_cost, d_high.spread_cost,
        );
    }
}
