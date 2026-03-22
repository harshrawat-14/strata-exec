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
        }
    }

    /// Record a single trade execution.
    ///
    /// * `quantity`           — notional size of the trade.
    /// * `market_price`       — mid-price at the time of execution.
    /// * `predicted_slippage` — estimated slippage fraction (from `predict_slippage`).
    pub fn record_trade(&mut self, quantity: f64, market_price: f64, predicted_slippage: f64) {
        let exec_price = market_price + predicted_slippage;
        let realized_cost = quantity * (exec_price - self.arrival_price);

        self.total_executed += quantity;
        self.vwap_numerator += quantity * exec_price;
        self.cumulative_cost += realized_cost;
        self.cost_sq_accumulator += realized_cost * realized_cost;
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
        m.record_trade(10.0, 100.0, 0.5);

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
        m.record_trade(10.0, 100.0, 0.5);
        // Trade 2: exec_price = 102 + 1.0 = 103.0, qty = 5
        m.record_trade(5.0, 102.0, 1.0);

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
        m.record_trade(10.0, 100.0, 0.5);
        // Trade 2: exec = 103.0, shortfall = 5 * (103.0 - 100) = 15.0
        m.record_trade(5.0, 102.0, 1.0);

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
        m.record_trade(10.0, 100.0, 0.5);

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
        m.record_trade(10.0, 100.0, 0.5);
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
        m.record_trade(10.0, 100.0, 0.5);

        // shortfall_pct = 5.0 / 100_000 * 100 = 0.005
        let pct = m.shortfall_percent().expect("should have pct");
        assert!((pct - 0.005).abs() < 1e-10, "expected 0.005%, got {pct}%",);
    }

    #[test]
    fn mean_cost_percent_correct() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // initial_notional = 100_000
        // Trade 1: cost = 5.0, Trade 2: cost = 15.0 → mean = 10
        m.record_trade(10.0, 100.0, 0.5);
        m.record_trade(5.0, 102.0, 1.0);

        // mean_cost_pct = 10 / 100_000 * 100 = 0.01
        let pct = m.mean_cost_percent().expect("should have pct");
        assert!((pct - 0.01).abs() < 1e-10, "expected 0.01%, got {pct}%",);
    }

    #[test]
    fn risk_penalty_percent_scaling() {
        let mut m = ExecutionMetrics::new(100.0, 1000.0);
        // initial_notional = 100_000
        m.record_trade(10.0, 100.0, 0.5); // Ensure trade_count > 0

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
}
