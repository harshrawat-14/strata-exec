use crate::core::risk::RiskManager;
use crate::features::liquidity::LiquiditySnapshot;

/// Block-driven adaptive execution scheduler.
///
/// On each block, computes an adaptive chunk size based on current
/// volatility and liquidity conditions, then gates execution through
/// the `RiskManager` before booking the chunk.
pub struct Scheduler {
    risk: RiskManager,
    total_notional: f64,
    remaining_notional: f64,
    base_chunk: f64,
    sigma_ref: f64,
    liquidity_ref: f64,
    alpha: f64,
    beta: f64,
    completed: bool,
}

impl Scheduler {
    pub fn new(
        risk: RiskManager,
        total_notional: f64,
        base_chunk: f64,
        sigma_ref: f64,
        liquidity_ref: f64,
    ) -> Self {
        Self {
            risk,
            total_notional,
            remaining_notional: total_notional,
            base_chunk,
            sigma_ref,
            liquidity_ref,
            alpha: 0.01,
            beta: 0.005,
            completed: false,
        }
    }

    /// Returns `true` once the full notional has been executed.
    pub fn is_completed(&self) -> bool {
        self.completed
    }

    /// Returns the remaining notional left to execute.
    pub fn remaining_notional(&self) -> f64 {
        self.remaining_notional
    }

    /// Process a single block.
    ///
    /// Adapts the chunk size to current market conditions and attempts
    /// to book it through the risk manager.  If denied, retries once
    /// at half size.  If still denied, the block is skipped.
    pub fn on_block(
        &mut self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
    ) -> Result<(), String> {
        if self.completed {
            return Ok(());
        }

        let liq = liquidity.depth_metric();

        let liquidity_factor = (liq / self.liquidity_ref).clamp(0.25, 2.0);
        let volatility_factor =
            (self.sigma_ref / volatility.max(1e-8)).clamp(0.25, 2.0);

        let mut adaptive_chunk =
            self.base_chunk * liquidity_factor * volatility_factor;
        adaptive_chunk = adaptive_chunk.min(self.remaining_notional);

        let predicted_slippage = self.predict_slippage(adaptive_chunk, liq, volatility);

        if self.risk.request_execute_chunk(adaptive_chunk, predicted_slippage).is_ok() {
            self.book(adaptive_chunk);
            return Ok(());
        }

        // Retry once at half size.
        adaptive_chunk = (adaptive_chunk * 0.5).min(self.remaining_notional);
        let predicted_slippage = self.predict_slippage(adaptive_chunk, liq, volatility);

        if self.risk.request_execute_chunk(adaptive_chunk, predicted_slippage).is_ok() {
            self.book(adaptive_chunk);
        }

        // If still denied, skip this block silently.
        Ok(())
    }

    /// Predict slippage for a given chunk under current conditions.
    fn predict_slippage(&self, chunk: f64, liq: f64, volatility: f64) -> f64 {
        let impact = self.alpha * (chunk / liq.max(1e-8));
        let timing = self.beta * volatility;
        impact + timing
    }

    /// Book a successfully risk-approved chunk.
    fn book(&mut self, chunk: f64) {
        self.remaining_notional -= chunk;
        if self.remaining_notional <= 1e-8 {
            self.remaining_notional = 0.0;
            self.completed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a scheduler with standard parameters.
    ///
    /// total_notional = 1000, base_chunk = 100, sigma_ref = 0.02,
    /// liquidity_ref = 500, max_slippage = 5%, reserve = 20%.
    fn default_scheduler() -> Scheduler {
        let risk = RiskManager::new(1000.0, 0.05, 0.2);
        Scheduler::new(risk, 1000.0, 100.0, 0.02, 500.0)
    }

    /// Helper: build a "normal" liquidity snapshot.
    fn normal_liq() -> LiquiditySnapshot {
        // depth_metric = sqrt(500 * 500) = 500 — matches liquidity_ref.
        LiquiditySnapshot::new(500.0, 500.0)
    }

    #[test]
    fn full_completion_across_blocks() {
        let mut sched = default_scheduler();
        let liq = normal_liq();
        let vol = 0.02; // matches sigma_ref → factor = 1.0

        // With factors = 1.0, adaptive_chunk = 100. Should complete in 10 blocks.
        for _ in 0..20 {
            sched.on_block(vol, &liq).expect("on_block should not fail");
            if sched.is_completed() {
                break;
            }
        }

        assert!(sched.is_completed(), "should be completed");
        assert!(sched.remaining_notional() < 1e-8);
    }

    #[test]
    fn volatility_reduces_chunk() {
        let mut sched = default_scheduler();
        let liq = normal_liq();

        // High volatility: 0.08 → factor = 0.02/0.08 = 0.25 (clamped floor).
        // Chunk = 100 * 1.0 * 0.25 = 25.
        sched.on_block(0.08, &liq).expect("should succeed");

        // After 1 block at chunk ≈ 25, remaining ≈ 975.
        let executed = 1000.0 - sched.remaining_notional();
        assert!(
            executed < 30.0,
            "high vol should reduce chunk, executed {executed}"
        );
    }

    #[test]
    fn liquidity_reduces_chunk() {
        let mut sched = default_scheduler();
        let vol = 0.02;

        // Low liquidity: depth = sqrt(125 * 125) = 125.
        // factor = 125/500 = 0.25 (clamped floor).
        let low_liq = LiquiditySnapshot::new(125.0, 125.0);

        sched.on_block(vol, &low_liq).expect("should succeed");

        let executed = 1000.0 - sched.remaining_notional();
        assert!(
            executed < 30.0,
            "low liquidity should reduce chunk, executed {executed}"
        );
    }

    #[test]
    fn risk_denial_causes_skip() {
        // Tiny budget: 0.001% slippage → budget = 0.01 — almost nothing.
        let risk = RiskManager::new(1000.0, 0.00001, 0.0);
        let mut sched = Scheduler::new(risk, 1000.0, 100.0, 0.02, 500.0);

        let liq = normal_liq();
        // Even the halved retry will likely exceed budget.
        sched.on_block(0.02, &liq).expect("should return Ok (skip)");

        // Nothing should have been booked.
        assert!(
            sched.remaining_notional() > 999.0,
            "denial should skip block, remaining {}",
            sched.remaining_notional(),
        );
    }

    #[test]
    fn completed_flag_triggers_early_return() {
        let mut sched = default_scheduler();
        let liq = normal_liq();
        let vol = 0.02;

        // Drive to completion.
        for _ in 0..20 {
            sched.on_block(vol, &liq).expect("on_block should not fail");
        }
        assert!(sched.is_completed());

        // Subsequent calls are no-ops.
        let remaining_before = sched.remaining_notional();
        sched.on_block(vol, &liq).expect("should succeed");
        assert_eq!(sched.remaining_notional(), remaining_before);
    }
}
