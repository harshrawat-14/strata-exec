use crate::core::optimal_execution::{AlmgrenChriss, ExecutionModel};
use crate::core::risk::RiskManager;
use crate::features::liquidity::LiquiditySnapshot;

// ---------------------------------------------------------------------------
// Execution mode
// ---------------------------------------------------------------------------

/// Selects between the adaptive heuristic and Almgren–Chriss optimal paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Volatility / liquidity–adaptive chunk sizing (original behaviour).
    Heuristic,
    /// Pre-computed Almgren–Chriss optimal trade schedule.
    Optimal,
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Block-driven adaptive execution scheduler.
///
/// Supports two modes:
/// * **Heuristic** — adapts chunk size each block to current volatility and
///   liquidity conditions, then gates execution through the `RiskManager`.
/// * **Optimal** — pre-computes a closed-form Almgren–Chriss schedule on the
///   first block, then steps through it one trade per block.
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

    // ── Optimal-mode fields ────────────────────────────────────────
    mode: ExecutionMode,
    execution_schedule: Option<Vec<f64>>,
    current_step: usize,
    eta: f64,
    lambda: f64,
}

impl Scheduler {
    pub fn new(
        risk: RiskManager,
        total_notional: f64,
        base_chunk: f64,
        sigma_ref: f64,
        liquidity_ref: f64,
        mode: ExecutionMode,
        eta: f64,
        lambda: f64,
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
            mode,
            execution_schedule: None,
            current_step: 0,
            eta,
            lambda,
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
    pub fn on_block(
        &mut self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
    ) -> Result<(), String> {
        if self.completed {
            return Ok(());
        }

        match self.mode {
            ExecutionMode::Heuristic => self.on_block_heuristic(volatility, liquidity),
            ExecutionMode::Optimal => self.on_block_optimal(volatility, liquidity),
        }
    }

    // ── Heuristic path (unchanged) ─────────────────────────────────

    /// Adapts the chunk size to current market conditions and attempts
    /// to book it through the risk manager.  If denied, retries once
    /// at half size.  If still denied, the block is skipped.
    fn on_block_heuristic(
        &mut self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
    ) -> Result<(), String> {
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

    // ── Optimal path ───────────────────────────────────────────────

    /// Steps through the pre-computed Almgren–Chriss schedule one
    /// trade per block.  The schedule is lazily initialised on the
    /// first call so that the live `sigma` value can be used.
    fn on_block_optimal(
        &mut self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
    ) -> Result<(), String> {
        // Lazy-init the schedule on first invocation.
        if self.execution_schedule.is_none() {
            let sigma = if volatility > 0.0 { volatility } else { self.sigma_ref };
            let ac = AlmgrenChriss::new(
                self.total_notional,
                self.eta,
                self.lambda,
                sigma,
                None,
            )?;
            let sched = ac.schedule();
            tracing::info!(
                horizon = sched.len(),
                sigma,
                "Almgren–Chriss schedule computed",
            );
            self.execution_schedule = Some(sched);
            self.current_step = 0;
        }

        let schedule = match self.execution_schedule.as_ref() {
            Some(s) => s,
            None => return Ok(()), // unreachable after lazy-init above
        };

        // All steps consumed → done.
        if self.current_step >= schedule.len() {
            self.remaining_notional = 0.0;
            self.completed = true;
            return Ok(());
        }

        let proposed_chunk = schedule[self.current_step].min(self.remaining_notional);
        let liq = liquidity.depth_metric();
        let predicted_slippage = self.predict_slippage(proposed_chunk, liq, volatility);

        if self.risk.request_execute_chunk(proposed_chunk, predicted_slippage).is_ok() {
            self.book(proposed_chunk);
            self.current_step += 1;
        }
        // If denied, do NOT increment current_step — retry next block.

        Ok(())
    }

    // ── Shared helpers ─────────────────────────────────────────────

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

    /// Helper: build a heuristic-mode scheduler with standard parameters.
    ///
    /// total_notional = 1000, base_chunk = 100, sigma_ref = 0.02,
    /// liquidity_ref = 500, max_slippage = 5%, reserve = 20%.
    fn default_scheduler() -> Scheduler {
        let risk = RiskManager::new(1000.0, 0.05, 0.2);
        Scheduler::new(
            risk, 1000.0, 100.0, 0.02, 500.0,
            ExecutionMode::Heuristic, 0.0, 0.0,
        )
    }

    /// Helper: build a "normal" liquidity snapshot.
    fn normal_liq() -> LiquiditySnapshot {
        // depth_metric = sqrt(500 * 500) = 500 — matches liquidity_ref.
        LiquiditySnapshot::new(500.0, 500.0)
    }

    // ── Existing heuristic tests (unchanged behaviour) ─────────────

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
        let mut sched = Scheduler::new(
            risk, 1000.0, 100.0, 0.02, 500.0,
            ExecutionMode::Heuristic, 0.0, 0.0,
        );

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

    // ── Optimal-mode tests ─────────────────────────────────────────

    #[test]
    fn optimal_mode_consumes_schedule() {
        // Generous risk budget so every chunk is approved.
        let risk = RiskManager::new(1000.0, 0.5, 0.0);
        let mut sched = Scheduler::new(
            risk, 1000.0, 100.0, 0.02, 500.0,
            ExecutionMode::Optimal, 0.1, 0.01,
        );

        let liq = normal_liq();
        let vol = 0.02;

        for _ in 0..300 {
            sched.on_block(vol, &liq).expect("should not fail");
            if sched.is_completed() {
                break;
            }
        }

        assert!(sched.is_completed(), "optimal schedule should complete");
        assert!(
            sched.remaining_notional() < 1e-8,
            "remaining {} should be ~0",
            sched.remaining_notional(),
        );
    }

    #[test]
    fn heuristic_vs_optimal_differ() {
        let liq = normal_liq();
        let vol = 0.02;

        // Heuristic
        let risk_h = RiskManager::new(1000.0, 0.5, 0.0);
        let mut h = Scheduler::new(
            risk_h, 1000.0, 100.0, 0.02, 500.0,
            ExecutionMode::Heuristic, 0.0, 0.0,
        );
        h.on_block(vol, &liq).expect("ok");
        let h_first = 1000.0 - h.remaining_notional();

        // Optimal
        let risk_o = RiskManager::new(1000.0, 0.5, 0.0);
        let mut o = Scheduler::new(
            risk_o, 1000.0, 100.0, 0.02, 500.0,
            ExecutionMode::Optimal, 0.1, 0.5,
        );
        o.on_block(vol, &liq).expect("ok");
        let o_first = 1000.0 - o.remaining_notional();

        // They should produce different first-chunk sizes.
        assert!(
            (h_first - o_first).abs() > 1e-6,
            "modes should differ: heuristic={h_first}, optimal={o_first}"
        );
    }
}
