use crate::engine::risk::RiskManager;
use crate::market::liquidity::LiquiditySnapshot;
use crate::strategies::adaptive::AdaptiveOptimalStrategy;
use crate::strategies::heuristic::HeuristicStrategy;
use crate::strategies::optimal::{AlmgrenChriss, ExecutionModel};
use crate::strategies::regime_ac::RegimeAcStrategy;

// ---------------------------------------------------------------------------
// Execution mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Equal-slice execution — no market-condition adjustments.
    /// The canonical cost-of-doing-nothing baseline.
    Twap,
    Heuristic,
    Optimal,
    AdaptiveOptimal,
    /// Regime-conditional Almgren–Chriss (Phase 3 Track B).
    /// Re-solves the AC formula each step with a λ selected by the joint
    /// (vol × liquidity) regime.  Requires `set_half_spread` to be called
    /// before each `on_block` when the LOB spread is available.
    RegimeAc,
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

pub struct Scheduler {
    risk: RiskManager,
    total_notional: f64,
    remaining_notional: f64,
    completed: bool,
    mode: ExecutionMode,
    execution_schedule: Option<Vec<f64>>,
    current_step: usize,

    // Strategy helpers
    heuristic: HeuristicStrategy,
    adaptive: AdaptiveOptimalStrategy,
    /// Regime-conditional AC strategy (Phase 3 Track B).
    pub regime_ac: Option<RegimeAcStrategy>,
    /// Fixed slice size for TWAP mode (= total_notional / horizon_steps).
    twap_qty_per_step: f64,

    eta: f64,
    lambda: f64,
    sigma_ref: f64,
    initial_horizon: usize,

    /// Current LOB half-spread, set by `set_half_spread` before each `on_block`.
    /// Used by RegimeAc mode; ignored by all other modes.
    current_half_spread: f64,
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
        horizon_override: Option<usize>,
    ) -> Self {
        let initial_horizon = if let Some(h) = horizon_override {
            h.max(1)
        } else if mode == ExecutionMode::AdaptiveOptimal && eta > 0.0 {
            let sigma = sigma_ref.max(1e-8);
            let kappa = ((lambda * sigma * sigma) / eta).sqrt();
            let epsilon = 1e-12;
            let base = 3.0;
            let t = (base / (kappa + epsilon)).ceil() as usize;
            t.clamp(1, 200)
        } else {
            0
        };

        // Build RegimeAcStrategy when mode == RegimeAc.
        // spread_ref is initialised to sigma_ref as a placeholder; the runner
        // will overwrite it via set_half_spread on the very first step.
        let regime_ac = if mode == ExecutionMode::RegimeAc {
            Some(RegimeAcStrategy::new(
                total_notional,
                initial_horizon.max(1),
                eta,
                sigma_ref,
                sigma_ref * 0.5, // placeholder; overwritten on first on_block call
            ))
        } else {
            None
        };

        Self {
            risk,
            total_notional,
            remaining_notional: total_notional,
            completed: false,
            mode,
            execution_schedule: None,
            current_step: 0,
            heuristic: HeuristicStrategy::new(base_chunk, sigma_ref, liquidity_ref, total_notional),
            adaptive: AdaptiveOptimalStrategy::new(
                eta,
                lambda,
                liquidity_ref,
                sigma_ref,
                total_notional,
            ),
            regime_ac,
            // For TWAP, base_chunk = total_notional / horizon_steps is set by the caller.
            twap_qty_per_step: base_chunk,
            eta,
            lambda,
            sigma_ref,
            initial_horizon,
            current_half_spread: 0.0,
        }
    }

    /// Set the current LOB half-spread before calling `on_block` for RegimeAc mode.
    ///
    /// The first call also initialises `spread_ref` on the underlying `RegimeAcStrategy`
    /// so that regime boundaries are anchored to the first observed spread.
    pub fn set_half_spread(&mut self, half_spread: f64) {
        self.current_half_spread = half_spread;
        if let Some(ref mut rac) = self.regime_ac {
            // Anchor spread_ref to the first non-zero observation.
            if rac.spread_ref <= 1e-15 || rac.spread_ref == rac.sigma_ref * 0.5 {
                rac.spread_ref = half_spread.max(1e-15);
            }
        }
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }

    pub fn remaining_notional(&self) -> f64 {
        self.remaining_notional
    }

    pub fn on_block(
        &mut self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
    ) -> Result<(), String> {
        if self.completed {
            return Ok(());
        }

        match self.mode {
            ExecutionMode::Twap => self.on_block_twap(),
            ExecutionMode::Heuristic => self.on_block_heuristic(volatility, liquidity),
            ExecutionMode::Optimal => self.on_block_optimal(volatility, liquidity),
            ExecutionMode::AdaptiveOptimal => self.on_block_adaptive_optimal(volatility, liquidity),
            ExecutionMode::RegimeAc => self.on_block_regime_ac(volatility),
        }
    }

    fn on_block_twap(&mut self) -> Result<(), String> {
        // TWAP executes exactly one fixed slice per step, with no slippage
        // budget check — TWAP is the unconditional baseline that always trades.
        let chunk = self.twap_qty_per_step.min(self.remaining_notional);
        if chunk > 1e-10 {
            self.book(chunk);
        }
        Ok(())
    }

    fn on_block_heuristic(
        &mut self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
    ) -> Result<(), String> {
        let mut adaptive_chunk =
            self.heuristic
                .compute_chunk(volatility, liquidity, self.remaining_notional);

        let liq = liquidity.depth_metric();
        let predicted_slippage = self
            .heuristic
            .predict_slippage(adaptive_chunk, liq, volatility);

        if self
            .risk
            .request_execute_chunk(adaptive_chunk, predicted_slippage)
            .is_ok()
        {
            self.book(adaptive_chunk);
            return Ok(());
        }

        // Retry at half size
        adaptive_chunk = (adaptive_chunk * 0.5).min(self.remaining_notional);
        let predicted_slippage = self
            .heuristic
            .predict_slippage(adaptive_chunk, liq, volatility);

        if self
            .risk
            .request_execute_chunk(adaptive_chunk, predicted_slippage)
            .is_ok()
        {
            self.book(adaptive_chunk);
        }

        Ok(())
    }

    fn on_block_optimal(
        &mut self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
    ) -> Result<(), String> {
        if self.execution_schedule.is_none() {
            let sigma = if volatility > 0.0 {
                volatility
            } else {
                self.sigma_ref
            };
            // Use initial_horizon as the schedule horizon when it was set via
            // horizon_override (i.e. > 0), so AC spans the same steps as the runner.
            let h_override = if self.initial_horizon > 0 {
                Some(self.initial_horizon)
            } else {
                None
            };
            let ac = AlmgrenChriss::new(self.total_notional, self.eta, self.lambda, sigma, h_override)?;
            self.execution_schedule = Some(ac.schedule());
            self.current_step = 0;
        }

        let schedule = match self.execution_schedule.as_ref() {
            Some(s) => s,
            None => return Ok(()),
        };

        if self.current_step >= schedule.len() {
            self.remaining_notional = 0.0;
            self.completed = true;
            return Ok(());
        }

        let proposed_chunk = schedule[self.current_step].min(self.remaining_notional);
        let liq = liquidity.depth_metric();

        // Let adaptive optimal strategy predict the expected optimal execution impact
        let predicted_slippage = self
            .adaptive
            .predict_slippage(proposed_chunk, liq, volatility);

        if self
            .risk
            .request_execute_chunk(proposed_chunk, predicted_slippage)
            .is_ok()
        {
            self.book(proposed_chunk);
            self.current_step += 1;
        }

        Ok(())
    }

    fn on_block_adaptive_optimal(
        &mut self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
    ) -> Result<(), String> {
        let remaining_horizon = self.initial_horizon.saturating_sub(self.current_step);
        let q_t = self.adaptive.compute_chunk(
            volatility,
            liquidity,
            self.remaining_notional,
            remaining_horizon,
        );

        let depth_t = liquidity.depth_metric();
        let predicted_slippage = self.adaptive.predict_slippage(q_t, depth_t, volatility);

        if self
            .risk
            .request_execute_chunk(q_t, predicted_slippage)
            .is_ok()
        {
            self.book(q_t);
            self.current_step += 1;
        }

        Ok(())
    }

    fn on_block_regime_ac(&mut self, volatility: f64) -> Result<(), String> {
        let half_spread = self.current_half_spread;
        let qty = match self.regime_ac.as_mut() {
            Some(rac) => rac.on_block(volatility, half_spread),
            None => return Ok(()),
        };
        if qty > 1e-10 {
            // RegimeAc bypasses the RiskManager slippage check — urgency is already
            // encoded in λ selection; blocking here would distort the regime signal.
            self.remaining_notional -= qty;
            if self.remaining_notional <= 1e-8 {
                self.remaining_notional = 0.0;
                self.completed = true;
            }
        }
        Ok(())
    }

    fn book(&mut self, chunk: f64) {
        self.remaining_notional -= chunk;
        if self.remaining_notional <= 1e-8 {
            self.remaining_notional = 0.0;
            self.completed = true;
        }
    }
}
