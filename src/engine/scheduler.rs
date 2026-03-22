use crate::engine::risk::RiskManager;
use crate::market::liquidity::LiquiditySnapshot;
use crate::strategies::adaptive::AdaptiveOptimalStrategy;
use crate::strategies::heuristic::HeuristicStrategy;
use crate::strategies::optimal::{AlmgrenChriss, ExecutionModel};

// ---------------------------------------------------------------------------
// Execution mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Heuristic,
    Optimal,
    AdaptiveOptimal,
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

    eta: f64,
    lambda: f64,
    sigma_ref: f64,
    initial_horizon: usize,
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
        let initial_horizon = if mode == ExecutionMode::AdaptiveOptimal && eta > 0.0 {
            let sigma = sigma_ref.max(1e-8);
            let kappa = ((lambda * sigma * sigma) / eta).sqrt();
            let epsilon = 1e-12;
            let base = 3.0;
            let t = (base / (kappa + epsilon)).ceil() as usize;
            t.clamp(1, 200)
        } else {
            0
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
            eta,
            lambda,
            sigma_ref,
            initial_horizon,
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
            ExecutionMode::Heuristic => self.on_block_heuristic(volatility, liquidity),
            ExecutionMode::Optimal => self.on_block_optimal(volatility, liquidity),
            ExecutionMode::AdaptiveOptimal => self.on_block_adaptive_optimal(volatility, liquidity),
        }
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
            let ac = AlmgrenChriss::new(self.total_notional, self.eta, self.lambda, sigma, None)?;
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

    fn book(&mut self, chunk: f64) {
        self.remaining_notional -= chunk;
        if self.remaining_notional <= 1e-8 {
            self.remaining_notional = 0.0;
            self.completed = true;
        }
    }
}
