use crate::market::regime::{LiqRegime, MarketRegime, VolRegime};

/// Per-regime urgency parameters for the Almgren–Chriss objective.
///
/// Higher λ → more urgency → front-load execution → less timing risk but more market impact.
pub struct RegimeAcParams {
    /// Lowest urgency: calm vol + deep book. λ = 1e-5.
    pub lambda_low_vol_deep: f64,
    /// Baseline urgency: normal conditions. λ = 1e-4.
    pub lambda_normal: f64,
    /// Elevated urgency: high volatility (any liquidity). λ = 1e-2.
    pub lambda_high_vol: f64,
    /// Elevated urgency: thin liquidity (any vol). λ = 1e-2.
    pub lambda_thin_liq: f64,
    /// Maximum urgency: high vol AND thin book simultaneously. λ = 5e-2.
    pub lambda_high_vol_thin: f64,
}

impl Default for RegimeAcParams {
    fn default() -> Self {
        Self {
            lambda_low_vol_deep:  1e-4,
            lambda_normal:        1e-4,
            lambda_high_vol:      1e-2,
            lambda_thin_liq:      1e-2,
            lambda_high_vol_thin: 5e-2,
        }
    }
}

/// Select the effective λ based on the joint regime.
///
/// Match order matters: the most specific (high-vol/thin) is checked first,
/// then single-axis elevations, then the default.
pub fn effective_lambda(regime: &MarketRegime, params: &RegimeAcParams) -> f64 {
    match (&regime.vol, &regime.liq) {
        (VolRegime::Low,  LiqRegime::Deep) => params.lambda_low_vol_deep,
        (VolRegime::High, LiqRegime::Thin) => params.lambda_high_vol_thin,
        (VolRegime::High, _)               => params.lambda_high_vol,
        (_,               LiqRegime::Thin) => params.lambda_thin_liq,
        _                                  => params.lambda_normal,
    }
}

/// Regime-conditional Almgren–Chriss execution strategy.
///
/// At each step the strategy:
///   1. Classifies the current regime from live sigma and LOB half-spread.
///   2. Selects an effective λ for that regime.
///   3. Re-solves the AC receding-horizon formula for the optimal trade quantity.
///   4. Clamps the result to remaining inventory.
///
/// During the first `warmup_steps` calls to `on_block`, regime switching is
/// suppressed.  Instead, baseline-λ AC trades are executed and volatility
/// observations are accumulated.  On the last warm-up step `sigma_ref` (and
/// optionally `spread_ref`) are set to the mean of the observed values so that
/// regime boundaries are anchored to realised market conditions rather than the
/// GBM construction parameter.
pub struct RegimeAcStrategy {
    params: RegimeAcParams,
    total_inventory: f64,
    remaining: f64,
    horizon: usize,
    steps_taken: usize,
    /// Market impact cost coefficient (η).
    pub eta: f64,
    /// Volatility reference for regime classification (updated after warm-up).
    pub sigma_ref: f64,
    /// Spread reference for regime classification (updated after warm-up).
    pub spread_ref: f64,
    pub current_regime: Option<MarketRegime>,
    pub regime_history: Vec<String>,

    // ── Warm-up fields ──────────────────────────────────────────────────────
    /// Number of steps to collect observations before switching regimes.
    pub warmup_steps: usize,
    /// Whether warm-up has completed.
    pub warmed_up: bool,
    /// Volatility observations collected during warm-up.
    sigma_warmup: Vec<f64>,
    /// Half-spread observations collected during warm-up.
    spread_warmup: Vec<f64>,
}

impl RegimeAcStrategy {
    pub fn new(
        total_inventory: f64,
        horizon: usize,
        eta: f64,
        sigma_ref: f64,
        spread_ref: f64,
    ) -> Self {
        Self {
            params: RegimeAcParams::default(),
            total_inventory,
            remaining: total_inventory,
            horizon,
            steps_taken: 0,
            eta,
            sigma_ref,
            spread_ref,
            current_regime: None,
            regime_history: Vec::new(),
            warmup_steps: 50,
            warmed_up: false,
            sigma_warmup: Vec::new(),
            spread_warmup: Vec::new(),
        }
    }

    /// Override default params (for calibration sweeps).
    pub fn with_params(mut self, params: RegimeAcParams) -> Self {
        self.params = params;
        self
    }

    pub fn remaining(&self) -> f64 {
        self.remaining
    }

    pub fn is_complete(&self) -> bool {
        self.remaining <= 1e-8
    }

    /// Execute one time step given current market conditions.
    ///
    /// Returns the quantity to trade this step, already clamped to `remaining`.
    pub fn on_block(&mut self, sigma: f64, half_spread: f64) -> f64 {
        if self.is_complete() {
            return 0.0;
        }

        // ── Warm-up phase ────────────────────────────────────────────────────
        // Collect observations for the first `warmup_steps` calls.  Regime
        // switching is suppressed; baseline-λ AC trades are used instead so
        // that execution continues normally while calibration runs.
        if !self.warmed_up {
            self.sigma_warmup.push(sigma);
            if half_spread > 1e-15 {
                self.spread_warmup.push(half_spread);
            }
            if self.sigma_warmup.len() >= self.warmup_steps {
                self.sigma_ref = self.sigma_warmup.iter().sum::<f64>()
                    / self.sigma_warmup.len() as f64;
                if !self.spread_warmup.is_empty() {
                    self.spread_ref = self.spread_warmup.iter().sum::<f64>()
                        / self.spread_warmup.len() as f64;
                }
                self.warmed_up = true;
            }
            // During warm-up record "warmup" as the regime label so the history
            // length stays consistent with steps_taken.
            self.regime_history.push("warmup".to_string());
            return self.compute_baseline_trade(sigma);
        }

        // ── Active regime-switching phase ────────────────────────────────────

        // 1. Classify regime.
        let regime = MarketRegime::classify(sigma, self.sigma_ref, half_spread, self.spread_ref);

        // 2. Effective λ for this regime.
        let lambda = effective_lambda(&regime, &self.params);

        // 3. Remaining horizon (steps left, including this one).
        let remaining_horizon = self.horizon.saturating_sub(self.steps_taken).max(1);

        // 4. AC receding-horizon trade quantity.
        //    κ = √(λ · σ² / η)
        let kappa = if self.eta > 1e-15 {
            ((lambda * sigma * sigma) / self.eta).sqrt()
        } else {
            0.0
        };
        let rh = remaining_horizon as f64;
        let trade_qty = if kappa * rh < 1e-6 {
            // Linear (TWAP) fallback when κ or horizon is negligible.
            self.remaining / rh
        } else {
            let sinh_denom = (kappa * rh).sinh();
            if !sinh_denom.is_finite() || sinh_denom.abs() < 1e-15 {
                self.remaining / rh
            } else {
                self.remaining * kappa / sinh_denom
            }
        };

        // 5. Record regime label.
        self.regime_history.push(regime.label().to_string());
        self.current_regime = Some(regime);
        self.steps_taken += 1;

        // 6. On the final step, sweep up any residual to guarantee full execution.
        let qty = if remaining_horizon == 1 {
            self.remaining
        } else {
            trade_qty.min(self.remaining).max(0.0)
        };
        self.remaining -= qty;
        if self.remaining <= 1e-8 {
            self.remaining = 0.0;
        }
        qty
    }

    /// Baseline-λ AC trade used during warm-up (no regime switching).
    ///
    /// Uses `lambda_normal` and the construction `sigma_ref` so execution
    /// proceeds at the standard pace while calibration observations accumulate.
    fn compute_baseline_trade(&mut self, sigma: f64) -> f64 {
        let remaining_horizon = self.horizon.saturating_sub(self.steps_taken).max(1);
        let lambda = self.params.lambda_normal;
        let sigma_for_kappa = if sigma > 1e-15 { sigma } else { self.sigma_ref };
        let kappa = if self.eta > 1e-15 {
            ((lambda * sigma_for_kappa * sigma_for_kappa) / self.eta).sqrt()
        } else {
            0.0
        };
        let rh = remaining_horizon as f64;
        let trade_qty = if kappa * rh < 1e-6 {
            self.remaining / rh
        } else {
            let sinh_denom = (kappa * rh).sinh();
            if !sinh_denom.is_finite() || sinh_denom.abs() < 1e-15 {
                self.remaining / rh
            } else {
                self.remaining * kappa / sinh_denom
            }
        };
        self.steps_taken += 1;
        let qty = if remaining_horizon == 1 {
            self.remaining
        } else {
            trade_qty.min(self.remaining).max(0.0)
        };
        self.remaining -= qty;
        if self.remaining <= 1e-8 {
            self.remaining = 0.0;
        }
        qty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_strategy(horizon: usize) -> RegimeAcStrategy {
        RegimeAcStrategy::new(
            1_000.0, // total_inventory
            horizon,
            0.001,   // eta
            0.20,    // sigma_ref
            1.0,     // spread_ref
        )
    }

    /// High-vol + thin liquidity → lambda_high_vol_thin (5e-2), the largest.
    #[test]
    fn high_vol_thin_uses_largest_lambda() {
        let params = RegimeAcParams::default();
        let regime = MarketRegime::classify(0.30, 0.20, 1.4, 1.0); // High + Thin
        let lambda = effective_lambda(&regime, &params);
        assert!(
            (lambda - 5e-2).abs() < 1e-15,
            "expected lambda_high_vol_thin=5e-2, got {lambda}"
        );
    }

    /// Low-vol + deep liquidity → lambda_low_vol_deep (1e-4 = lambda_normal).
    /// RegimeAC never decelerates below baseline — only escalates.
    #[test]
    fn low_vol_deep_uses_smallest_lambda() {
        let params = RegimeAcParams::default();
        let regime = MarketRegime::classify(0.10, 0.20, 0.6, 1.0); // Low + Deep
        let lambda = effective_lambda(&regime, &params);
        assert!(
            (lambda - 1e-4).abs() < 1e-15,
            "expected lambda_low_vol_deep=1e-4 (baseline), got {lambda}"
        );
    }

    /// Normal vol + normal spread → lambda_normal (1e-4).
    #[test]
    fn normal_regime_uses_baseline_lambda() {
        let params = RegimeAcParams::default();
        let regime = MarketRegime::classify(0.20, 0.20, 1.0, 1.0); // Normal + Normal
        let lambda = effective_lambda(&regime, &params);
        assert!(
            (lambda - 1e-4).abs() < 1e-15,
            "expected lambda_normal=1e-4, got {lambda}"
        );
    }

    /// After running the full horizon, remaining inventory must be (near) zero.
    #[test]
    fn regime_ac_exhausts_full_inventory() {
        let mut strat = make_strategy(50);
        let sigma = 0.20;
        let hs = 1.0;
        for _ in 0..50 {
            strat.on_block(sigma, hs);
        }
        assert!(
            strat.remaining() <= 1e-6,
            "remaining inventory must be near zero after full horizon, got {}",
            strat.remaining()
        );
    }

    /// Regime labels are recorded in regime_history on each on_block call.
    /// Uses warmup_steps=0 to bypass warm-up so regime switching is immediate.
    #[test]
    fn regime_switches_are_recorded_in_history() {
        let mut strat = RegimeAcStrategy {
            warmup_steps: 0,
            warmed_up: true,
            ..make_strategy(10)
        };
        // First 5 steps: normal/normal
        for _ in 0..5 {
            strat.on_block(0.20, 1.0);
        }
        // Next 5 steps: high-vol/thin
        for _ in 0..5 {
            strat.on_block(0.30, 1.4);
        }
        assert_eq!(strat.regime_history.len(), 10);
        for label in &strat.regime_history[0..5] {
            assert_eq!(label, "normal/normal");
        }
        for label in &strat.regime_history[5..10] {
            assert_eq!(label, "high-vol/thin");
        }
    }

    /// The quantity returned by on_block must never exceed remaining inventory.
    #[test]
    fn trade_qty_never_exceeds_remaining() {
        let mut strat = make_strategy(5);
        let sigma = 0.20;
        let hs = 1.0;
        let mut prev_remaining = strat.remaining();
        for _ in 0..5 {
            let qty = strat.on_block(sigma, hs);
            assert!(
                qty <= prev_remaining + 1e-12,
                "trade_qty {qty} exceeded remaining {prev_remaining}"
            );
            prev_remaining = strat.remaining();
        }
    }

    /// After 50 warm-up steps with sigma=0.01, sigma_ref must converge to ~0.01
    /// and must differ from the construction value of 0.20.
    #[test]
    fn regime_ac_sigma_ref_set_from_warmup_not_construction_param() {
        // Large horizon so execution does not complete during warm-up.
        let mut strat = RegimeAcStrategy::new(1_000_000.0, 200, 0.001, 0.20, 1.0);
        let sigma = 0.01;
        let hs = 1.0;
        for _ in 0..50 {
            strat.on_block(sigma, hs);
        }
        assert!(
            strat.warmed_up,
            "strategy must be warmed up after 50 steps"
        );
        assert!(
            (strat.sigma_ref - 0.01).abs() < 1e-10,
            "sigma_ref should converge to 0.01 after warm-up, got {}",
            strat.sigma_ref
        );
        assert!(
            (strat.sigma_ref - 0.20).abs() > 1e-6,
            "sigma_ref must differ from construction value 0.20"
        );
    }

    /// During the first warmup_steps calls no regime switching should occur —
    /// all history entries must be tagged "warmup".
    #[test]
    fn regime_ac_uses_baseline_lambda_during_warmup() {
        let mut strat = RegimeAcStrategy::new(1_000_000.0, 200, 0.001, 0.20, 1.0);
        // Feed sigma that would trigger high-vol regime if active.
        for _ in 0..49 {
            strat.on_block(0.35, 1.5);
        }
        // Warm-up not yet complete after 49 steps.
        assert!(!strat.warmed_up, "should still be in warm-up after 49 steps");
        for label in &strat.regime_history {
            assert_eq!(
                label, "warmup",
                "regime label during warm-up must be 'warmup', got '{label}'"
            );
        }
    }

    /// After warm-up with sigma=0.01, a subsequent sigma=0.025 (2.5× ref) must
    /// be classified as high-vol.
    #[test]
    fn regime_ac_activates_high_vol_after_warmup() {
        let mut strat = RegimeAcStrategy::new(1_000_000.0, 200, 0.001, 0.20, 1.0);
        // Complete warm-up with sigma=0.01.
        for _ in 0..50 {
            strat.on_block(0.01, 1.0);
        }
        assert!(strat.warmed_up);
        // sigma=0.025 is 2.5× sigma_ref≈0.01, well above the 1.4× High threshold.
        strat.on_block(0.025, 1.0);
        let last_label = strat.regime_history.last().expect("history non-empty");
        assert!(
            last_label.starts_with("high-vol"),
            "expected high-vol regime after warm-up, got '{last_label}'"
        );
    }
}
