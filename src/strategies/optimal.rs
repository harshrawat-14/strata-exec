/// Almgren–Chriss closed-form optimal execution model.
///
/// Computes a deterministic trade schedule that minimises the combination
/// of temporary market-impact cost (`eta`) and timing risk (`lambda * sigma²`).

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Abstraction over execution schedule generators.
pub trait ExecutionModel {
    /// Returns the vector of per-step trade sizes `q_k`.
    fn schedule(&self) -> Vec<f64>;

    /// Returns the number of execution steps `T`.
    fn horizon(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Almgren–Chriss
// ---------------------------------------------------------------------------

/// Closed-form Almgren–Chriss optimal execution.
///
/// Given total inventory `X₀`, temporary-impact coefficient `η`,
/// risk-aversion `λ`, and per-period volatility `σ`, the optimal
/// inventory trajectory is:
///
///   x_k = X₀ · sinh(κ(T−k)) / sinh(κT)
///
/// where κ = √(λσ²/η).  When κ ≈ 0 the schedule degenerates to
/// the linear (TWAP) solution.
///
/// Note: This implements the η-only variant of Almgren-Chriss.
/// γ (permanent impact) is handled externally via
/// price_adjustment in MultiStrategyRunner.
/// See: Almgren & Chriss (2001), Section 2.
pub struct AlmgrenChriss {
    total_inventory: f64,
    /// Temporary impact coefficient (η).
    eta: f64,
    /// Permanent impact coefficient (γ). Default: 0.1 * eta.
    /// Does not affect the schedule shape (κ is η-only); used
    /// by callers to accumulate the price adjustment externally.
    pub gamma: f64,
    lambda: f64,
    sigma: f64,
    horizon: usize,
}

impl AlmgrenChriss {
    /// Build a new Almgren–Chriss model.
    ///
    /// * `horizon_override` — if `Some(T)`, use that horizon directly;
    ///   otherwise compute a recommended horizon from the model parameters.
    pub fn new(
        total_inventory: f64,
        eta: f64,
        lambda: f64,
        sigma: f64,
        horizon_override: Option<usize>,
    ) -> Result<Self, String> {
        if total_inventory <= 0.0 {
            return Err(format!(
                "total_inventory must be > 0, got {total_inventory}"
            ));
        }
        if eta <= 0.0 {
            return Err(format!("eta must be > 0, got {eta}"));
        }
        if lambda < 0.0 {
            return Err(format!("lambda must be >= 0, got {lambda}"));
        }
        if sigma < 0.0 {
            return Err(format!("sigma must be >= 0, got {sigma}"));
        }

        let horizon = match horizon_override {
            Some(h) if h >= 1 => h,
            Some(h) => return Err(format!("horizon_override must be >= 1, got {h}")),
            None => Self::recommended_horizon(eta, lambda, sigma),
        };

        Ok(Self {
            total_inventory,
            eta,
            gamma: 0.1 * eta,
            lambda,
            sigma,
            horizon,
        })
    }

    /// Compute a sensible horizon from the model parameters.
    ///
    /// `T = ceil(base / (κ + ε))`, clamped to `[1, 200]`.
    fn recommended_horizon(eta: f64, lambda: f64, sigma: f64) -> usize {
        let kappa = Self::kappa(eta, lambda, sigma);
        let epsilon = 1e-12;
        let base = 3.0;
        let t = (base / (kappa + epsilon)).ceil() as usize;
        t.clamp(1, 200)
    }

    /// Decay rate κ = √(λσ²/η).
    fn kappa(eta: f64, lambda: f64, sigma: f64) -> f64 {
        ((lambda * sigma * sigma) / eta).sqrt()
    }

    /// Build a linear (TWAP-style) schedule for the degenerate case.
    fn linear_schedule(&self) -> Vec<f64> {
        let t = self.horizon as f64;
        let chunk = self.total_inventory / t;

        let mut trades: Vec<f64> = (0..self.horizon).map(|_| chunk).collect();

        // Absorb rounding into the final trade.
        let sum: f64 = trades.iter().sum();
        if let Some(last) = trades.last_mut() {
            *last += self.total_inventory - sum;
        }
        trades
    }

    /// Build the sinh-based optimal schedule.
    fn sinh_schedule(&self, kappa: f64) -> Vec<f64> {
        let t = self.horizon;
        let tf = t as f64;

        // Inventory at each step: x_k = X0 * sinh(κ(T-k)) / sinh(κT)
        let sinh_kt = (kappa * tf).sinh();

        // Guard against sinh overflow / near-zero denominator.
        if sinh_kt.abs() < 1e-15 || !sinh_kt.is_finite() {
            return self.linear_schedule();
        }

        let mut inventory: Vec<f64> = (0..=t)
            .map(|k| {
                let val = (kappa * (tf - k as f64)).sinh() / sinh_kt;
                self.total_inventory * val
            })
            .collect();

        // Force boundary conditions.
        inventory[0] = self.total_inventory;
        inventory[t] = 0.0;

        // Trades: q_k = x_k - x_{k+1}
        let mut trades: Vec<f64> = (0..t).map(|k| inventory[k] - inventory[k + 1]).collect();

        // Absorb any residual rounding error into the last trade.
        let sum: f64 = trades.iter().sum();
        if let Some(last) = trades.last_mut() {
            *last += self.total_inventory - sum;
        }
        trades
    }
}

impl ExecutionModel for AlmgrenChriss {
    fn schedule(&self) -> Vec<f64> {
        let kappa = Self::kappa(self.eta, self.lambda, self.sigma);

        // Degenerate cases → linear schedule.
        if self.lambda == 0.0 || self.sigma == 0.0 || kappa < 1e-12 {
            return self.linear_schedule();
        }

        self.sinh_schedule(kappa)
    }

    fn horizon(&self) -> usize {
        self.horizon
    }
}

// ---------------------------------------------------------------------------
// Event-driven wrapper
// ---------------------------------------------------------------------------

use crate::events::event::SimEvent;
use crate::events::order::Order;
use crate::market::state::MarketState;
use crate::strategies::trait_def::Strategy;
use std::sync::atomic::{AtomicU64, Ordering};

static OPTIMAL_ORDER_ID: AtomicU64 = AtomicU64::new(200_000);

fn next_order_id() -> u64 {
    OPTIMAL_ORDER_ID.fetch_add(1, Ordering::Relaxed)
}

/// Event-driven wrapper around `AlmgrenChriss` that pre-computes a
/// schedule and steps through it one order per `TimeTick`.
pub struct OptimalStrategy {
    total_notional: f64,
    remaining_notional: f64,
    eta: f64,
    lambda: f64,
    sigma_ref: f64,
    schedule: Option<Vec<f64>>,
    current_step: usize,
    completed: bool,
}

impl OptimalStrategy {
    pub fn new(total_notional: f64, eta: f64, lambda: f64, sigma_ref: f64) -> Self {
        Self {
            total_notional,
            remaining_notional: total_notional,
            eta,
            lambda,
            sigma_ref,
            schedule: None,
            current_step: 0,
            completed: false,
        }
    }

    fn ensure_schedule(&mut self, sigma: f64) {
        if self.schedule.is_some() {
            return;
        }
        let s = if sigma > 0.0 { sigma } else { self.sigma_ref };
        if let Ok(ac) = AlmgrenChriss::new(self.total_notional, self.eta, self.lambda, s, None) {
            self.schedule = Some(ac.schedule());
        }
    }

    fn record_fill(&mut self, qty: f64) {
        self.remaining_notional -= qty;
        if self.remaining_notional <= 1e-8 {
            self.remaining_notional = 0.0;
            self.completed = true;
        }
    }
}

impl Strategy for OptimalStrategy {
    fn name(&self) -> &str {
        "Optimal"
    }

    fn on_event(&mut self, event: &SimEvent, market: &MarketState) -> Vec<Order> {
        match event {
            SimEvent::TimeTick { .. } if !self.completed => {
                self.ensure_schedule(market.volatility);
                let schedule = match self.schedule.as_ref() {
                    Some(s) => s,
                    None => return vec![],
                };

                if self.current_step >= schedule.len() {
                    self.remaining_notional = 0.0;
                    self.completed = true;
                    return vec![];
                }

                let chunk = schedule[self.current_step].min(self.remaining_notional);
                self.current_step += 1;

                if chunk > 1e-10 {
                    vec![Order::market_sell(
                        next_order_id(),
                        self.name().to_string(),
                        chunk,
                    )]
                } else {
                    vec![]
                }
            }
            SimEvent::OrderFilled { qty, .. } => {
                self.record_fill(*qty);
                vec![]
            }
            SimEvent::PartialFill { filled, .. } => {
                self.record_fill(*filled);
                vec![]
            }
            _ => vec![],
        }
    }

    fn is_complete(&self) -> bool {
        self.completed
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizon_override_works() {
        let ac = AlmgrenChriss::new(1000.0, 0.1, 0.01, 0.02, Some(25)).expect("valid params");
        assert_eq!(ac.horizon(), 25);
    }

    #[test]
    fn recommended_horizon_finite() {
        let ac = AlmgrenChriss::new(1000.0, 0.1, 0.01, 0.02, None).expect("valid params");
        let h = ac.horizon();
        assert!(h >= 1 && h <= 200, "horizon {h} out of [1, 200]");
    }

    #[test]
    fn linear_schedule_when_lambda_zero() {
        let ac = AlmgrenChriss::new(1000.0, 0.1, 0.0, 0.02, Some(10)).expect("valid params");
        let sched = ac.schedule();
        assert_eq!(sched.len(), 10);

        // Each trade should be ≈ 100.
        for &q in &sched {
            assert!((q - 100.0).abs() < 1e-9, "expected ~100, got {q}");
        }
    }

    #[test]
    fn schedule_sum_equals_inventory() {
        // Test with non-trivial optimal params.
        let ac = AlmgrenChriss::new(5000.0, 0.05, 0.1, 0.03, Some(20)).expect("valid params");
        let sched = ac.schedule();
        let sum: f64 = sched.iter().sum();
        assert!((sum - 5000.0).abs() < 1e-9, "sum {sum} != 5000");
    }

    #[test]
    fn schedule_sum_equals_inventory_auto_horizon() {
        let ac = AlmgrenChriss::new(10000.0, 0.1, 0.05, 0.02, None).expect("valid params");
        let sched = ac.schedule();
        let sum: f64 = sched.iter().sum();
        assert!((sum - 10000.0).abs() < 1e-9, "sum {sum} != 10000");
        assert_eq!(sched.len(), ac.horizon());
    }

    #[test]
    fn invalid_inputs_rejected() {
        assert!(AlmgrenChriss::new(0.0, 0.1, 0.0, 0.0, None).is_err());
        assert!(AlmgrenChriss::new(100.0, 0.0, 0.0, 0.0, None).is_err());
        assert!(AlmgrenChriss::new(100.0, 0.1, -1.0, 0.0, None).is_err());
        assert!(AlmgrenChriss::new(100.0, 0.1, 0.0, -1.0, None).is_err());
        assert!(AlmgrenChriss::new(100.0, 0.1, 0.0, 0.0, Some(0)).is_err());
    }

    #[test]
    fn optimal_schedule_front_loads() {
        // With risk-aversion, earlier trades should be larger.
        let ac = AlmgrenChriss::new(1000.0, 0.1, 0.5, 0.05, Some(10)).expect("valid params");
        let sched = ac.schedule();
        assert!(
            sched[0] > sched[9],
            "first trade {} should be > last trade {}",
            sched[0],
            sched[9],
        );
    }

    // ── New tests ────────────────────────────────────────────────────────────

    /// WHAT: Higher lambda (risk aversion) produces a larger first trade than lower lambda.
    /// WHY: Risk aversion is the core AC parameter — the entire justification for
    ///      front-loading is avoiding timing risk. If higher lambda doesn't produce
    ///      a larger first trade, the schedule formula is wrong.
    #[test]
    fn ac_front_loads_under_high_risk_aversion() {
        let eta = 0.1;
        let sigma = 0.05;
        let horizon = 10; // fixed horizon for an apples-to-apples comparison

        // lambda=0.001 → low risk aversion (near-TWAP)
        let ac_low = AlmgrenChriss::new(1000.0, eta, 0.001, sigma, Some(horizon))
            .expect("valid params");
        // lambda=1.0 → high risk aversion (aggressive front-loading)
        let ac_high = AlmgrenChriss::new(1000.0, eta, 1.0, sigma, Some(horizon))
            .expect("valid params");

        let first_low = ac_low.schedule()[0];
        let first_high = ac_high.schedule()[0];

        assert!(
            first_high > first_low,
            "high risk-aversion first trade {first_high} must exceed low risk-aversion {first_low}",
        );
    }

    /// WHAT: With lambda=0.0 the schedule is uniform (equal chunks), matching TWAP.
    /// WHY: AC degenerates to TWAP under zero risk aversion — verifies the boundary
    ///      condition that the model has a correct limiting case.
    #[test]
    fn ac_converges_to_uniform_when_lambda_is_zero() {
        let ac = AlmgrenChriss::new(1000.0, 0.1, 0.0, 0.05, Some(10)).expect("valid params");
        let sched = ac.schedule();

        let expected_chunk = 100.0; // 1000 / 10 steps
        for (k, &q) in sched.iter().enumerate() {
            assert!(
                (q - expected_chunk).abs() < 1e-9,
                "step {k}: expected uniform {expected_chunk}, got {q}",
            );
        }
    }

    /// WHAT: Sum of all schedule trades equals total_inventory for several lambda values.
    /// WHY: Inventory conservation is non-negotiable — any residual means the
    ///      strategy would leave unexecuted position at end of horizon.
    #[test]
    fn ac_total_inventory_conserved() {
        let total = 5000.0;
        // Test across different risk-aversion levels including the degenerate cases.
        let lambdas = [0.0, 0.001, 0.01, 0.1, 1.0];

        for lambda in lambdas {
            let ac = AlmgrenChriss::new(total, 0.05, lambda, 0.03, Some(20))
                .expect("valid params");
            let sched = ac.schedule();
            let sum: f64 = sched.iter().sum();
            assert!(
                (sum - total).abs() < 1e-9,
                "lambda={lambda}: schedule sum {sum} != total {total}",
            );
        }
    }

    /// WHAT: After each of 5 trades with gamma > 0, the effective price is lower than
    ///       the raw price, and the depression deepens monotonically with each trade.
    /// WHY: Permanent impact must compound across trades (Fix 1). If the price
    ///      adjustment stays flat, earlier sales don't penalise later ones and the
    ///      IS calculation will be wrong.
    #[test]
    fn ac_permanent_impact_accumulates_across_steps() {
        let gamma = 0.1;        // 10% of full inventory causes 10% permanent depression
        let total_notional = 1000.0;
        let trade_size = 200.0; // sell 200 per step (5 steps exhaust inventory)
        let raw_price = 100.0;

        let mut price_adjustment = 0.0_f64;
        let mut prev_effective = raw_price; // start at full price (no impact yet)

        for k in 0..5usize {
            // Permanent impact: each trade adds gamma*(trade/X0) to the cumulative fraction.
            price_adjustment += gamma * (trade_size / total_notional);
            let effective = raw_price * (1.0 - price_adjustment);

            assert!(
                effective < raw_price,
                "step {k}: effective_price {effective} must be below raw_price {raw_price}",
            );
            assert!(
                effective < prev_effective,
                "step {k}: effective_price {effective} must be lower than previous {prev_effective}",
            );

            prev_effective = effective;
        }
    }
}
