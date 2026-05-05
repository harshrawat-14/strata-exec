use crate::events::event::SimEvent;
use crate::events::order::Order;
use crate::market::liquidity::LiquiditySnapshot;
use crate::market::state::MarketState;
use crate::strategies::trait_def::Strategy;

use std::sync::atomic::{AtomicU64, Ordering};

/// Global monotonic order-ID counter.
static NEXT_ORDER_ID: AtomicU64 = AtomicU64::new(1);

fn next_order_id() -> u64 {
    NEXT_ORDER_ID.fetch_add(1, Ordering::Relaxed)
}

pub struct HeuristicStrategy {
    pub base_chunk: f64,
    pub sigma_ref: f64,
    pub liquidity_ref: f64,
    pub alpha: f64,
    pub beta: f64,
    remaining_notional: f64,
    completed: bool,
    /// EWM-smoothed volatility estimate (α=0.05 ≈ 19-step memory).
    /// Tracks the current volatility regime rather than a fixed initial reference,
    /// preventing the vol_factor from going stale in GARCH clustering periods.
    sigma_ema: f64,
}

impl HeuristicStrategy {
    pub fn new(base_chunk: f64, sigma_ref: f64, liquidity_ref: f64, total_notional: f64) -> Self {
        Self {
            base_chunk,
            sigma_ref,
            liquidity_ref,
            alpha: 0.01,
            beta: 0.005,
            remaining_notional: total_notional,
            completed: false,
            sigma_ema: sigma_ref,
        }
    }

    pub fn predict_slippage(&self, chunk: f64, liq: f64, volatility: f64) -> f64 {
        let impact = self.alpha * (chunk / liq.max(1e-8));
        let timing = self.beta * volatility;
        impact + timing
    }

    pub fn compute_chunk(
        &mut self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
        remaining_notional: f64,
    ) -> f64 {
        // Update EWM: α=0.05 gives ~19-step half-life memory.
        // sigma_ema tracks the current vol regime rather than a frozen initial ref,
        // so vol_factor stays meaningful throughout a GARCH clustering period.
        self.sigma_ema = 0.95 * self.sigma_ema + 0.05 * volatility;

        let liq = liquidity.depth_metric();

        let liquidity_factor = (liq / self.liquidity_ref).clamp(0.25, 2.0);
        // Use sigma_ema (rolling estimate) instead of frozen sigma_ref.
        let volatility_factor = (self.sigma_ema / volatility.max(1e-8)).clamp(0.25, 2.0);

        let adaptive_chunk = self.base_chunk * liquidity_factor * volatility_factor;

        adaptive_chunk.min(remaining_notional)
    }

    /// Called when a fill notification arrives.
    pub fn record_fill(&mut self, qty: f64) {
        self.remaining_notional -= qty;
        if self.remaining_notional <= 1e-8 {
            self.remaining_notional = 0.0;
            self.completed = true;
        }
    }
}

impl HeuristicStrategy {
    /// Read-only accessor for `sigma_ema` — exposed for white-box testing only.
    #[cfg(test)]
    pub(crate) fn sigma_ema(&self) -> f64 {
        self.sigma_ema
    }
}

impl Strategy for HeuristicStrategy {
    fn name(&self) -> &str {
        "Heuristic"
    }

    fn on_event(&mut self, event: &SimEvent, market: &MarketState) -> Vec<Order> {
        match event {
            SimEvent::TimeTick { .. } if !self.completed => {
                let chunk = self.compute_chunk(
                    market.volatility,
                    &market.liquidity,
                    self.remaining_notional,
                );
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ───────────────────────────────────────────────────────────────

    fn make_liquidity(depth: f64) -> LiquiditySnapshot {
        // depth_metric = sqrt(r_in * r_out); for equal reserves: sqrt(d*d) = d
        LiquiditySnapshot::new(depth, depth)
    }

    /// WHAT: sigma_ema at step 10 is higher than at step 1 when fed an increasing
    ///       volatility sequence.
    /// WHY: The EWM must track direction. If sigma_ema never rises when actual
    ///      volatility rises, the heuristic is blind to vol regimes — the EWM
    ///      update is broken.
    #[test]
    fn heuristic_sigma_ema_tracks_volatility_direction() {
        let sigma_ref = 0.02; // 2% initial reference
        let mut strat = HeuristicStrategy::new(
            100.0,    // base_chunk
            sigma_ref,
            500.0,    // liquidity_ref
            10_000.0, // total_notional
        );
        let liq = make_liquidity(500.0);

        let sigma_ema_at_step_1 = {
            strat.compute_chunk(0.03, &liq, 10_000.0); // step 1: slightly above ref
            strat.sigma_ema()
        };

        // Feed escalating volatility from step 2 to step 10.
        for i in 2..=10 {
            let vol = 0.03 + i as f64 * 0.02; // 0.05, 0.07, ... up to 0.23
            strat.compute_chunk(vol, &liq, 10_000.0);
        }
        let sigma_ema_at_step_10 = strat.sigma_ema();

        assert!(
            sigma_ema_at_step_10 > sigma_ema_at_step_1,
            "sigma_ema must rise with escalating vol: step1={sigma_ema_at_step_1:.6}, step10={sigma_ema_at_step_10:.6}",
        );
    }

    /// WHAT: chunk at σ=0.30 is less than chunk at σ=0.02 (all else equal).
    /// WHY: This is the core economic claim of the heuristic — trade smaller
    ///      in high-volatility regimes to avoid adverse timing risk. If this
    ///      is reversed, the strategy worsens execution quality in volatile markets.
    #[test]
    fn heuristic_trades_less_in_high_volatility() {
        let liq = make_liquidity(500.0); // fixed liquidity — isolate vol effect
        let remaining = 10_000.0;

        // Low-vol strategy: sigma_ema initialised at 0.02, feeds 0.02 consistently.
        let mut strat_low = HeuristicStrategy::new(100.0, 0.02, 500.0, remaining);
        // Warm up sigma_ema so it converges to 0.02.
        for _ in 0..30 {
            strat_low.compute_chunk(0.02, &liq, remaining);
        }
        let chunk_low_vol = strat_low.compute_chunk(0.02, &liq, remaining);

        // High-vol strategy: sigma_ema initialised at 0.02, but actual vol is 0.30.
        // The vol_factor = sigma_ema / actual_vol → < 1 → chunk shrinks.
        let mut strat_high = HeuristicStrategy::new(100.0, 0.02, 500.0, remaining);
        // Feed one step at high vol — vol_factor immediately < 1.
        let chunk_high_vol = strat_high.compute_chunk(0.30, &liq, remaining);

        assert!(
            chunk_high_vol < chunk_low_vol,
            "high-vol chunk {chunk_high_vol:.4} must be less than low-vol chunk {chunk_low_vol:.4}",
        );
    }

    /// WHAT: With extreme sigma values (0.001 and 10.0), chunk is never zero,
    ///       negative, or greater than remaining inventory.
    /// WHY: Edge cases in production cause silent failures. The clamp in
    ///      compute_chunk must prevent degenerate outputs at both extremes.
    #[test]
    fn heuristic_clamp_prevents_zero_or_negative_chunks() {
        let liq = make_liquidity(500.0);
        let remaining = 1_000.0;

        let extreme_vols = [
            0.001_f64, // near-zero vol → vol_factor approaches clamp ceiling of 2.0
            10.0_f64,  // extreme vol → vol_factor approaches clamp floor of 0.25
        ];

        for vol in extreme_vols {
            let mut strat = HeuristicStrategy::new(50.0, 0.02, 500.0, remaining);
            let chunk = strat.compute_chunk(vol, &liq, remaining);

            assert!(
                chunk > 0.0,
                "chunk must be positive at sigma={vol}, got {chunk}",
            );
            assert!(
                chunk <= remaining + 1e-10,
                "chunk {chunk} must not exceed remaining {remaining} at sigma={vol}",
            );
        }
    }
}
