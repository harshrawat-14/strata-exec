use crate::events::event::SimEvent;
use crate::events::order::Order;
use crate::market::state::MarketState;
use crate::strategies::trait_def::Strategy;

use std::sync::atomic::{AtomicU64, Ordering};

static TWAP_ORDER_ID: AtomicU64 = AtomicU64::new(300_000);

fn next_order_id() -> u64 {
    TWAP_ORDER_ID.fetch_add(1, Ordering::Relaxed)
}

/// True Time-Weighted Average Price execution strategy.
///
/// Divides `total_notional` into `horizon` equal slices and submits
/// exactly `qty_per_step` on every `TimeTick`, with no volatility
/// adjustment, no liquidity adjustment, and no risk-model gating.
/// This is the canonical cost-of-doing-nothing baseline.
pub struct TwapStrategy {
    qty_per_step: f64,
    remaining_notional: f64,
    completed: bool,
}

impl TwapStrategy {
    /// * `total_notional` — total quantity to liquidate.
    /// * `horizon`        — number of time steps over which to spread execution.
    pub fn new(total_notional: f64, horizon: usize) -> Self {
        let qty_per_step = if horizon > 0 {
            total_notional / horizon as f64
        } else {
            total_notional
        };
        Self {
            qty_per_step,
            remaining_notional: total_notional,
            completed: false,
        }
    }
}

impl Strategy for TwapStrategy {
    fn name(&self) -> &str {
        "TWAP"
    }

    fn on_event(&mut self, event: &SimEvent, _market: &MarketState) -> Vec<Order> {
        match event {
            SimEvent::TimeTick { .. } if !self.completed => {
                let chunk = self.qty_per_step.min(self.remaining_notional);
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
                self.remaining_notional -= qty;
                if self.remaining_notional <= 1e-8 {
                    self.remaining_notional = 0.0;
                    self.completed = true;
                }
                vec![]
            }
            SimEvent::PartialFill { filled, .. } => {
                self.remaining_notional -= filled;
                if self.remaining_notional <= 1e-8 {
                    self.remaining_notional = 0.0;
                    self.completed = true;
                }
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

    fn make_market(vol: f64) -> crate::market::state::MarketState {
        crate::market::state::MarketState::new(
            100.0,
            vol,
            crate::market::liquidity::LiquiditySnapshot::new(500.0, 500.0),
        )
    }

    // ── New tests ────────────────────────────────────────────────────────────

    /// WHAT: Chunk size is identical when run with σ=0.02 and σ=0.20.
    /// WHY: TWAP is the unconditional baseline — any vol-dependence would make
    ///      it not-TWAP and invalidate it as a cost-of-doing-nothing reference.
    #[test]
    fn twap_sends_equal_chunks_regardless_of_volatility() {
        let mut strat_low_vol = TwapStrategy::new(
            1000.0, // total inventory
            10,     // 10 equal steps
        );
        let mut strat_high_vol = TwapStrategy::new(1000.0, 10);

        let market_low = make_market(0.02);   // 2% daily vol
        let market_high = make_market(0.20);  // 20% daily vol — 10× spike

        let orders_low = strat_low_vol.on_event(&SimEvent::TimeTick { step: 0 }, &market_low);
        let orders_high = strat_high_vol.on_event(&SimEvent::TimeTick { step: 0 }, &market_high);

        assert_eq!(orders_low.len(), 1, "expected one order from low-vol TWAP");
        assert_eq!(orders_high.len(), 1, "expected one order from high-vol TWAP");
        assert!(
            (orders_low[0].quantity - orders_high[0].quantity).abs() < 1e-10,
            "TWAP chunk must be identical across vol regimes: low={}, high={}",
            orders_low[0].quantity,
            orders_high[0].quantity,
        );
    }

    /// WHAT: Sum of all chunks equals total_inventory when run for exactly
    ///       horizon steps with simulated fills.
    /// WHY: Incomplete execution is a critical failure mode; inventory
    ///      conservation is non-negotiable for any live strategy.
    #[test]
    fn twap_exhausts_full_inventory_by_horizon() {
        let total = 1000.0; // arbitrary total quantity
        let horizon = 10;   // exactly 10 steps → 100.0 per step
        let mut strat = TwapStrategy::new(total, horizon);
        let market = make_market(0.02);
        let mut total_filled = 0.0;

        for step in 0..horizon {
            let orders = strat.on_event(&SimEvent::TimeTick { step }, &market);
            for order in &orders {
                let qty = order.quantity;
                total_filled += qty;
                strat.on_event(
                    &SimEvent::OrderFilled {
                        order_id: order.id,
                        qty,
                        price: 100.0,
                        impact: 0.0,
                    },
                    &market,
                );
            }
        }

        assert!(
            (total_filled - total).abs() < 1e-8,
            "TWAP must exhaust full inventory: filled={total_filled}, expected={total}",
        );
    }

    /// WHAT: At every step, the submitted chunk never exceeds the remaining inventory.
    /// WHY: Selling more than you own is an invalid state that would cause
    ///      negative inventory — a silent failure in production.
    #[test]
    fn twap_does_not_overshoot_remaining_inventory() {
        let total = 1000.0;
        let horizon = 7; // non-round: 1000/7 ≈ 142.857 leaves a fractional last slice
        let mut strat = TwapStrategy::new(total, horizon);
        let market = make_market(0.02);
        let mut remaining = total;

        for step in 0..horizon {
            let orders = strat.on_event(&SimEvent::TimeTick { step }, &market);
            for order in &orders {
                assert!(
                    order.quantity <= remaining + 1e-10,
                    "chunk {} exceeds remaining inventory {} at step {}",
                    order.quantity,
                    remaining,
                    step,
                );
                remaining -= order.quantity;
                strat.on_event(
                    &SimEvent::OrderFilled {
                        order_id: order.id,
                        qty: order.quantity,
                        price: 100.0,
                        impact: 0.0,
                    },
                    &market,
                );
            }
        }
    }

    // ── Pre-existing tests ───────────────────────────────────────────────────

    #[test]
    fn equal_slices_over_horizon() {
        let strat = TwapStrategy::new(1000.0, 10);
        assert!((strat.qty_per_step - 100.0).abs() < 1e-10);
    }

    #[test]
    fn zero_horizon_does_not_panic() {
        let strat = TwapStrategy::new(1000.0, 0);
        // horizon=0 → qty_per_step = total_notional (execute everything in one step)
        assert!((strat.qty_per_step - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn submits_order_on_timetick() {
        let mut strat = TwapStrategy::new(500.0, 5);
        let market = crate::market::state::MarketState::new(
            100.0,
            0.02,
            crate::market::liquidity::LiquiditySnapshot::new(500.0, 500.0),
        );
        let orders = strat.on_event(&SimEvent::TimeTick { step: 0 }, &market);
        assert_eq!(orders.len(), 1);
        assert!((orders[0].quantity - 100.0).abs() < 1e-10);
    }

    #[test]
    fn ignores_price_update() {
        let mut strat = TwapStrategy::new(500.0, 5);
        let market = crate::market::state::MarketState::new(
            100.0,
            0.02,
            crate::market::liquidity::LiquiditySnapshot::new(500.0, 500.0),
        );
        let orders = strat.on_event(
            &SimEvent::PriceUpdate { price: 105.0, volatility: 0.03 },
            &market,
        );
        assert!(orders.is_empty());
    }

    #[test]
    fn completes_after_full_fill() {
        let mut strat = TwapStrategy::new(100.0, 1);
        let market = crate::market::state::MarketState::new(
            100.0,
            0.02,
            crate::market::liquidity::LiquiditySnapshot::new(500.0, 500.0),
        );
        // Submit the one slice.
        let orders = strat.on_event(&SimEvent::TimeTick { step: 0 }, &market);
        assert_eq!(orders.len(), 1);
        // Simulate the fill arriving back.
        strat.on_event(
            &SimEvent::OrderFilled { order_id: orders[0].id, qty: 100.0, price: 99.9, impact: 0.001 },
            &market,
        );
        assert!(strat.is_complete());
    }
}
