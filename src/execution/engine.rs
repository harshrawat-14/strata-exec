use crate::events::event::SimEvent;
use crate::events::order::{Order, Side};
use crate::events::queue::TimestampedEvent;
use crate::execution::impact::SquareRootImpact;
use crate::execution::transient_impact::TransientImpactTracker;
use crate::market::state::MarketState;

// ---------------------------------------------------------------------------
// Execution engine
// ---------------------------------------------------------------------------

/// Simulates order execution with realistic mechanics:
/// - Temporary market impact via the Square-Root Empirical Model
/// - Transient market impact decay tracking over time
/// - Partial fills for large orders
/// - Queue priority (FIFO)
pub struct ExecutionEngine {
    /// Pending orders awaiting execution.
    pending_orders: Vec<Order>,
    /// Impact model used for simulating instantaneous slippage.
    impact_model: SquareRootImpact,
    /// Tracks decayed impact from past executed orders.
    transient_tracker: TransientImpactTracker,
    /// Maximum single-fill quantity (orders larger than this get partial fills).
    max_fill_qty: f64,
}

impl ExecutionEngine {
    pub fn new(impact_model: SquareRootImpact, rho: f64, max_fill_qty: f64) -> Self {
        Self {
            pending_orders: Vec::new(),
            impact_model,
            transient_tracker: TransientImpactTracker::new(rho),
            max_fill_qty,
        }
    }

    /// Process an event and return follow-on events to inject into the queue.
    ///
    /// `current_time` is the simulation time of the triggering event so that
    /// follow-on events can be scheduled at `current_time + epsilon`.
    pub fn handle(&mut self, tse: &TimestampedEvent, market: &MarketState) -> Vec<(f64, SimEvent)> {
        match &tse.event {
            SimEvent::SubmitOrder { order } => {
                self.pending_orders.push(order.clone());
                // Attempt immediate execution.
                self.try_fill(tse.time, market)
            }
            SimEvent::CancelOrder { order_id } => {
                self.pending_orders.retain(|o| o.id != *order_id);
                vec![]
            }
            // On every tick or price update, try to fill pending orders.
            SimEvent::TimeTick { .. } | SimEvent::PriceUpdate { .. } => {
                self.try_fill(tse.time, market)
            }
            _ => vec![],
        }
    }

    /// Attempt to fill pending orders against the market snapshot, calculating
    /// execution price using the square-root impact formula plus past transient impact.
    fn try_fill(&mut self, current_time: f64, market: &MarketState) -> Vec<(f64, SimEvent)> {
        let mut events = Vec::new();
        let mut filled_ids = Vec::new();

        for order in self.pending_orders.iter_mut() {
            if order.quantity <= 0.0 {
                filled_ids.push(order.id);
                continue;
            }

            let fill_qty = order.quantity.min(self.max_fill_qty);

            // 1) Compute instantaneous base impact for THIS specific trade block
            let base_impact = self
                .impact_model
                .compute_impact(fill_qty, market.volatility);

            // 2) Query lingering impact from previous trades inside this simulation path
            let transient_impact = self.transient_tracker.current_impact(current_time);

            // 3) Total impact combines both effects
            let total_impact_fraction = base_impact + transient_impact;

            // Calculate execution price
            // A sell depresses the price down, a buy elevates the price up.
            let exec_price = match order.side {
                Side::Sell => market.price * (1.0 - total_impact_fraction),
                Side::Buy => market.price * (1.0 + total_impact_fraction),
            };

            // 4) Record this trade into history so its own base impact can decay forward
            self.transient_tracker
                .record_trade(current_time, base_impact);

            let remaining = order.quantity - fill_qty;
            order.quantity = remaining;

            if remaining <= 1e-10 {
                // Fully filled.
                filled_ids.push(order.id);
                events.push((
                    current_time + 1e-9,
                    SimEvent::OrderFilled {
                        order_id: order.id,
                        qty: fill_qty,
                        price: exec_price,
                        impact: total_impact_fraction,
                    },
                ));
            } else {
                // Partial fill — the remainder stays in the book.
                events.push((
                    current_time + 1e-9,
                    SimEvent::PartialFill {
                        order_id: order.id,
                        filled: fill_qty,
                        remaining,
                        price: exec_price,
                    },
                ));
            }
        }

        // Remove fully filled orders.
        self.pending_orders.retain(|o| !filled_ids.contains(&o.id));

        events
    }

    /// Number of pending orders.
    pub fn pending_count(&self) -> usize {
        self.pending_orders.len()
    }

    /// Borrow the impact model (for external inspection).
    pub fn impact_model(&self) -> &SquareRootImpact {
        &self.impact_model
    }

    /// Expose tracking variables for debugging or verification
    pub fn transient_impact(&self, current_time: f64) -> f64 {
        self.transient_tracker.current_impact(current_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::order::Order;
    use crate::market::liquidity::LiquiditySnapshot;

    fn test_market() -> MarketState {
        MarketState::new(100.0, 0.02, LiquiditySnapshot::new(500.0, 500.0))
    }

    fn test_engine() -> ExecutionEngine {
        let impact_model = SquareRootImpact::new(5_000_000.0, 0.5);
        ExecutionEngine::new(impact_model, 50.0, 500.0)
    }

    #[test]
    fn submit_and_fill_market_order() {
        let mut engine = test_engine();
        let market = test_market();
        let order = Order::market_sell(1, "Test".into(), 10.0);

        let tse = TimestampedEvent {
            time: 1.0,
            seq: 0,
            event: SimEvent::SubmitOrder { order },
        };

        let follow_on = engine.handle(&tse, &market);
        assert_eq!(follow_on.len(), 1);

        match &follow_on[0].1 {
            SimEvent::OrderFilled {
                order_id,
                qty,
                price,
                impact: _impact,
            } => {
                assert_eq!(*order_id, 1);
                assert!((*qty - 10.0).abs() < 1e-6);
                // execution price should be strictly below 100.0 for a sell
                assert!(*price < 100.0);
            }
            other => panic!("expected OrderFilled, got {:?}", other),
        }

        assert_eq!(engine.pending_count(), 0);

        // Ensure trade was recorded in transient history
        assert!(engine.transient_impact(1.0) > 0.0);
    }

    #[test]
    fn large_order_gets_partial_fill() {
        let impact_model = SquareRootImpact::new(5_000_000.0, 0.5);
        // max_fill_qty = 50 → order of 100 must be partially filled.
        let mut engine = ExecutionEngine::new(impact_model, 50.0, 50.0);
        let market = test_market();
        let order = Order::market_sell(2, "Test".into(), 100.0);

        let tse = TimestampedEvent {
            time: 1.0,
            seq: 0,
            event: SimEvent::SubmitOrder { order },
        };

        let follow_on = engine.handle(&tse, &market);
        assert_eq!(follow_on.len(), 1);

        match &follow_on[0].1 {
            SimEvent::PartialFill {
                filled, remaining, ..
            } => {
                assert!((*filled - 50.0).abs() < 1e-6);
                assert!((*remaining - 50.0).abs() < 1e-6);
            }
            other => panic!("expected PartialFill, got {:?}", other),
        }

        // One pending order remains.
        assert_eq!(engine.pending_count(), 1);

        // Ensure the partial fill generated some transient impact base
        assert!(engine.transient_impact(1.0) > 0.0);
    }

    #[test]
    fn cancel_removes_order() {
        let mut engine = test_engine();
        let market = test_market();

        // Submit then cancel.
        let order = Order::market_sell(3, "Test".into(), 10.0);
        engine.pending_orders.push(order);
        assert_eq!(engine.pending_count(), 1);

        let cancel = TimestampedEvent {
            time: 2.0,
            seq: 1,
            event: SimEvent::CancelOrder { order_id: 3 },
        };
        engine.handle(&cancel, &market);
        assert_eq!(engine.pending_count(), 0);
    }
}
