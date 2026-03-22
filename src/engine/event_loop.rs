use crate::events::event::SimEvent;
use crate::events::queue::EventQueue;
use crate::execution::engine::ExecutionEngine;
use crate::market::state::MarketState;
use crate::strategies::trait_def::Strategy;

// ---------------------------------------------------------------------------
// DES Simulation Loop
// ---------------------------------------------------------------------------

/// The core discrete event simulation loop.
///
/// Processes events from a priority queue in chronological order,
/// routing each event through the market module, execution engine,
/// and all registered strategies.  Follow-on events (fills, orders)
/// are injected back into the queue for future processing.
pub struct DesSimulator {
    pub queue: EventQueue,
    pub market: MarketState,
    pub execution: ExecutionEngine,
    pub strategies: Vec<Box<dyn Strategy>>,
}

impl DesSimulator {
    pub fn new(
        queue: EventQueue,
        market: MarketState,
        execution: ExecutionEngine,
        strategies: Vec<Box<dyn Strategy>>,
    ) -> Self {
        Self {
            queue,
            market,
            execution,
            strategies,
        }
    }

    /// Run the simulation until the event queue is exhausted or
    /// `SimulationEnd` is encountered.
    ///
    /// Returns the total number of events processed.
    pub fn run(&mut self) -> usize {
        let mut processed = 0;

        while let Some(tse) = self.queue.pop() {
            // ── 1. Check for termination ────────────────────────────
            if matches!(tse.event, SimEvent::SimulationEnd) {
                processed += 1;
                break;
            }

            // ── 2. Market update ────────────────────────────────────
            let market_events = self.market.handle(&tse);
            for (t, e) in market_events {
                self.queue.schedule(t, e);
            }

            // ── 3. Execution engine ─────────────────────────────────
            let exec_events = self.execution.handle(&tse, &self.market);
            for (t, e) in exec_events {
                self.queue.schedule(t, e);
            }

            // ── 4. Strategies ───────────────────────────────────────
            for strat in &mut self.strategies {
                let orders = strat.on_event(&tse.event, &self.market);
                for order in orders {
                    // Schedule the SubmitOrder slightly after the current
                    // event so it is processed in a subsequent iteration.
                    self.queue
                        .schedule(tse.time + 1e-6, SimEvent::SubmitOrder { order });
                }
            }

            processed += 1;
        }

        processed
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::event::SimEvent;
    use crate::events::queue::EventQueue;
    use crate::execution::engine::ExecutionEngine;
    use crate::execution::impact::SquareRootImpact;
    use crate::market::liquidity::LiquiditySnapshot;
    use crate::market::state::MarketState;

    /// A trivial strategy that submits one order on the first TimeTick.
    struct OneShotStrategy {
        fired: bool,
    }
    impl Strategy for OneShotStrategy {
        fn name(&self) -> &str {
            "OneShot"
        }
        fn on_event(
            &mut self,
            event: &SimEvent,
            _market: &MarketState,
        ) -> Vec<crate::events::order::Order> {
            match event {
                SimEvent::TimeTick { .. } if !self.fired => {
                    self.fired = true;
                    vec![crate::events::order::Order::market_sell(
                        999,
                        "OneShot".into(),
                        10.0,
                    )]
                }
                _ => vec![],
            }
        }
        fn is_complete(&self) -> bool {
            self.fired
        }
    }

    #[test]
    fn des_loop_processes_all_events() {
        let mut queue = EventQueue::new();
        queue.schedule(0.0, SimEvent::TimeTick { step: 0 });
        queue.schedule(
            1.0,
            SimEvent::PriceUpdate {
                price: 100.0,
                volatility: 0.02,
            },
        );
        queue.schedule(2.0, SimEvent::TimeTick { step: 1 });
        queue.schedule(10.0, SimEvent::SimulationEnd);

        let market = MarketState::new(100.0, 0.02, LiquiditySnapshot::new(500.0, 500.0));
        let impact = SquareRootImpact::new(5_000_000.0, 0.5);
        let execution = ExecutionEngine::new(impact, 50.0, 500.0);

        let mut sim = DesSimulator::new(queue, market, execution, vec![]);
        let processed = sim.run();

        // 3 events + SimulationEnd
        assert_eq!(processed, 4);
    }

    #[test]
    fn des_loop_routes_strategy_orders_to_execution() {
        let mut queue = EventQueue::new();
        queue.schedule(0.0, SimEvent::TimeTick { step: 0 });
        queue.schedule(100.0, SimEvent::SimulationEnd);

        let market = MarketState::new(100.0, 0.02, LiquiditySnapshot::new(500.0, 500.0));
        let impact = SquareRootImpact::new(5_000_000.0, 0.5);
        let execution = ExecutionEngine::new(impact, 50.0, 500.0);

        let strat: Box<dyn Strategy> = Box::new(OneShotStrategy { fired: false });

        let mut sim = DesSimulator::new(queue, market, execution, vec![strat]);
        let processed = sim.run();

        // TimeTick → SubmitOrder (injected) → OrderFilled (injected) → SimulationEnd
        assert!(processed >= 3, "expected >= 3 events, got {processed}");
    }
}
