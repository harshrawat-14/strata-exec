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
        }
    }

    pub fn predict_slippage(&self, chunk: f64, liq: f64, volatility: f64) -> f64 {
        let impact = self.alpha * (chunk / liq.max(1e-8));
        let timing = self.beta * volatility;
        impact + timing
    }

    pub fn compute_chunk(
        &self,
        volatility: f64,
        liquidity: &LiquiditySnapshot,
        remaining_notional: f64,
    ) -> f64 {
        let liq = liquidity.depth_metric();

        let liquidity_factor = (liq / self.liquidity_ref).clamp(0.25, 2.0);
        let volatility_factor = (self.sigma_ref / volatility.max(1e-8)).clamp(0.25, 2.0);

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
