use crate::events::event::SimEvent;
use crate::events::order::Order;
use crate::market::liquidity::LiquiditySnapshot;
use crate::market::state::MarketState;
use crate::strategies::trait_def::Strategy;

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ORDER_ID: AtomicU64 = AtomicU64::new(100_000);

fn next_order_id() -> u64 {
    NEXT_ORDER_ID.fetch_add(1, Ordering::Relaxed)
}

pub struct AdaptiveOptimalStrategy {
    pub eta: f64,
    pub lambda: f64,
    pub liquidity_ref: f64,
    pub alpha: f64,
    pub beta: f64,
    remaining_notional: f64,
    initial_horizon: usize,
    current_step: usize,
    completed: bool,
}

impl AdaptiveOptimalStrategy {
    pub fn new(
        eta: f64,
        lambda: f64,
        liquidity_ref: f64,
        sigma_ref: f64,
        total_notional: f64,
    ) -> Self {
        let initial_horizon = if eta > 0.0 {
            let sigma = sigma_ref.max(1e-8);
            let kappa = ((lambda * sigma * sigma) / eta).sqrt();
            let epsilon = 1e-12;
            let base = 3.0;
            let t = (base / (kappa + epsilon)).ceil() as usize;
            t.clamp(1, 200)
        } else {
            50
        };

        Self {
            eta,
            lambda,
            liquidity_ref,
            alpha: 0.01,
            beta: 0.005,
            remaining_notional: total_notional,
            initial_horizon,
            current_step: 0,
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
        remaining_inventory: f64,
        remaining_horizon: usize,
    ) -> f64 {
        let sigma_t = volatility.max(1e-8);
        let depth_t = liquidity.depth_metric();
        let eta_t = self.eta * (self.liquidity_ref / depth_t.max(1e-8)).clamp(0.25, 4.0);
        let kappa_t = ((self.lambda * sigma_t * sigma_t) / eta_t).sqrt();

        let remaining_horizon = remaining_horizon.max(1);

        let q_t = if kappa_t < 1e-12 {
            remaining_inventory / remaining_horizon as f64
        } else {
            let rh = remaining_horizon as f64;
            let sinh_denom = (kappa_t * rh).sinh();
            if !sinh_denom.is_finite() || sinh_denom.abs() < 1e-15 {
                remaining_inventory / rh
            } else {
                let x1 = remaining_inventory * (kappa_t * (rh - 1.0)).sinh() / sinh_denom;
                remaining_inventory - x1
            }
        };

        q_t.clamp(0.0, remaining_inventory)
    }

    fn record_fill(&mut self, qty: f64) {
        self.remaining_notional -= qty;
        if self.remaining_notional <= 1e-8 {
            self.remaining_notional = 0.0;
            self.completed = true;
        }
    }
}

impl Strategy for AdaptiveOptimalStrategy {
    fn name(&self) -> &str {
        "AdaptiveOptimal"
    }

    fn on_event(&mut self, event: &SimEvent, market: &MarketState) -> Vec<Order> {
        match event {
            SimEvent::TimeTick { .. } if !self.completed => {
                let remaining_horizon = self.initial_horizon.saturating_sub(self.current_step);
                let chunk = self.compute_chunk(
                    market.volatility,
                    &market.liquidity,
                    self.remaining_notional,
                    remaining_horizon,
                );
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
