use crate::events::event::SimEvent;
use crate::events::queue::TimestampedEvent;
use crate::market::liquidity::LiquiditySnapshot;

// ---------------------------------------------------------------------------
// Market state
// ---------------------------------------------------------------------------

/// Centralised market state updated by events flowing through the DES loop.
///
/// The market module is the single source of truth for the current price,
/// volatility, and liquidity conditions visible to strategies and the
/// execution engine.
pub struct MarketState {
    pub price: f64,
    pub volatility: f64,
    pub liquidity: LiquiditySnapshot,
    pub step: usize,
}

impl MarketState {
    pub fn new(initial_price: f64, initial_volatility: f64, liquidity: LiquiditySnapshot) -> Self {
        Self {
            price: initial_price,
            volatility: initial_volatility,
            liquidity,
            step: 0,
        }
    }

    /// Process an event and return any follow-on events to inject into
    /// the queue.
    pub fn handle(&mut self, tse: &TimestampedEvent) -> Vec<(f64, SimEvent)> {
        match &tse.event {
            SimEvent::PriceUpdate { price, volatility } => {
                self.price = *price;
                self.volatility = *volatility;
                vec![]
            }
            SimEvent::MarketTrade { qty: _, price } => {
                // External trade shifts the mid-price slightly.
                self.price = *price;
                vec![]
            }
            SimEvent::TimeTick { step } => {
                self.step = *step;
                vec![]
            }
            _ => vec![],
        }
    }
}
