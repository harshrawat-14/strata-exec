use crate::events::event::SimEvent;
use crate::events::order::Order;
use crate::market::state::MarketState;

// ---------------------------------------------------------------------------
// Strategy trait
// ---------------------------------------------------------------------------

/// Event-reactive strategy interface for the DES engine.
///
/// Strategies observe simulation events and react by emitting zero or more
/// `Order`s that will be wrapped as `SubmitOrder` events and injected into
/// the event queue.
pub trait Strategy {
    /// Human-readable name (e.g. "Heuristic", "Optimal").
    fn name(&self) -> &str;

    /// React to a simulation event.
    ///
    /// Called for *every* event in the DES loop.  The strategy inspects
    /// the event and current market state, then returns any orders it
    /// wishes to submit.
    fn on_event(&mut self, event: &SimEvent, market: &MarketState) -> Vec<Order>;

    /// Returns `true` once the strategy has fully executed its inventory.
    fn is_complete(&self) -> bool;
}
