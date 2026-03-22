use crate::events::event::Event;

/// Abstract representation of an execution fill.
#[derive(Debug, Clone)]
pub struct Fill {
    pub order_id: u64,
    pub strategy: String,
    pub quantity: f64,
    pub price: f64,
    pub impact: f64,
}

impl Fill {
    pub fn new(order_id: u64, strategy: String, quantity: f64, price: f64, impact: f64) -> Self {
        Self {
            order_id,
            strategy,
            quantity,
            price,
            impact,
        }
    }

    /// Convert a fill directly to a structured Simulation Event.
    pub fn into_event(self) -> Event {
        Event::OrderFilled {
            order_id: self.order_id,
            qty: self.quantity,
            price: self.price,
            impact: self.impact,
        }
    }
}
