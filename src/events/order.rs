use std::fmt;

// ---------------------------------------------------------------------------
// Order side
// ---------------------------------------------------------------------------

/// Direction of an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
        }
    }
}

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------

/// A single order submitted by a strategy to the execution engine.
#[derive(Debug, Clone)]
pub struct Order {
    /// Unique identifier (monotonically increasing within a simulation).
    pub id: u64,
    /// Which strategy originated this order.
    pub strategy_name: String,
    /// Buy or sell.
    pub side: Side,
    /// Outstanding quantity (reduced on partial fills).
    pub quantity: f64,
    /// Optional limit price.  `None` means market order.
    pub limit_price: Option<f64>,
}

impl Order {
    pub fn market_sell(id: u64, strategy_name: String, quantity: f64) -> Self {
        Self {
            id,
            strategy_name,
            side: Side::Sell,
            quantity,
            limit_price: None,
        }
    }

    pub fn market_buy(id: u64, strategy_name: String, quantity: f64) -> Self {
        Self {
            id,
            strategy_name,
            side: Side::Buy,
            quantity,
            limit_price: None,
        }
    }
}
