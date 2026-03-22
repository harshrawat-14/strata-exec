use super::order::Order;
use std::fmt;

// ---------------------------------------------------------------------------
// Core simulation event
// ---------------------------------------------------------------------------

/// Every interaction in the discrete event simulator is represented as a
/// `SimEvent`.  Events are placed on a priority queue keyed by timestamp
/// and processed in chronological order.
#[derive(Debug, Clone)]
pub enum SimEvent {
    /// A new price observation from the market model.
    PriceUpdate { price: f64, volatility: f64 },

    /// An exogenous market trade (not originated by our strategies).
    MarketTrade { qty: f64, price: f64 },

    /// A strategy wishes to submit an order to the execution engine.
    SubmitOrder { order: Order },

    /// A strategy wishes to cancel a previously submitted order.
    CancelOrder { order_id: u64 },

    /// The execution engine has fully filled an order.
    OrderFilled {
        order_id: u64,
        qty: f64,
        price: f64,
        impact: f64,
    },

    /// The execution engine has partially filled an order.
    PartialFill {
        order_id: u64,
        filled: f64,
        remaining: f64,
        price: f64,
    },

    /// A discrete time tick — used to drive periodic strategy decisions.
    TimeTick { step: usize },

    /// Emitted when a multi-path monte-carlo simulation starts a new path.
    PathStart { path_id: usize },

    /// Emitted when a monte-carlo simulation finishes a particular path.
    PathEnd { path_id: usize },

    /// Emitted periodically to communicate monte-carlo progress.
    Progress { completed: usize, total: usize },

    /// Signals the end of the simulation run.
    SimulationEnd,
}

impl fmt::Display for SimEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PriceUpdate { price, volatility } => {
                write!(
                    f,
                    "[PRICE]      price={price:.4}  vol={:.4}%",
                    volatility * 100.0
                )
            }
            Self::MarketTrade { qty, price } => {
                write!(f, "[MKT_TRADE]  qty={qty:.4}  price={price:.4}")
            }
            Self::SubmitOrder { order } => {
                write!(
                    f,
                    "[SUBMIT]     id={}  strategy={}  qty={:.4}",
                    order.id, order.strategy_name, order.quantity
                )
            }
            Self::CancelOrder { order_id } => {
                write!(f, "[CANCEL]     id={order_id}")
            }
            Self::OrderFilled {
                order_id,
                qty,
                price,
                impact,
            } => {
                write!(
                    f,
                    "[FILLED]     id={order_id}  qty={qty:.4}  price={price:.4}  impact={impact:.6}"
                )
            }
            Self::PartialFill {
                order_id,
                filled,
                remaining,
                price,
            } => {
                write!(
                    f,
                    "[PARTIAL]    id={order_id}  filled={filled:.4}  remaining={remaining:.4}  price={price:.4}"
                )
            }
            Self::TimeTick { step } => {
                write!(f, "[TICK]       step={step}")
            }
            Self::PathStart { path_id } => {
                write!(f, "[PATH_START] id={path_id}")
            }
            Self::PathEnd { path_id } => {
                write!(f, "[PATH_END]   id={path_id}")
            }
            Self::Progress { completed, total } => {
                write!(f, "[{} / {} paths completed]", completed, total)
            }
            Self::SimulationEnd => {
                write!(f, "[SIM_END]")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy aliases — keep the observability logger compiling
// ---------------------------------------------------------------------------

/// Re-export under the old name so `observability::logger` and
/// `research::multi_runner` continue to compile without modification.
pub type Event = SimEvent;
pub type EventSender = crossbeam_channel::Sender<SimEvent>;
pub type EventReceiver = crossbeam_channel::Receiver<SimEvent>;
