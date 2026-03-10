/// Structured debug events emitted by simulation components.
///
/// Each variant captures a specific moment in the simulation lifecycle.
/// Events are designed to be lightweight (stack-allocated, no heap
/// allocations for the common case) so that emitting them has near-zero
/// cost on the hot path.

use std::fmt;

/// Type alias for the sending half of the debug channel.
pub type EventSender = crossbeam_channel::Sender<DebugEvent>;

/// Type alias for the receiving half of the debug channel.
pub type EventReceiver = crossbeam_channel::Receiver<DebugEvent>;

/// A single debug event from the simulation.
#[derive(Debug, Clone)]
pub enum DebugEvent {
    /// Emitted once at the start of the entire simulation run.
    SimulationStart {
        model: String,
        paths: usize,
    },

    /// Emitted at the start of each Monte Carlo path.
    PathStart {
        path_id: usize,
    },

    /// Emitted on every price-model step.
    PriceUpdate {
        step: usize,
        price: f64,
        volatility: f64,
    },

    /// Emitted after a strategy decides on a trade size.
    StrategyDecision {
        strategy: String,
        order_size: f64,
        remaining: f64,
    },

    /// Emitted after a trade is executed through the AMM.
    TradeExecuted {
        strategy: String,
        price: f64,
        quantity: f64,
        impact: f64,
    },

    /// Emitted at the end of each Monte Carlo path.
    PathEnd {
        path_id: usize,
    },

    /// Emitted once at the end of the entire simulation run.
    SimulationEnd,
}

impl fmt::Display for DebugEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SimulationStart { model, paths } => {
                write!(f, "[SIM_START]  model={model}  paths={paths}")
            }
            Self::PathStart { path_id } => {
                write!(f, "[PATH_START] path={path_id}")
            }
            Self::PriceUpdate { step, price, volatility } => {
                write!(
                    f,
                    "[PRICE]      step={step:<4}  price={price:.4}  vol={:.4}%",
                    volatility * 100.0
                )
            }
            Self::StrategyDecision { strategy, order_size, remaining } => {
                write!(
                    f,
                    "[DECISION]   {strategy:<18} order={order_size:.4}  remaining={remaining:.4}"
                )
            }
            Self::TradeExecuted { strategy, price, quantity, impact } => {
                write!(
                    f,
                    "[TRADE]      {strategy:<18} qty={quantity:.4}  price={price:.4}  impact={impact:.6}"
                )
            }
            Self::PathEnd { path_id } => {
                write!(f, "[PATH_END]   path={path_id}")
            }
            Self::SimulationEnd => {
                write!(f, "[SIM_END]")
            }
        }
    }
}
