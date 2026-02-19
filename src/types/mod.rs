use serde::{Deserialize, Serialize};

/// Represents a unique identifier for a strategy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StrategyId(pub String);

/// Represents a normalised signal value in the range [0.0, 1.0].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Signal(pub f64);
