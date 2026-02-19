use thiserror::Error;

/// Top-level error type for the StrataExec engine.
#[derive(Debug, Error)]
pub enum StrataError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("blockchain error: {0}")]
    Blockchain(String),

    #[error("engine error: {0}")]
    Engine(String),
}
