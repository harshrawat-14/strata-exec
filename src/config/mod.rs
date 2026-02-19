use serde::Deserialize;

/// Application-level configuration.
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub rpc_url: String,
    pub chain_id: u64,
}

impl AppConfig {
    /// Load configuration with sensible defaults.
    pub fn load() -> Self {
        Self {
            rpc_url: String::from("http://localhost:8545"),
            chain_id: 1,
        }
    }
}
