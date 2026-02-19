use async_trait::async_trait;

/// Defines the interface for blockchain interactions.
#[async_trait]
pub trait BlockchainClient: Send + Sync {
    /// Retrieve the latest block number.
    async fn get_block_number(&self) -> anyhow::Result<u64>;
    async fn get_gas_price(&self) -> anyhow::Result<u64>;
}

/// JSON-RPC based blockchain client.
pub struct RpcClient {
    rpc_url: String,
}

impl RpcClient {
    pub fn new(rpc_url: String) -> Self {
        Self { rpc_url }
    }

    /// Returns the configured RPC endpoint URL.
    pub fn rpc_url(&self) -> &str {
        &self.rpc_url
    }
}

#[async_trait]
impl BlockchainClient for RpcClient {
    async fn get_block_number(&self) -> anyhow::Result<u64> {
        Ok(0)
    }

    async fn get_gas_price(&self) -> anyhow::Result<u64> {
        Ok(0)
    }
}
