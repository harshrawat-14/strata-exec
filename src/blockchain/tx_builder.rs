use ethers::types::{TransactionRequest, U256};

/// Constructs unsigned transactions for on-chain execution.
pub struct TxBuilder {
    chain_id: u64,
}

impl TxBuilder {
    pub fn new(chain_id: u64) -> Self {
        Self { chain_id }
    }

    /// Build a placeholder transaction request.
    pub fn build(&self, to: ethers::types::Address, value: U256) -> TransactionRequest {
        TransactionRequest::new()
            .to(to)
            .value(value)
            .chain_id(self.chain_id)
    }
}
