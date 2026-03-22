use std::future::Future;
use std::time::Duration;

use async_trait::async_trait;
use ethers::prelude::{abigen, Middleware};
use ethers::providers::{Http, Provider};
use ethers::types::{Address, U256};

/// Maximum number of retry attempts before giving up.
const MAX_RETRIES: u32 = 5;

/// Base delay for exponential backoff (doubles on each retry).
const BASE_DELAY_MS: u64 = 100;

/// Defines the interface for read-only blockchain interactions.
#[async_trait]
pub trait BlockchainClient: Send + Sync {
    /// Retrieve the latest block number.
    async fn get_block_number(&self) -> anyhow::Result<u64>;

    /// Retrieve the current gas price.
    async fn get_gas_price(&self) -> anyhow::Result<U256>;

    /// Fetch Uniswap V2 pair reserves (reserve0, reserve1).
    async fn get_uniswap_v2_reserves(&self, pair: Address) -> anyhow::Result<(u128, u128)>;

    /// Fetch the `token0` address from a Uniswap V2 pair contract.
    async fn get_token0(&self, pair: Address) -> anyhow::Result<Address>;

    /// Fetch the `token1` address from a Uniswap V2 pair contract.
    async fn get_token1(&self, pair: Address) -> anyhow::Result<Address>;

    /// Fetch the `decimals()` value for an ERC-20 token.
    async fn get_erc20_decimals(&self, token: Address) -> anyhow::Result<u8>;
}

// Generate minimal ABI bindings for Uniswap V2 Pair view functions.
abigen!(
    IUniswapV2Pair,
    r#"[
        function getReserves() external view returns (uint112, uint112, uint32)
        function token0() external view returns (address)
        function token1() external view returns (address)
    ]"#
);

// Generate minimal ABI binding for the ERC-20 `decimals()` view function.
abigen!(
    IERC20Metadata,
    r#"[function decimals() external view returns (uint8)]"#
);

/// JSON-RPC based blockchain client backed by `ethers::Provider<Http>`.
pub struct RpcClient {
    provider: Provider<Http>,
}

impl RpcClient {
    /// Create a new client with the given RPC endpoint URL.
    pub fn new(rpc_url: &str) -> anyhow::Result<Self> {
        let provider = Provider::<Http>::try_from(rpc_url)?;
        Ok(Self { provider })
    }
}

#[async_trait]
impl BlockchainClient for RpcClient {
    async fn get_block_number(&self) -> anyhow::Result<u64> {
        retry_with_backoff("get_block_number", || async {
            let block = self.provider.get_block_number().await?;
            Ok(block.as_u64())
        })
        .await
    }

    async fn get_gas_price(&self) -> anyhow::Result<U256> {
        retry_with_backoff("get_gas_price", || async {
            let price = self.provider.get_gas_price().await?;
            Ok(price)
        })
        .await
    }

    async fn get_uniswap_v2_reserves(&self, pair: Address) -> anyhow::Result<(u128, u128)> {
        retry_with_backoff("get_uniswap_v2_reserves", || async {
            let contract = IUniswapV2Pair::new(pair, self.provider.clone().into());
            let (r0, r1, _timestamp) = contract.get_reserves().call().await?;
            Ok((r0, r1))
        })
        .await
    }

    async fn get_token0(&self, pair: Address) -> anyhow::Result<Address> {
        retry_with_backoff("get_token0", || async {
            let contract = IUniswapV2Pair::new(pair, self.provider.clone().into());
            let addr = contract.token_0().call().await?;
            Ok(addr)
        })
        .await
    }

    async fn get_token1(&self, pair: Address) -> anyhow::Result<Address> {
        retry_with_backoff("get_token1", || async {
            let contract = IUniswapV2Pair::new(pair, self.provider.clone().into());
            let addr = contract.token_1().call().await?;
            Ok(addr)
        })
        .await
    }

    async fn get_erc20_decimals(&self, token: Address) -> anyhow::Result<u8> {
        retry_with_backoff("get_erc20_decimals", || async {
            let contract = IERC20Metadata::new(token, self.provider.clone().into());
            let decimals = contract.decimals().call().await?;
            Ok(decimals)
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// Retry helper
// ---------------------------------------------------------------------------

/// Retry an async operation with exponential backoff.
///
/// - Up to [`MAX_RETRIES`] attempts (100ms → 200ms → 400ms → 800ms → 1600ms).
/// - Logs a warning on each retry and an error if all attempts fail.
/// - Returns the last error to the caller (never panics).
async fn retry_with_backoff<T, F, Fut>(op_name: &str, f: F) -> anyhow::Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = anyhow::Result<T>>,
{
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 0..MAX_RETRIES {
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                let delay = Duration::from_millis(BASE_DELAY_MS * 2u64.pow(attempt));
                tracing::warn!(
                    op = op_name,
                    attempt = attempt + 1,
                    max = MAX_RETRIES,
                    delay_ms = delay.as_millis() as u64,
                    error = %e,
                    "RPC call failed, retrying",
                );
                tokio::time::sleep(delay).await;
                last_err = Some(e);
            }
        }
    }

    match last_err {
        Some(err) => {
            tracing::error!(op = op_name, "RPC call failed after {MAX_RETRIES} retries",);
            Err(err)
        }
        None => anyhow::bail!("{op_name}: retry loop completed with no attempts"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// In-memory mock for unit tests — no network required.
    struct MockClient {
        block_counter: AtomicU64,
    }

    impl MockClient {
        fn new() -> Self {
            Self {
                block_counter: AtomicU64::new(1),
            }
        }
    }

    #[async_trait]
    impl BlockchainClient for MockClient {
        async fn get_block_number(&self) -> anyhow::Result<u64> {
            Ok(self.block_counter.fetch_add(1, Ordering::SeqCst))
        }

        async fn get_gas_price(&self) -> anyhow::Result<U256> {
            Ok(U256::from(20_000_000_000u64)) // 20 gwei
        }

        async fn get_uniswap_v2_reserves(&self, _pair: Address) -> anyhow::Result<(u128, u128)> {
            Ok((1_000_000, 2_000_000))
        }

        async fn get_token0(&self, _pair: Address) -> anyhow::Result<Address> {
            Ok(Address::zero())
        }

        async fn get_token1(&self, _pair: Address) -> anyhow::Result<Address> {
            Ok(Address::zero())
        }

        async fn get_erc20_decimals(&self, _token: Address) -> anyhow::Result<u8> {
            Ok(18)
        }
    }

    #[tokio::test]
    async fn mock_client_returns_incrementing_blocks() {
        let client = MockClient::new();
        let b1 = client.get_block_number().await.expect("should succeed");
        let b2 = client.get_block_number().await.expect("should succeed");
        assert_eq!(b2, b1 + 1);
    }

    #[tokio::test]
    async fn mock_client_returns_gas_price() {
        let client = MockClient::new();
        let price = client.get_gas_price().await.expect("should succeed");
        assert_eq!(price, U256::from(20_000_000_000u64));
    }

    #[tokio::test]
    async fn mock_client_returns_reserves() {
        let client = MockClient::new();
        let (r0, r1) = client
            .get_uniswap_v2_reserves(Address::zero())
            .await
            .expect("should succeed");
        assert_eq!(r0, 1_000_000);
        assert_eq!(r1, 2_000_000);
    }
}
