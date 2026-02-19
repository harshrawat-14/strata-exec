use std::sync::Arc;
use std::time::Duration;

use tokio::time::interval;

use super::state::EngineState;
use crate::blockchain::client::BlockchainClient;

/// Central orchestrator for the execution pipeline.
pub struct Engine {
    client: Arc<dyn BlockchainClient>,
    state: EngineState,
    last_seen_block: u64,
    poll_interval: Duration,
}

impl Engine {
    pub fn new(client: Arc<dyn BlockchainClient>, poll_interval: Duration) -> Self {
        Self {
            client,
            state: EngineState::Idle,
            last_seen_block: 0,
            poll_interval,
        }
    }

    /// Returns the current engine state.
    pub fn state(&self) -> &EngineState {
        &self.state
    }

    /// Returns the last observed block number.
    pub fn last_seen_block(&self) -> u64 {
        self.last_seen_block
    }

    /// Start the block-polling loop. Runs indefinitely.
    ///
    /// Uses `tokio::time::interval` for drift-free cadence — the tick
    /// accounts for processing time so polls stay evenly spaced.
    ///
    // Future: wrap the inner loop with `tokio::select!` to support
    // graceful shutdown via a `tokio::sync::watch` or `CancellationToken`.
    pub async fn run(&mut self) -> anyhow::Result<()> {
        self.transition(EngineState::Running);

        let mut ticker = interval(self.poll_interval);

        loop {
            ticker.tick().await;

            let block_number = self.client.get_block_number().await?;

            if block_number > self.last_seen_block {
                self.last_seen_block = block_number;
                self.on_new_block(block_number).await?;
            }
        }
    }

    /// Centralised state transition — single choke-point for future
    /// logging, validation, or event emission on state changes.
    fn transition(&mut self, new_state: EngineState) {
        tracing::debug!(
            from = ?self.state,
            to = ?new_state,
            "engine state transition",
        );
        self.state = new_state;
    }

    /// Called when a new block is observed.
    async fn on_new_block(&mut self, block_number: u64) -> anyhow::Result<()> {
        tracing::info!(block_number, "new block detected");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    use async_trait::async_trait;

    /// Mock client that returns an atomically incrementing block number.
    struct MockBlockchainClient {
        counter: AtomicU64,
    }

    impl MockBlockchainClient {
        fn new() -> Self {
            Self {
                counter: AtomicU64::new(0),
            }
        }
    }

    #[async_trait]
    impl BlockchainClient for MockBlockchainClient {
        async fn get_block_number(&self) -> anyhow::Result<u64> {
            Ok(self.counter.fetch_add(1, Ordering::SeqCst) + 1)
        }

        async fn get_gas_price(&self) -> anyhow::Result<u64> {
            Ok(0)
        }
    }

    /// Deterministic test using tokio virtual time — completes in ~0ms wall-clock.
    #[tokio::test]
    async fn engine_tracks_new_blocks() {
        tokio::time::pause();

        let client = Arc::new(MockBlockchainClient::new());
        let mut engine = Engine::new(client, Duration::from_millis(100));

        assert_eq!(engine.last_seen_block(), 0);
        assert_eq!(*engine.state(), EngineState::Idle);

        // Virtual time: 500ms timeout with 100ms interval → at least 5 ticks.
        let result = tokio::time::timeout(Duration::from_millis(500), engine.run()).await;

        // The engine loop is infinite, so we expect a timeout.
        assert!(result.is_err(), "engine should still be running at timeout");

        // With 100ms interval over 500ms, expect at least 5 block updates.
        assert!(
            engine.last_seen_block() >= 5,
            "expected at least 5 blocks, got {}",
            engine.last_seen_block(),
        );
    }
}

