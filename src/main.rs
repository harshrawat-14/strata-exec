#![allow(dead_code)]

mod config;
mod core;
mod features;
mod ml;
mod blockchain;
mod types;
mod utils;
mod error;

use std::sync::Arc;
use std::time::Duration;

use crate::blockchain::client::RpcClient;
use crate::core::engine::Engine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    utils::logging::init();

    tracing::info!("StrataExec starting...");

    let client = Arc::new(RpcClient::new("http://localhost:8545".into()));
    let _engine = Engine::new(client, Duration::from_secs(2));

    Ok(())
}
