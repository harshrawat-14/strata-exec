#![allow(dead_code)]

mod api;
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

use tokio::net::TcpListener;

use crate::api::AppState;
use crate::blockchain::client::RpcClient;
use crate::core::engine::Engine;
use crate::core::risk::RiskManager;
use crate::core::scheduler::{ExecutionMode, Scheduler};
use crate::features::volatility::VolatilityEstimator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    utils::logging::init();

    tracing::info!("StrataExec starting...");

    let cfg = config::AppConfig::load()?;
    tracing::info!(rpc_url = %cfg.rpc_url, pair = ?cfg.pair_address, "config loaded");

    // ── Shared state ───────────────────────────────────────────────
    let app_state = Arc::new(AppState::new(cfg.total_notional));

    // ── Core components ────────────────────────────────────────────
    let client = Arc::new(RpcClient::new(&cfg.rpc_url)?);

    let risk = RiskManager::new(cfg.total_notional, cfg.max_slippage_pct, cfg.emergency_reserve_frac);

    let exec_mode = match cfg.execution_mode.as_str() {
        "optimal" => ExecutionMode::Optimal,
        _ => ExecutionMode::Heuristic,
    };

    let scheduler = Scheduler::new(
        risk,
        cfg.total_notional,
        cfg.base_chunk,
        cfg.sigma_ref,
        cfg.liquidity_ref,
        exec_mode,
        cfg.eta,
        cfg.lambda,
    );
    let vol_estimator = VolatilityEstimator::new(cfg.vol_window);

    let mut engine = Engine::new(
        client,
        Duration::from_secs(2),
        scheduler,
        vol_estimator,
        cfg.pair_address,
        app_state.clone(),
    );

    // ── HTTP server ────────────────────────────────────────────────
    let router = api::build_router(app_state.clone());
    let bind_addr = format!("0.0.0.0:{}", cfg.http_port);
    let listener = TcpListener::bind(&bind_addr).await?;
    tracing::info!(addr = %bind_addr, "HTTP server listening");

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, router)
            .await
            .map_err(|e| anyhow::anyhow!("server error: {e}"))
    });

    // ── Engine loop ────────────────────────────────────────────────
    let engine_handle = tokio::spawn(async move { engine.run().await });

    // ── Graceful shutdown ──────────────────────────────────────────
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("ctrl-c received, shutting down");
        }
        result = server_handle => {
            match result {
                Ok(Ok(())) => tracing::info!("HTTP server exited"),
                Ok(Err(e)) => tracing::error!(%e, "HTTP server error"),
                Err(e) => tracing::error!(%e, "HTTP server task panicked"),
            }
        }
        result = engine_handle => {
            match result {
                Ok(Ok(())) => tracing::info!("engine exited"),
                Ok(Err(e)) => tracing::error!(%e, "engine error"),
                Err(e) => tracing::error!(%e, "engine task panicked"),
            }
        }
    }

    tracing::info!("StrataExec shutdown complete");
    Ok(())
}
