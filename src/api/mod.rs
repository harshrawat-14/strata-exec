use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::sync::RwLock;

/// Shared application state exposed to both the HTTP layer and the engine.
///
/// All fields use lock-free atomics or async-aware `RwLock` so readers
/// never block the engine and vice-versa.
pub struct AppState {
    pub is_running: AtomicBool,
    pub blocks_processed: AtomicU64,
    pub chunks_executed: AtomicU64,
    pub risk_denials: AtomicU64,
    pub last_volatility: RwLock<f64>,
    pub remaining_notional: RwLock<f64>,
}

impl AppState {
    /// Create a new `AppState` with sensible defaults.
    pub fn new(initial_notional: f64) -> Self {
        Self {
            is_running: AtomicBool::new(true),
            blocks_processed: AtomicU64::new(0),
            chunks_executed: AtomicU64::new(0),
            risk_denials: AtomicU64::new(0),
            last_volatility: RwLock::new(0.0),
            remaining_notional: RwLock::new(initial_notional),
        }
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the Axum router with all monitoring / control routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/metrics", get(metrics))
        .route("/start", post(start))
        .route("/stop", post(stop))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn status(State(state): State<Arc<AppState>>) -> Json<Value> {
    let is_running = state.is_running.load(Ordering::Relaxed);
    let remaining_notional = *state.remaining_notional.read().await;
    let last_volatility = *state.last_volatility.read().await;

    Json(json!({
        "is_running": is_running,
        "remaining_notional": remaining_notional,
        "last_volatility": last_volatility,
    }))
}

async fn metrics(State(state): State<Arc<AppState>>) -> Json<Value> {
    let blocks_processed = state.blocks_processed.load(Ordering::Relaxed);
    let chunks_executed = state.chunks_executed.load(Ordering::Relaxed);
    let risk_denials = state.risk_denials.load(Ordering::Relaxed);

    Json(json!({
        "blocks_processed": blocks_processed,
        "chunks_executed": chunks_executed,
        "risk_denials": risk_denials,
    }))
}

async fn start(State(state): State<Arc<AppState>>) -> (StatusCode, Json<Value>) {
    state.is_running.store(true, Ordering::Relaxed);
    (StatusCode::OK, Json(json!({ "is_running": true })))
}

async fn stop(State(state): State<Arc<AppState>>) -> (StatusCode, Json<Value>) {
    state.is_running.store(false, Ordering::Relaxed);
    (StatusCode::OK, Json(json!({ "is_running": false })))
}
