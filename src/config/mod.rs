use std::env;
use std::str::FromStr;

use anyhow::{bail, Context};
use ethers::types::Address;

/// Production-grade application configuration loaded from environment variables.
#[derive(Debug)]
pub struct AppConfig {
    pub rpc_url: String,
    pub pair_address: Address,
    pub total_notional: f64,
    pub base_chunk: f64,
    pub max_slippage_pct: f64,
    pub emergency_reserve_frac: f64,
    pub vol_window: usize,
    pub sigma_ref: f64,
    pub liquidity_ref: f64,
    pub log_level: String,
    pub http_port: u16,
    pub execution_mode: String,
    pub eta: f64,
    pub lambda: f64,
    pub horizon_blocks: Option<usize>,
}

impl AppConfig {
    /// Load configuration from environment variables (and `.env` file if present).
    ///
    /// Returns a descriptive error if any required variable is missing, cannot be
    /// parsed, or fails validation constraints.
    pub fn load() -> anyhow::Result<Self> {
        // Best-effort: load .env; missing file is fine.
        dotenvy::dotenv().ok();

        // --- required string vars ---
        let rpc_url = env_required("RPC_URL")?;
        let pair_address_raw = env_required("PAIR_ADDRESS")?;
        let log_level = env_or("LOG_LEVEL", "info");
        let http_port: u16 = env_or("HTTP_PORT", "3000")
            .parse()
            .with_context(|| "failed to parse HTTP_PORT")?;

        // --- parse address ---
        let pair_address = Address::from_str(&pair_address_raw)
            .with_context(|| format!("PAIR_ADDRESS is not a valid Ethereum address: {pair_address_raw}"))?;

        // --- parse numerics ---
        let total_notional = parse_env_f64("TOTAL_NOTIONAL")?;
        let base_chunk = parse_env_f64("BASE_CHUNK")?;
        let max_slippage_pct = parse_env_f64("MAX_SLIPPAGE_PCT")?;
        let emergency_reserve_frac = parse_env_f64("EMERGENCY_RESERVE_FRAC")?;
        let vol_window = parse_env::<usize>("VOL_WINDOW")?;
        let sigma_ref = parse_env_f64("SIGMA_REF")?;
        let liquidity_ref = parse_env_f64("LIQUIDITY_REF")?;

        // --- execution model ---
        let execution_mode = env_or("EXECUTION_MODE", "heuristic").to_lowercase();
        let eta = parse_env_or_default("ETA", 0.0)?;
        let lambda = parse_env_or_default("LAMBDA", 0.0)?;
        let horizon_blocks = match env::var("HORIZON_BLOCKS") {
            Ok(v) => Some(v.parse::<usize>()
                .with_context(|| format!("failed to parse HORIZON_BLOCKS={v}"))?),
            Err(_) => None,
        };

        // --- validation ---
        if total_notional <= 0.0 {
            bail!("TOTAL_NOTIONAL must be > 0, got {total_notional}");
        }
        if base_chunk <= 0.0 {
            bail!("BASE_CHUNK must be > 0, got {base_chunk}");
        }
        if max_slippage_pct <= 0.0 || max_slippage_pct >= 1.0 {
            bail!("MAX_SLIPPAGE_PCT must be in (0, 1), got {max_slippage_pct}");
        }
        if emergency_reserve_frac < 0.0 || emergency_reserve_frac >= 1.0 {
            bail!("EMERGENCY_RESERVE_FRAC must be in [0, 1), got {emergency_reserve_frac}");
        }
        if vol_window < 2 {
            bail!("VOL_WINDOW must be >= 2, got {vol_window}");
        }
        if http_port == 0 {
            bail!("HTTP_PORT must be > 0");
        }
        if execution_mode != "heuristic" && execution_mode != "optimal" {
            bail!("EXECUTION_MODE must be 'heuristic' or 'optimal', got '{execution_mode}'");
        }
        if execution_mode == "optimal" && eta <= 0.0 {
            bail!("ETA must be > 0 for optimal mode, got {eta}");
        }
        if lambda < 0.0 {
            bail!("LAMBDA must be >= 0, got {lambda}");
        }

        Ok(Self {
            rpc_url,
            pair_address,
            total_notional,
            base_chunk,
            max_slippage_pct,
            emergency_reserve_frac,
            vol_window,
            sigma_ref,
            liquidity_ref,
            log_level,
            http_port,
            execution_mode,
            eta,
            lambda,
            horizon_blocks,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a required environment variable, returning a descriptive error if absent.
fn env_required(key: &str) -> anyhow::Result<String> {
    env::var(key).with_context(|| format!("missing required environment variable: {key}"))
}

/// Read an environment variable with a fallback default.
fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Read and parse an environment variable into the requested type.
fn parse_env<T>(key: &str) -> anyhow::Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let raw = env_required(key)?;
    raw.parse::<T>()
        .with_context(|| format!("failed to parse {key}={raw}"))
}

/// Convenience wrapper for `f64` parsing.
fn parse_env_f64(key: &str) -> anyhow::Result<f64> {
    parse_env::<f64>(key)
}

/// Parse an env var with a fallback default value.
fn parse_env_or_default<T>(key: &str, default: T) -> anyhow::Result<T>
where
    T: std::str::FromStr + std::fmt::Display,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let raw = env_or(key, &default.to_string());
    raw.parse::<T>()
        .with_context(|| format!("failed to parse {key}={raw}"))
}
