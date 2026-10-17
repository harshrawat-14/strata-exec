//! Constants, the environment configuration, and CLI parsing.




/// Penalty reward applied when an episode is force-terminated by a data-ingestion
/// failure (missing/corrupt historical file, zero valid snapshots) rather than
/// crashing the whole training process. Deliberately an order of magnitude larger
/// in absolute value than a normal step reward (see the reward-scale diagnostic in
/// `RlEnv::reset`, ~1e-4 to 1e-2), so it is unambiguously distinguishable from
/// ordinary reward noise in training logs, without being so extreme it destabilizes
/// return statistics if it fires legitimately in production.
pub(crate) const DATA_INGESTION_FAILURE_PENALTY: f64 = -1.0;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Discrete action → fraction of remaining inventory to trade.
/// Log-scale spacing gives resolution where it matters (small trades).
pub const ACTION_FRACTIONS: [f64; 20] = [
    0.000, 0.001, 0.003, 0.005, 0.008,
    0.010, 0.015, 0.020, 0.030, 0.050,
    0.070, 0.100, 0.150, 0.200, 0.300,
    0.400, 0.500, 0.700, 0.900, 1.000,
];

/// Default permanent-impact fraction; full-inventory sell causes
/// `GAMMA × 100%` price depression.
pub(crate) const DEFAULT_GAMMA: f64 = 0.005;

/// Default annualised volatility (synthetic mode only — GBM/GARCH book
/// generation, unrelated to real dates, deliberately left untouched).
pub(crate) const DEFAULT_SIGMA: f64 = 0.20;

/// Calibrated `sigma_ref` for historical/counterfactual mode, replacing the
/// synthetic-mode 0.20 default that Phase 3 found silently pinned s10
/// (vol_regime) at LOW on every real date, calm or crash (see
/// docs/proposals/research-roadmap.md, section C — bug-class instance #4).
///
/// Derivation: annualised realised volatility (log-returns, 30s bars, from
/// `TradeData/AggTrades`), computed with the SAME annualisation this file
/// now uses (`TRADING_STEPS_PER_YEAR`, fixed at one trading day × 365 —
/// see below), then the median across the 7 `TRAINING_DATES` in
/// `rl/evaluate.py` — NOT `HELD_OUT_CRASH_DATES`, which would be the same
/// hindsight leak already killed for λ. Per-date values: 2024-01-15 0.446,
/// 02-20 0.500, 04-15 0.772, 06-10 0.260, 09-15 0.316, 10-15 0.600,
/// 11-06 0.859. Mean 0.536, median 0.500 (robust: excluding 11-06, the
/// post-election outlier, median is 0.473 — still close). Frozen like λ:
/// calibrated once from training data only, never touched per-eval-date.
pub(crate) const CALIBRATED_SIGMA_REF: f64 = 0.4999;

/// Number of initial books used to compute `depth_ref`.
pub(crate) const DEPTH_REF_SAMPLES: usize = 50;

/// Default step count — one full trading day at ~30s cadence.
pub(crate) const DEFAULT_TOTAL_STEPS: usize = 2880;
/// Historical mode default step count (one trading day ≈ 30s cadence).
pub(crate) const HISTORICAL_DEFAULT_STEPS: usize = 2880;

/// Trading periods used to annualise the rolling realised-vol estimator
/// (`realised_vol()`). Fixed at one trading day (`HISTORICAL_DEFAULT_STEPS`
/// steps, ~30s cadence) × 365 trading days/year — deliberately NOT
/// `cfg.total_steps`, which is the per-episode horizon and is randomised
/// 200-1440 during training (`RANDOM_HORIZON_MIN`/`MAX`). Annualising
/// against the episode horizon would make `current_sigma` swing with
/// episode length alone, for identical underlying market volatility.
pub(crate) const TRADING_STEPS_PER_YEAR: f64 = HISTORICAL_DEFAULT_STEPS as f64 * 365.0;

/// Synthetic LOB depth scaling so the book can absorb realistic order sizes.
/// `base_quantity = total_inventory / SYNTHETIC_DEPTH_SCALE` per level.
pub(crate) const SYNTHETIC_DEPTH_SCALE: f64 = 5.0;
/// Number of levels per side in the synthetic LOB (deeper than the default 10).
pub(crate) const SYNTHETIC_NUM_LEVELS: usize = 20;

// ── Per-episode randomization ranges (Changes 1 & 4) ───────────────────────
/// Minimum random execution horizon (steps). At 200, action 9 (5%) still leaves
/// most inventory if used naively — the agent has to escalate to larger actions.
pub(crate) const RANDOM_HORIZON_MIN: usize = 200;
/// Maximum random execution horizon (steps).
pub(crate) const RANDOM_HORIZON_MAX: usize = 1_440;
/// Minimum random order size as a fraction of the configured `total_inventory`.
pub(crate) const RANDOM_SIZE_MIN: f64 = 0.1;
/// Maximum random order size as a fraction of the configured `total_inventory`.
pub(crate) const RANDOM_SIZE_MAX: f64 = 1.0;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum RlMode {
    Synthetic,
    Historical,
    Counterfactual,
}

#[derive(Clone, Debug)]
pub struct EnvConfig {
    pub mode: RlMode,
    pub total_inventory: f64,
    pub total_steps: usize,
    pub seed: u64,
    pub sigma_ref: f64,
    pub gamma: f64,
    pub adversarial: bool,
    pub lob_data_path: Option<String>,
    /// BTC equivalent of total_inventory (counterfactual mode).
    pub btc_target: f64,
    /// ADV in BTC; 0.0 means use the default (240_000.0).
    pub adv_btc: f64,
    pub agg_trades_path: Option<String>,
    /// When true, use the configured `total_steps` every episode (eval mode).
    /// When false (default), draw a fresh random horizon per episode in replay modes.
    pub fixed_steps: bool,
    /// When true, use the configured `total_inventory` every episode (eval mode).
    /// When false (default), draw a fresh random order size per episode in replay modes.
    pub fixed_size: bool,
    /// When true, build_state() returns the 12-dim v5 vector (no intraday time features).
    /// When false (default), build_state() appends 4 time-of-day dims -> 16-dim v6 vector.
    /// Existing v4/v5 evaluation must pass --no-time-features to keep observations
    /// dimension-compatible with checkpoints trained before this field existed.
    pub no_time_features: bool,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            mode: RlMode::Synthetic,
            total_inventory: 1_000_000.0,
            total_steps: DEFAULT_TOTAL_STEPS,
            seed: 42,
            sigma_ref: DEFAULT_SIGMA,
            gamma: DEFAULT_GAMMA,
            adversarial: false,
            lob_data_path: None,
            btc_target: 500.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: false,
            fixed_size: false,
            no_time_features: false,
        }
    }
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

pub(crate) fn parse_cli() -> EnvConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = EnvConfig::default();
    // Tracks whether --sigma-ref was passed explicitly, so the
    // historical/counterfactual calibrated default below doesn't clobber a
    // caller's deliberate override (e.g. a future λ-style sigma_ref sweep).
    let mut sigma_ref_explicit = false;

    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--rl-mode" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.mode = match v.to_lowercase().as_str() {
                        "historical" => RlMode::Historical,
                        "counterfactual" => RlMode::Counterfactual,
                        _ => RlMode::Synthetic,
                    };
                    i += 1;
                }
            }
            "--adversarial" => cfg.adversarial = true,
            "--total-inventory" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.total_inventory = v.parse::<f64>().unwrap_or(cfg.total_inventory);
                    i += 1;
                }
            }
            "--total-steps" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.total_steps = v.parse::<usize>().unwrap_or(cfg.total_steps);
                    i += 1;
                }
            }
            "--seed" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.seed = v.parse::<u64>().unwrap_or(cfg.seed);
                    i += 1;
                }
            }
            "--lob-data" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.lob_data_path = Some(v.clone());
                    i += 1;
                }
            }
            "--btc-target" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.btc_target = v.parse::<f64>().unwrap_or(cfg.btc_target);
                    i += 1;
                }
            }
            "--adv-btc" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.adv_btc = v.parse::<f64>().unwrap_or(0.0);
                    i += 1;
                }
            }
            "--agg-trades" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.agg_trades_path = Some(v.clone());
                    i += 1;
                }
            }
            "--sigma-ref" => {
                if let Some(v) = args.get(i + 1) {
                    cfg.sigma_ref = v.parse::<f64>().unwrap_or(cfg.sigma_ref);
                    sigma_ref_explicit = true;
                    i += 1;
                }
            }
            "--fixed-steps" => cfg.fixed_steps = true,
            "--fixed-size" => cfg.fixed_size = true,
            "--no-time-features" => cfg.no_time_features = true,
            _ => {}
        }
        i += 1;
    }

    // Adjust default step count by mode.
    if matches!(cfg.mode, RlMode::Historical | RlMode::Counterfactual)
        && cfg.total_steps == DEFAULT_TOTAL_STEPS
    {
        cfg.total_steps = HISTORICAL_DEFAULT_STEPS;
    }
    // Historical/counterfactual runs default to the calibrated sigma_ref
    // (see CALIBRATED_SIGMA_REF) instead of the synthetic-mode 0.20 — every
    // caller gets this automatically, no per-call-site flag needed, so
    // there is exactly one place this can silently regress. Explicit
    // --sigma-ref still wins, for tests/sweeps that need a different value.
    if matches!(cfg.mode, RlMode::Historical | RlMode::Counterfactual) && !sigma_ref_explicit {
        cfg.sigma_ref = CALIBRATED_SIGMA_REF;
    }
    cfg
}
