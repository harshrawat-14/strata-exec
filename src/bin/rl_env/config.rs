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
    0.000, 0.001, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150,
    0.200, 0.300, 0.400, 0.500, 0.700, 0.900, 1.000,
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

/// What a counterfactual fill's reward is charged against (doc 25).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RewardBaseline {
    /// The mid without our impact: execution and own-impact cost only;
    /// market moves stay out of the reward.
    Mid,
    /// The arrival price (full IS), minus TWAP's gain from market moves on
    /// the same price path, so the episode return is about IS minus TWAP's IS.
    Twap,
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
    /// Evaluation protocol v3 (`--episode-steps H`): every episode is exactly
    /// H steps and may start anywhere that still leaves H snapshots of data,
    /// chosen by the reset message's `start_frac` or drawn from the seed.
    /// `None` keeps the legacy protocol (horizon = day length or random,
    /// start = seed % (day/3)) byte-for-byte.
    pub episode_steps: Option<usize>,
    /// `--reward-baseline mid|twap` (counterfactual only; default mid).
    pub reward_baseline: RewardBaseline,
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
            episode_steps: None,
            reward_baseline: RewardBaseline::Mid,
        }
    }
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

/// Parse the process arguments, exiting with status 2 on invalid input.
pub(crate) fn parse_cli() -> EnvConfig {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("rl-env: {e}");
            std::process::exit(2);
        }
    }
}

fn flag_value<'a>(args: &'a [String], i: usize, flag: &str) -> Result<&'a str, String> {
    args.get(i + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_num<T: std::str::FromStr>(v: &str, flag: &str) -> Result<T, String> {
    v.parse::<T>()
        .map_err(|_| format!("invalid value for {flag}: {v:?}"))
}

/// Parse rl-env arguments (without the program name).
///
/// Strict on purpose: unknown flags, unknown modes, missing values and
/// unparseable numbers are errors. The previous parser ignored all four,
/// so e.g. Python's `--jump-diffusion` was silently a no-op and a mistyped
/// `--rl-mode` silently ran synthetic mode.
pub(crate) fn parse_args(args: &[String]) -> Result<EnvConfig, String> {
    let mut cfg = EnvConfig::default();
    // Tracks whether --sigma-ref was passed explicitly, so the
    // historical/counterfactual calibrated default below doesn't clobber a
    // caller's deliberate override (e.g. a future λ-style sigma_ref sweep).
    let mut sigma_ref_explicit = false;
    let mut total_steps_explicit = false;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--rl-mode" => {
                let v = flag_value(args, i, flag)?;
                cfg.mode = match v.to_lowercase().as_str() {
                    "synthetic" => RlMode::Synthetic,
                    "historical" => RlMode::Historical,
                    "counterfactual" => RlMode::Counterfactual,
                    _ => return Err(format!("unknown --rl-mode {v:?}")),
                };
                i += 1;
            }
            "--total-inventory" => {
                cfg.total_inventory = parse_num(flag_value(args, i, flag)?, flag)?;
                i += 1;
            }
            "--total-steps" => {
                cfg.total_steps = parse_num(flag_value(args, i, flag)?, flag)?;
                total_steps_explicit = true;
                i += 1;
            }
            "--reward-baseline" => {
                cfg.reward_baseline = match flag_value(args, i, flag)? {
                    "mid" => RewardBaseline::Mid,
                    "twap" => RewardBaseline::Twap,
                    other => return Err(format!("--reward-baseline must be mid or twap, got {other:?}")),
                };
                i += 1;
            }
            "--episode-steps" => {
                let h: usize = parse_num(flag_value(args, i, flag)?, flag)?;
                if h == 0 {
                    return Err("--episode-steps must be positive".to_string());
                }
                cfg.episode_steps = Some(h);
                i += 1;
            }
            "--seed" => {
                cfg.seed = parse_num(flag_value(args, i, flag)?, flag)?;
                i += 1;
            }
            "--lob-data" => {
                cfg.lob_data_path = Some(flag_value(args, i, flag)?.to_string());
                i += 1;
            }
            "--btc-target" => {
                cfg.btc_target = parse_num(flag_value(args, i, flag)?, flag)?;
                i += 1;
            }
            "--adv-btc" => {
                cfg.adv_btc = parse_num(flag_value(args, i, flag)?, flag)?;
                i += 1;
            }
            "--agg-trades" => {
                cfg.agg_trades_path = Some(flag_value(args, i, flag)?.to_string());
                i += 1;
            }
            "--sigma-ref" => {
                cfg.sigma_ref = parse_num(flag_value(args, i, flag)?, flag)?;
                sigma_ref_explicit = true;
                i += 1;
            }
            // Consumed by parse_batch_args; validated here so they are not
            // mistaken for unknown flags.
            "--batch-size" | "--socket-path" => {
                flag_value(args, i, flag)?;
                i += 1;
            }
            "--adversarial" => cfg.adversarial = true,
            "--fixed-steps" => cfg.fixed_steps = true,
            "--fixed-size" => cfg.fixed_size = true,
            "--no-time-features" => cfg.no_time_features = true,
            // Query modes handled in main().
            "--print-action-fractions" | "--print-env-constants" | "--print-calibration" => {}
            _ => return Err(format!("unknown argument {flag:?}")),
        }
        i += 1;
    }

    if let Some(h) = cfg.episode_steps {
        if total_steps_explicit {
            return Err("--episode-steps and --total-steps are mutually exclusive".to_string());
        }
        if matches!(cfg.mode, RlMode::Synthetic) {
            return Err(
                "--episode-steps applies to historical and counterfactual modes only".to_string(),
            );
        }
        cfg.total_steps = h;
        cfg.fixed_steps = true;
    }

    // The adversarial agents act on the historical/synthetic book only; the
    // counterfactual step has no adversary, so accepting the flag there would
    // silently run a non-adversarial episode under an adversarial label.
    if cfg.adversarial && matches!(cfg.mode, RlMode::Counterfactual) {
        return Err("--adversarial is not supported in counterfactual mode \
                    (adversarial agents exist only in historical and synthetic modes)"
            .to_string());
    }

    if cfg.reward_baseline == RewardBaseline::Twap && !matches!(cfg.mode, RlMode::Counterfactual) {
        return Err("--reward-baseline twap applies to counterfactual mode only".to_string());
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
    Ok(cfg)
}
