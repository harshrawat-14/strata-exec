// rl_env — Long-running RL environment server.
//
// Python controls this process via line-delimited JSON over stdin/stdout.
// One JSON object per line in both directions.
//
//   reset:  {"reset": true, "seed": 42}
//   action: {"action": <0..=19>}
//
// The response is a JSON object {state, reward, done, info}.

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Timing-instrumented allocator (separate from dhat-heap, which counts
/// allocations/bytes but not wall time spent inside them). Wraps the system
/// allocator, timing every alloc/dealloc/realloc call via `Instant::now()`
/// deltas summed into atomics, so a step-loop benchmark can report what
/// fraction of its own wall time is actually spent allocating -- the
/// follow-up measurement to the dhat-heap allocation-COUNT numbers in
/// docs/system_design/phase_s_baseline.md. `Instant::now()` itself has
/// measurable overhead (~20-40ns on macOS via mach_absolute_time), which is
/// a source of over-counting on very cheap allocations -- treat this as an
/// order-of-magnitude answer, documented as such wherever it's reported.
#[cfg(feature = "alloc-timing")]
mod alloc_timing {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub static ALLOC_NANOS: AtomicU64 = AtomicU64::new(0);
    pub static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        ALLOC_NANOS.store(0, Ordering::SeqCst);
        ALLOC_CALLS.store(0, Ordering::SeqCst);
    }

    pub fn snapshot() -> (u64, u64) {
        (ALLOC_NANOS.load(Ordering::SeqCst), ALLOC_CALLS.load(Ordering::SeqCst))
    }

    pub struct TimingAlloc;

    unsafe impl GlobalAlloc for TimingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let t0 = Instant::now();
            let ptr = System.alloc(layout);
            ALLOC_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            let t0 = Instant::now();
            System.dealloc(ptr, layout);
            ALLOC_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let t0 = Instant::now();
            let p = System.realloc(ptr, layout, new_size);
            ALLOC_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            p
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let t0 = Instant::now();
            let ptr = System.alloc_zeroed(layout);
            ALLOC_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ptr
        }
    }
}

#[cfg(feature = "alloc-timing")]
#[global_allocator]
static ALLOC_TIMING: alloc_timing::TimingAlloc = alloc_timing::TimingAlloc;

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io::{self, BufRead, Write};
use std::rc::Rc;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use strata_exec::market::agg_trades::AggTradesDay;
use strata_exec::market::adversarial::{FrontRunner, LiquidityWithdrawer};
use strata_exec::market::counterfactual_lob::{CounterfactualLob, CounterfactualLobConfig};
use strata_exec::market::garch::GarchSimulator;
use strata_exec::market::gbm::PriceSimulator;
use strata_exec::market::lob_generator::{generate_lob, LobGeneratorParams};
use strata_exec::market::lob_replay::{LobReplay, LobSnapshot};
use strata_exec::market::order_book::OrderBook;

/// Penalty reward applied when an episode is force-terminated by a data-ingestion
/// failure (missing/corrupt historical file, zero valid snapshots) rather than
/// crashing the whole training process. Deliberately an order of magnitude larger
/// in absolute value than a normal step reward (see the reward-scale diagnostic in
/// `RlEnv::reset`, ~1e-4 to 1e-2), so it is unambiguously distinguishable from
/// ordinary reward noise in training logs, without being so extreme it destabilizes
/// return statistics if it fires legitimately in production.
const DATA_INGESTION_FAILURE_PENALTY: f64 = -1.0;

/// Caches parsed per-date market data (`LobReplay` snapshots, `AggTradesDay`) across
/// repeated `reset()` calls within one `RlEnv`'s process lifetime.
///
/// Phase S baseline measurement: `LobReplay::from_csv` + `AggTradesDay::from_csv`
/// account for ~99% of a historical/counterfactual `reset()`'s allocations and ~100%
/// of its wall time (re-parsing a 1M+-row aggTrades CSV and a 28.8K-row bookDepth CSV
/// from disk), yet the same date is replayed across many episodes both in evaluation
/// (50 episodes/strategy/date) and training (repeated sampling from `--multi-date`).
/// Parsing once per date per process and sharing the immutable parsed data via `Arc`
/// (a cheap refcount bump per reset, not a copy) removes that redundant I/O/parsing
/// entirely after the first reset for a given date. Deliberately process-local, not
/// persisted across `rl-env` invocations: an updated CSV on disk is always picked up
/// by the next process start.
#[derive(Default)]
pub struct MarketDataCache {
    lob: HashMap<String, Arc<Vec<LobSnapshot>>>,
    agg: HashMap<String, Arc<AggTradesDay>>,
}

impl MarketDataCache {
    fn load_lob(&mut self, path: &str) -> Result<LobReplay, String> {
        if let Some(snapshots) = self.lob.get(path) {
            return Ok(LobReplay::from_arc(Arc::clone(snapshots)));
        }
        let replay = LobReplay::from_csv(path)
            .map_err(|e| format!("failed to load LOB data '{path}': {e}"))?;
        let snapshots = replay.snapshots_arc();
        self.lob.insert(path.to_string(), Arc::clone(&snapshots));
        Ok(LobReplay::from_arc(snapshots))
    }

    fn load_agg(&mut self, path: &str) -> Result<Arc<AggTradesDay>, String> {
        if let Some(agg) = self.agg.get(path) {
            return Ok(Arc::clone(agg));
        }
        let agg = Arc::new(AggTradesDay::from_csv(path)?);
        self.agg.insert(path.to_string(), Arc::clone(&agg));
        Ok(agg)
    }
}

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
const DEFAULT_GAMMA: f64 = 0.005;

/// Default annualised volatility (synthetic mode).
const DEFAULT_SIGMA: f64 = 0.20;

/// Number of initial books used to compute `depth_ref`.
const DEPTH_REF_SAMPLES: usize = 50;

/// Default step count — one full trading day at ~30s cadence.
const DEFAULT_TOTAL_STEPS: usize = 2880;
/// Historical mode default step count (one trading day ≈ 30s cadence).
const HISTORICAL_DEFAULT_STEPS: usize = 2880;

/// Synthetic LOB depth scaling so the book can absorb realistic order sizes.
/// `base_quantity = total_inventory / SYNTHETIC_DEPTH_SCALE` per level.
const SYNTHETIC_DEPTH_SCALE: f64 = 5.0;
/// Number of levels per side in the synthetic LOB (deeper than the default 10).
const SYNTHETIC_NUM_LEVELS: usize = 20;

// ── Per-episode randomization ranges (Changes 1 & 4) ───────────────────────
/// Minimum random execution horizon (steps). At 200, action 9 (5%) still leaves
/// most inventory if used naively — the agent has to escalate to larger actions.
const RANDOM_HORIZON_MIN: usize = 200;
/// Maximum random execution horizon (steps).
const RANDOM_HORIZON_MAX: usize = 1_440;
/// Minimum random order size as a fraction of the configured `total_inventory`.
const RANDOM_SIZE_MIN: f64 = 0.1;
/// Maximum random order size as a fraction of the configured `total_inventory`.
const RANDOM_SIZE_MAX: f64 = 1.0;

/// Build a synthetic LOB whose total depth is sized to the configured inventory.
fn synthetic_book(mid: f64, sigma: f64, total_inventory: f64) -> OrderBook {
    let params = LobGeneratorParams {
        num_levels: SYNTHETIC_NUM_LEVELS,
        base_quantity: (total_inventory / SYNTHETIC_DEPTH_SCALE).max(1_000.0),
        ..LobGeneratorParams::default()
    };
    generate_lob(mid, sigma, &params)
}

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
// JSON wire types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct InMsg {
    #[serde(default)]
    reset: bool,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    action: Option<usize>,
}

#[derive(Serialize)]
struct OutMsg {
    state: Vec<f64>,
    reward: f64,
    done: bool,
    info: Value,
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

pub struct StepResult {
    pub state: Vec<f64>,
    pub reward: f64,
    pub done: bool,
    pub info: Value,
}

pub struct RlEnv {
    cfg: EnvConfig,

    // ── Live state after `reset()` ──
    sim: Option<Box<dyn PriceSimulator>>,
    lob_replay: Option<LobReplay>,
    current_book: Option<OrderBook>,
    current_mid: f64,

    arrival_price: f64,
    spread_ref: f64,
    depth_ref: f64,
    /// Historical-mode price rescale factor (100.0 / arrival raw-BTC mid), fixed once
    /// at `reset_historical` and reused for every snapshot in the episode. Must NOT be
    /// recomputed per-step -- doing so renormalises every book back to mid≈100 and
    /// erases real intra-episode price drift (see `advance_market`).
    price_factor: f64,

    step_idx: usize,
    remaining: f64,

    /// Last `current_mid` values, oldest first, capacity = 11 (so we can look back 10 steps).
    price_history: VecDeque<f64>,
    /// Signed slippage on the most recent fill: (exec_price − mid). Negative = adverse for a sell.
    last_slippage: f64,
    /// Current annualised volatility estimate (synthetic: from simulator, historical: rolling).
    current_sigma: f64,

    /// Total notional received from fills (sum of fill_qty × fill_price).
    total_notional_received: f64,
    /// Total quantity filled across the episode.
    total_filled: f64,

    /// Rolling buffer of recent log-returns for the historical-mode realised-vol estimate.
    return_buffer: VecDeque<f64>,

    /// Adversarial state (Component 2). `None` when `--adversarial` is not set.
    adversarial: Option<AdversarialState>,

    /// Counterfactual LOB (real Binance snapshots + self-impact model).
    cf_lob: Option<CounterfactualLob>,

    /// Loaded aggTrades for OFI computation (historical/counterfactual modes).
    agg_trades: Option<Arc<AggTradesDay>>,
    /// Precomputed OFI per snapshot, indexed across the full replay (not offset-adjusted).
    ofi_series: Vec<f64>,
    /// Snapshot index of episode start; ofi_series[ofi_start + step_idx] = OFI at step.
    ofi_start: usize,

    /// Parsed-per-date cache shared across resets within this process (see
    /// `MarketDataCache`). `Rc<RefCell<_>>` (not a plain field) so a batched env's
    /// N instances can share ONE cache instead of each parsing independently --
    /// single-threaded (one process, sequential stepping), so RefCell is sufficient.
    market_cache: Rc<RefCell<MarketDataCache>>,

    /// Snapshot of `cfg.total_steps` at construction time. The per-episode
    /// `cfg.total_steps` is rebuilt from this each reset (random or fixed).
    base_total_steps: usize,
    /// Snapshot of `cfg.total_inventory` at construction time. The per-episode
    /// `cfg.total_inventory` is rebuilt from this each reset (random or fixed).
    base_total_inventory: f64,
    /// Per-episode order size as a fraction of `base_total_inventory` (1.0 = full).
    /// Exposed in the reset info JSON for diagnostics.
    episode_size_fraction: f64,
}

/// Persistent adversarial-agent state attached to the environment.
/// The agents' behaviour is stateful; the underlying book lives in `RlEnv::current_book`
/// and is mutated by these agents through the public bid-ladder accessors.
struct AdversarialState {
    withdrawer: LiquidityWithdrawer,
    front_runner: FrontRunner,
    last_front_runner_qty: f64,
    last_depth_withdrawn: f64,
    last_pre_fill_depth: f64,
}

/// When a fresh book is sourced (synthetic regeneration or new historical snapshot),
/// carry over the LiquidityWithdrawer's outstanding per-level deficit so the
/// adversary's effect persists across step boundaries.
fn apply_adversarial_deficit(book: &mut OrderBook, adv: Option<&AdversarialState>) {
    let Some(adv) = adv else {
        return;
    };
    let deficits = &adv.withdrawer.withdrawn_depth_per_level;
    if deficits.iter().all(|d| *d <= 0.0) {
        return;
    }
    let levels = book.bid_levels_mut();
    for (i, lvl) in levels.iter_mut().enumerate() {
        if i < deficits.len() && deficits[i] > 0.0 {
            lvl.quantity = (lvl.quantity - deficits[i]).max(0.0);
        }
    }
}

impl RlEnv {
    pub fn new(cfg: EnvConfig) -> Self {
        Self::new_with_shared_cache(cfg, Rc::new(RefCell::new(MarketDataCache::default())))
    }

    /// Construct an `RlEnv` sharing an existing `MarketDataCache` with other
    /// instances (used by `BatchedRlEnv` so N instances parse each date's CSV
    /// once between them, not N times).
    pub fn new_with_shared_cache(cfg: EnvConfig, market_cache: Rc<RefCell<MarketDataCache>>) -> Self {
        let adversarial = if cfg.adversarial {
            Some(AdversarialState {
                withdrawer: LiquidityWithdrawer::new(),
                front_runner: FrontRunner::new(),
                last_front_runner_qty: 0.0,
                last_depth_withdrawn: 0.0,
                last_pre_fill_depth: 0.0,
            })
        } else {
            None
        };
        let base_total_steps = cfg.total_steps;
        let base_total_inventory = cfg.total_inventory;
        Self {
            cfg,
            sim: None,
            lob_replay: None,
            current_book: None,
            current_mid: 0.0,
            arrival_price: 0.0,
            spread_ref: 1.0,
            depth_ref: 1.0,
            price_factor: 1.0,
            step_idx: 0,
            remaining: 0.0,
            price_history: VecDeque::with_capacity(16),
            last_slippage: 0.0,
            current_sigma: DEFAULT_SIGMA,
            total_notional_received: 0.0,
            total_filled: 0.0,
            return_buffer: VecDeque::with_capacity(64),
            adversarial,
            cf_lob: None,
            agg_trades: None,
            ofi_series: Vec::new(),
            ofi_start: 0,
            market_cache,
            base_total_steps,
            base_total_inventory,
            episode_size_fraction: 1.0,
        }
    }

    /// Reset the environment for a new episode.
    pub fn reset(&mut self, seed_override: Option<u64>) -> StepResult {
        let seed = seed_override.unwrap_or(self.cfg.seed);

        // Restore per-episode-overridable values from the construction-time snapshot
        // so randomisation does not drift across repeated resets.
        self.cfg.total_steps = self.base_total_steps;
        self.cfg.total_inventory = self.base_total_inventory;
        self.episode_size_fraction = 1.0;

        // Replay modes: randomise execution horizon and order size per episode.
        // Eval can opt out via --fixed-steps and --fixed-size.
        if matches!(self.cfg.mode, RlMode::Historical | RlMode::Counterfactual) {
            let mut rng = StdRng::seed_from_u64(seed);
            if !self.cfg.fixed_steps {
                self.cfg.total_steps =
                    rng.gen_range(RANDOM_HORIZON_MIN..=RANDOM_HORIZON_MAX);
            }
            if !self.cfg.fixed_size {
                let size_fraction: f64 = rng.gen_range(RANDOM_SIZE_MIN..=RANDOM_SIZE_MAX);
                self.episode_size_fraction = size_fraction;
                self.cfg.total_inventory =
                    (self.base_total_inventory * size_fraction).max(1.0);
            }
        }

        self.step_idx = 0;
        self.remaining = self.cfg.total_inventory;
        self.price_history.clear();
        self.return_buffer.clear();
        self.last_slippage = 0.0;
        self.total_notional_received = 0.0;
        self.total_filled = 0.0;
        self.current_sigma = self.cfg.sigma_ref;

        let ingestion_result = match self.cfg.mode {
            RlMode::Synthetic => {
                self.reset_synthetic(seed);
                Ok(())
            }
            RlMode::Historical => self.reset_historical(seed),
            RlMode::Counterfactual => self.reset_counterfactual(seed),
        };

        if let Err(e) = ingestion_result {
            eprintln!("Data ingestion error at reset (episode force-terminated): {e}");
            let n_dims = if self.cfg.no_time_features { 12 } else { 16 };
            return StepResult {
                state: vec![0.0; n_dims],
                reward: DATA_INGESTION_FAILURE_PENALTY,
                done: true,
                info: json!({
                    "error": e,
                    "terminated_reason": "data_ingestion_failure",
                }),
            };
        }

        // Reward-scale diagnostic (printed to stderr at each reset).
        // wait_at_full = -(var_penalty + inv_pressure) × 0.0001; at q=1, sigma=sigma_ref.
        let wait_at_full = -(0.01_f64 + 0.0002_f64) * 0.0001;
        eprintln!("Reward scale (new): wait_at_full_inv={:.2e}", wait_at_full);

        let state = self.build_state();
        let (vol_regime, liq_regime) = self.regime_dims();
        let mut info = json!({
            "step": 0_usize,
            "remaining": self.remaining,
            "arrival_price": self.arrival_price,
            "action_9_fraction": ACTION_FRACTIONS[9],
            "episode_horizon": self.cfg.total_steps,
            "total_inventory": self.cfg.total_inventory,
            "size_fraction": self.episode_size_fraction,
            "n_state_dims": state.len(),
            "vol_regime": vol_regime,
            "liq_regime": liq_regime,
            "time_features_active": !self.cfg.no_time_features,
        });
        if let Some(cf) = &self.cf_lob {
            let btc_scale = self.cfg.btc_target / self.cfg.total_inventory.max(1.0);
            let adv_btc = if self.cfg.adv_btc > 0.0 { self.cfg.adv_btc } else { 240_000.0 };
            if let Some(obj) = info.as_object_mut() {
                obj.insert("btc_scale".to_string(), json!(btc_scale));
                obj.insert("adv_btc".to_string(), json!(adv_btc));
                obj.insert("depth_ref".to_string(), json!(cf.depth_ref));
            }
        }
        StepResult { state, reward: 0.0, done: false, info }
    }

    fn reset_synthetic(&mut self, seed: u64) {
        let total_steps = self.cfg.total_steps;
        let dt = 1.0 / total_steps as f64;
        // Initial price=100 — calibration-independent.
        let initial_price = 100.0;
        let sigma = self.cfg.sigma_ref;
        let sim = GarchSimulator::new(
            initial_price,
            0.05,
            0.000_002,
            0.08,
            0.90,
            sigma * sigma,
            dt,
            seed,
        );

        let book = synthetic_book(initial_price, sigma, self.cfg.total_inventory);
        self.arrival_price = book.mid_price().unwrap_or(initial_price);
        self.current_mid = self.arrival_price;
        self.spread_ref = book.half_spread().unwrap_or(0.01).max(1e-9);

        // depth_ref: mean bid_depth_1pct over the next 50 synthetic books (non-destructive).
        let mut depth_sum = 0.0_f64;
        let mut count = 0usize;
        // Project forward without mutating sim: just generate books at slightly varying prices.
        // For a stable reference, regenerate from initial_price with the same vol — they are
        // identical, so 50 samples is just `1 × initial sample`. Vary `mid` slightly to broaden.
        for k in 0..DEPTH_REF_SAMPLES {
            // small spread in mid to give a representative sample
            let mid = initial_price * (1.0 + 0.001 * (k as f64 - 25.0) / 25.0);
            let b = synthetic_book(mid, sigma, self.cfg.total_inventory);
            let m = b.mid_price().unwrap_or(mid);
            depth_sum += b.bid_depth_up_to(m * 0.99);
            count += 1;
        }
        self.depth_ref = (depth_sum / count.max(1) as f64).max(1.0);

        self.current_book = Some(book);
        self.price_history.push_back(self.current_mid);
        self.sim = Some(Box::new(sim));
        self.lob_replay = None;
    }

    fn reset_historical(&mut self, seed: u64) -> Result<(), String> {
        let path = self
            .cfg
            .lob_data_path
            .clone()
            .ok_or_else(|| "historical mode requires --lob-data".to_string())?;
        let mut replay = self.market_cache.borrow_mut().load_lob(&path)?;
        let n = replay.len();
        // Auto-align total_steps to replay length.
        if self.cfg.total_steps == DEFAULT_TOTAL_STEPS
            || self.cfg.total_steps == HISTORICAL_DEFAULT_STEPS
        {
            self.cfg.total_steps = n;
        }

        // Rescaling factors: same logic as counterfactual mode so fill prices and
        // state observations are in simulation scale (~100) rather than raw BTC (~42000).
        let btc_scale = (self.cfg.btc_target / self.cfg.total_inventory).max(1e-12);
        let qty_factor = 1.0 / btc_scale;

        // Compute depth_ref from the first DEPTH_REF_SAMPLES snapshots (cheap, non-destructive).
        let timestamps = replay.snapshot_timestamps();
        let _ = timestamps; // currently unused; reserved for future agg-trades alignment
        let mut depth_sum = 0.0_f64;
        let mut count = 0usize;
        let mut first_book: Option<OrderBook> = None;
        for _ in 0..DEPTH_REF_SAMPLES {
            match replay.next_snapshot() {
                Some(b) => {
                    if first_book.is_none() {
                        first_book = Some(b.clone());
                    }
                    if let Some(m) = b.mid_price() {
                        depth_sum += b.bid_depth_up_to(m * 0.99);
                        count += 1;
                    }
                }
                None => break,
            }
        }
        replay.reset();
        // Scale depth_ref to simulation units (raw BTC depth × qty_factor).
        self.depth_ref = ((depth_sum / count.max(1) as f64) * qty_factor).max(1.0);

        // Random start within the first third of the day.
        // Different seeds → different arrival prices → non-zero IS variance across episodes.
        let max_start = (replay.len() / 3).max(1);
        let start_offset = (seed as usize) % max_start;
        replay.seek(start_offset);

        // Pull the snapshot at start_offset as the initial book.
        let first = replay.next_snapshot().ok_or_else(|| {
            format!(
                "LOB replay for '{path}' yielded zero snapshots at start_offset={start_offset} \
                 (possible data gap or truncated file)"
            )
        })?;
        let raw_btc_mid = first.mid_price().unwrap_or(100.0);
        let price_factor = 100.0 / raw_btc_mid;
        // Fixed once for the whole episode -- advance_market() must reuse this exact
        // factor for every later snapshot so real price drift survives the rescale
        // instead of being renormalised back to mid≈100 every step.
        self.price_factor = price_factor;
        let scaled_first = first.rescale_full(price_factor, qty_factor);

        self.spread_ref = scaled_first.half_spread().unwrap_or(0.01).max(1e-9);
        // current_mid stays at the raw BTC price so log-returns between consecutive
        // raw-BTC snapshots are computed correctly for the volatility estimator.
        self.current_mid = raw_btc_mid;
        // Store the rescaled book so fills and state observations use simulation scale.
        self.current_book = Some(scaled_first);
        self.price_history.push_back(self.current_mid);
        // FIX A: IS reward formula uses arrival_price as the normalisation denominator.
        // Set to simulation scale (100.0) so reward magnitudes are mode-independent.
        self.arrival_price = 100.0;

        // Precompute OFI series from aggTrades before replay is moved into self.
        // AggTrades load failure is intentionally non-fatal: OFI is an optional
        // signal (state dim s5 falls back to 0.0), unlike the LOB replay itself.
        if let Some(agg_path) = self.cfg.agg_trades_path.clone() {
            let timestamps = replay.snapshot_timestamps();
            match self.market_cache.borrow_mut().load_agg(&agg_path) {
                Ok(agg) => {
                    self.ofi_series = agg.build_ofi_series(&timestamps, 30_000);
                    self.ofi_start = start_offset;
                    self.agg_trades = Some(agg);
                }
                Err(e) => {
                    eprintln!("AggTrades load error (historical): {e}");
                    self.agg_trades = None;
                    self.ofi_series.clear();
                    self.ofi_start = 0;
                }
            }
        } else {
            self.agg_trades = None;
            self.ofi_series.clear();
            self.ofi_start = 0;
        }

        self.lob_replay = Some(replay);
        self.sim = None;
        Ok(())
    }

    fn reset_counterfactual(&mut self, seed: u64) -> Result<(), String> {
        let path = self
            .cfg
            .lob_data_path
            .clone()
            .ok_or_else(|| "counterfactual mode requires --lob-data".to_string())?;
        let mut replay = self.market_cache.borrow_mut().load_lob(&path)?;
        let n = replay.len();
        if self.cfg.total_steps == DEFAULT_TOTAL_STEPS
            || self.cfg.total_steps == HISTORICAL_DEFAULT_STEPS
        {
            self.cfg.total_steps = n;
        }

        let adv_btc = if self.cfg.adv_btc > 0.0 {
            self.cfg.adv_btc
        } else {
            240_000.0
        };
        let btc_scale = self.cfg.btc_target / self.cfg.total_inventory;

        // Print startup diagnostics (once per reset — only informative on first call).
        let cf_config = CounterfactualLobConfig {
            btc_scale,
            adv_btc,
            ..CounterfactualLobConfig::default()
        };

        eprintln!("Mode: counterfactual LOB");
        eprintln!("  BTC target:    {} BTC", self.cfg.btc_target);
        eprintln!("  Scale factor:  {} BTC per unit", btc_scale);
        eprintln!("  ADV:           {} BTC", adv_btc);
        eprintln!(
            "  Impact at 1%:  {:.2}% depth reduction",
            (0.01_f64).sqrt() * 0.1 * 100.0
        );
        eprintln!("  Rho (recovery):{}", cf_config.rho);
        eprintln!(
            "  Half-life:     {:.0} steps ({:.0} minutes)",
            (2_f64.ln() / cf_config.rho / cf_config.dt),
            (2_f64.ln() / cf_config.rho / cf_config.dt) * 0.5,
        );

        // Build depth_ref from first few snapshots.
        let mut depth_sum = 0.0_f64;
        let mut count = 0usize;
        let mut first_book: Option<OrderBook> = None;
        for _ in 0..DEPTH_REF_SAMPLES {
            match replay.next_snapshot() {
                Some(b) => {
                    if first_book.is_none() {
                        first_book = Some(b.clone());
                    }
                    if let Some(m) = b.mid_price() {
                        depth_sum += b.bid_depth_up_to(m * 0.99);
                        count += 1;
                    }
                }
                None => break,
            }
        }
        replay.reset();
        self.depth_ref = (depth_sum / count.max(1) as f64).max(1.0);

        // Random start within the first third of the day.
        let max_start = (replay.len() / 3).max(1);
        let start_offset = (seed as usize) % max_start;
        replay.seek(start_offset);

        let first = replay.next_snapshot().ok_or_else(|| {
            format!(
                "LOB replay for '{path}' yielded zero snapshots at start_offset={start_offset} \
                 (possible data gap or truncated file)"
            )
        })?;
        self.spread_ref = first.half_spread().unwrap_or(0.01).max(1e-9);
        self.current_book = Some(first);
        // FIX A: cf_lob rescales all BTC prices to simulation scale (~100).
        // Both arrival_price and current_mid must match that scale, otherwise the IS
        // reward formula gets a ~42000 denominator while fills are priced at ~100.
        self.arrival_price = 100.0;
        self.current_mid = 100.0;
        self.price_history.push_back(self.current_mid);

        // Precompute OFI series from aggTrades (replay cursor is beyond start_offset here,
        // but snapshot_timestamps() returns all timestamps regardless of cursor position).
        if let Some(agg_path) = self.cfg.agg_trades_path.clone() {
            let timestamps = replay.snapshot_timestamps();
            match self.market_cache.borrow_mut().load_agg(&agg_path) {
                Ok(agg) => {
                    self.ofi_series = agg.build_ofi_series(&timestamps, 30_000);
                    self.ofi_start = start_offset;
                    self.agg_trades = Some(agg);
                }
                Err(e) => {
                    eprintln!("AggTrades load error (counterfactual): {e}");
                    self.agg_trades = None;
                    self.ofi_series.clear();
                    self.ofi_start = 0;
                }
            }
        } else {
            self.agg_trades = None;
            self.ofi_series.clear();
            self.ofi_start = 0;
        }

        self.lob_replay = None;
        self.sim = None;

        // Build CounterfactualLob on the first episode; reuse and explicitly reset on
        // subsequent resets so accumulated impact from the previous episode never carries over.
        if self.cf_lob.is_none() {
            let cf_replay = self.market_cache.borrow_mut().load_lob(&path)?;
            self.cf_lob = Some(CounterfactualLob::new(cf_replay, cf_config));
        }
        let cf = self.cf_lob.as_mut().unwrap();
        cf.reset();
        // Variable per-episode inventory changes btc_scale; push the fresh value
        // so the impact model converts trade-units → BTC with the right ratio.
        cf.set_btc_scale(btc_scale);
        cf.seek(start_offset);
        Ok(())
    }

    /// Execute one action and advance the environment by one step.
    pub fn step(&mut self, action: usize) -> StepResult {
        if matches!(self.cfg.mode, RlMode::Counterfactual) {
            return self.step_counterfactual(action);
        }
        let action = action.min(ACTION_FRACTIONS.len() - 1);

        let book_mid = self
            .current_book
            .as_ref()
            .and_then(|b| b.mid_price())
            .unwrap_or(self.current_mid);

        // Quantity to trade this step.
        let intended = ACTION_FRACTIONS[action] * self.remaining;

        let mut fill_qty = 0.0;
        let mut fill_price = book_mid;
        let mut slippage_signed = 0.0; // (exec − mid); negative is adverse for a sell
        let mut step_fr_qty = 0.0_f64;
        let mut step_withdrawn = 0.0_f64;
        let mut step_pre_fill_depth = 0.0_f64;

        if intended > 1e-9 {
            if let Some(book) = self.current_book.as_mut() {
                // Adversarial path: front-runner steals depth, then we fill, then withdrawer pulls liquidity.
                if let Some(adv) = self.adversarial.as_mut() {
                    step_pre_fill_depth = book.total_bid_depth();
                    step_fr_qty = adv.front_runner.pre_fill(book, intended);
                    let pre_our_depth = book.total_bid_depth();
                    let fr_result = book.fill_market_sell(intended);
                    step_withdrawn =
                        adv.withdrawer.post_fill(book, fr_result.filled_qty, pre_our_depth);
                    adv.withdrawer.recover_step(book);
                    adv.front_runner.update_volume(fr_result.filled_qty);

                    adv.last_front_runner_qty = step_fr_qty;
                    adv.last_depth_withdrawn = step_withdrawn;
                    adv.last_pre_fill_depth = step_pre_fill_depth;

                    if fr_result.filled_qty > 1e-12 {
                        fill_qty = fr_result.filled_qty;
                        fill_price = fr_result.avg_fill_price;
                        slippage_signed = fr_result.avg_fill_price - book_mid;
                    }
                } else {
                    let fr = book.fill_market_sell(intended);
                    if fr.filled_qty > 1e-12 {
                        fill_qty = fr.filled_qty;
                        fill_price = fr.avg_fill_price;
                        slippage_signed = fr.avg_fill_price - book_mid;
                    }
                }
            }
        }

        // Update inventory.
        self.remaining = (self.remaining - fill_qty).max(0.0);
        self.total_filled += fill_qty;
        self.total_notional_received += fill_qty * fill_price;
        self.last_slippage = slippage_signed;

        // Reward components.
        let total = self.cfg.total_inventory.max(1.0);
        let q_frac = self.remaining / total;
        let sigma_ref = self.cfg.sigma_ref.max(1e-9);

        // Keep fill_cost/perm_cost as fractions for is_contribution in info.
        let fill_cost = if fill_qty > 0.0 && book_mid > 1e-12 {
            (book_mid - fill_price) / book_mid // positive = adverse for a seller
        } else {
            0.0
        };
        let perm_cost = self.cfg.gamma * (fill_qty / total);

        // Dimensionless IS-normalised reward.
        // Fill step: signed slippage × fill fraction (negative when sell < mid).
        // Wait step: small penalty for remaining exposure (keeps wait rewards tiny).
        let var_penalty = 0.01 * (self.current_sigma / sigma_ref).powi(2) * q_frac.powi(2);
        let inv_pressure = 0.0002 * q_frac.powi(2);

        let reward = if fill_qty > 1e-9 {
            (fill_price - book_mid) * fill_qty / (self.arrival_price.max(1e-12) * total)
        } else {
            -(var_penalty + inv_pressure) * 0.0001
        };

        // Advance time.
        self.step_idx += 1;
        self.advance_market();

        // Termination check.
        let inv_done = self.remaining <= 0.001 * self.cfg.total_inventory;
        let time_done = self.step_idx >= self.cfg.total_steps;
        let mut done = inv_done || time_done;
        let mut forced_liq_qty = 0.0;
        let mut total_reward_adj = 0.0;

        if time_done && self.remaining > 0.001 * self.cfg.total_inventory {
            // Forced liquidation: walk the book for the full remaining size via the same
            // fill_market_sell used for voluntary trades, so a last-minute dump is priced
            // by genuinely consumed depth/impact instead of a flat, size-independent penalty.
            // Invariant: never-trading must not be reward-superior to competent execution --
            // the old flat `gamma * (forced_liq_qty/total)`-only penalty was cheap enough
            // that the agent learned to always wait and eat it instead of trading normally.
            forced_liq_qty = self.remaining;
            let book_mid_at_end = self
                .current_book
                .as_ref()
                .and_then(|b| b.mid_price())
                .unwrap_or(self.arrival_price);

            let liquidated_notional = if let Some(book) = self.current_book.as_mut() {
                let fr = book.fill_market_sell(forced_liq_qty);
                let mut notional = fr.filled_qty * fr.avg_fill_price;
                if fr.residual_qty > 1e-9 {
                    // Book depth exhausted before absorbing the full size -- price the
                    // unfillable residual punitively (worse than the worst level actually
                    // consumed) rather than defaulting it to a free/mid-priced fill.
                    let worst_price = if fr.filled_qty > 1e-12 {
                        fr.avg_fill_price
                    } else {
                        book_mid_at_end
                    };
                    let punitive_price = (worst_price - self.spread_ref.max(1e-6) * 5.0).max(1e-6);
                    notional += fr.residual_qty * punitive_price;
                }
                notional
            } else {
                forced_liq_qty * book_mid_at_end
            };
            let liquidated_at = if forced_liq_qty > 1e-12 {
                liquidated_notional / forced_liq_qty
            } else {
                book_mid_at_end
            };

            self.total_filled += forced_liq_qty;
            self.total_notional_received += liquidated_notional;

            // Same slippage-based reward formula as a normal fill, plus the existing
            // permanent-impact term scaled to the full dumped quantity.
            let slippage_cost = (book_mid_at_end - liquidated_at) * forced_liq_qty
                / (self.arrival_price.max(1e-12) * total);
            let forced_perm = self.cfg.gamma * (forced_liq_qty / total);
            total_reward_adj -= slippage_cost + forced_perm;
            self.remaining = 0.0;
            done = true;
        }

        let mut info = json!({
            "step": self.step_idx,
            "remaining": self.remaining,
            "fill_qty": fill_qty,
            "fill_price": fill_price,
            "book_mid": book_mid,
            "is_contribution": -(fill_cost + perm_cost),
            "regime": self.regime_label(),
        });

        if self.adversarial.is_some() {
            let withdrawn_pct = if step_pre_fill_depth > 0.0 {
                (step_withdrawn / step_pre_fill_depth).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if let Some(obj) = info.as_object_mut() {
                obj.insert("front_runner_qty".to_string(), json!(step_fr_qty));
                obj.insert("depth_withdrawn_pct".to_string(), json!(withdrawn_pct));
            }
        }

        if done {
            // Total implementation shortfall at end of episode.
            let ideal = self.arrival_price * self.cfg.total_inventory;
            let received = self.total_notional_received;
            let is_dollar = ideal - received;
            let is_pct = if ideal.abs() > 1e-12 { is_dollar / ideal * 100.0 } else { 0.0 };
            if let Some(obj) = info.as_object_mut() {
                obj.insert("total_IS_pct".to_string(), json!(is_pct));
                obj.insert("total_is_dollar".to_string(), json!(is_dollar));
                if forced_liq_qty > 0.0 {
                    obj.insert("forced_liquidation_qty".to_string(), json!(forced_liq_qty));
                }
            }
        }

        let state = self.build_state();
        StepResult {
            state,
            reward: reward + total_reward_adj,
            done,
            info,
        }
    }

    /// Execute one step in counterfactual mode.
    fn step_counterfactual(&mut self, action: usize) -> StepResult {
        let action = action.min(ACTION_FRACTIONS.len() - 1);

        let intended = ACTION_FRACTIONS[action] * self.remaining;

        // Get the adjusted book (real Binance − accumulated impact).
        let adjusted_book = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.current_adjusted_book())
            .unwrap_or_else(|| self.current_book.clone().unwrap_or(OrderBook::new(vec![], vec![])));

        let book_mid = adjusted_book.mid_price().unwrap_or(self.current_mid);
        let real_depth = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.real_total_bid_depth())
            .unwrap_or(0.0);

        let mut fill_qty = 0.0;
        let mut fill_price = book_mid;
        let mut slippage_signed = 0.0;

        if intended > 1e-9 {
            let mut book = adjusted_book.clone();
            let fr = book.fill_market_sell(intended);
            if fr.filled_qty > 1e-12 {
                fill_qty = fr.filled_qty;
                fill_price = fr.avg_fill_price;
                slippage_signed = fr.avg_fill_price - book_mid;
            }
        }

        // Record impact and advance cf_lob to next snapshot.
        if let Some(cf) = self.cf_lob.as_mut() {
            cf.record_trade_and_advance(intended.max(0.0));
        }

        // Update inventory.
        self.remaining = (self.remaining - fill_qty).max(0.0);
        self.total_filled += fill_qty;
        self.total_notional_received += fill_qty * fill_price;
        self.last_slippage = slippage_signed;

        // Sync current_book/mid from the updated cf_lob.
        if let Some(cf) = &self.cf_lob {
            let new_book = cf.current_adjusted_book();
            let new_mid = new_book.mid_price().unwrap_or(self.current_mid);
            let prev_mid = self.current_mid.max(1e-12);
            let log_ret = (new_mid / prev_mid).ln();
            self.push_return(log_ret);
            self.current_mid = new_mid;
            self.current_sigma = self.realised_vol();
            self.current_book = Some(new_book);
            self.price_history.push_back(new_mid);
            while self.price_history.len() > 16 {
                self.price_history.pop_front();
            }
        }

        self.step_idx += 1;

        // Reward components (same normalised formula as standard mode).
        let total = self.cfg.total_inventory.max(1.0);
        let q_frac = self.remaining / total;
        let sigma_ref = self.cfg.sigma_ref.max(1e-9);
        let fill_cost = if fill_qty > 0.0 && book_mid > 1e-12 {
            (book_mid - fill_price) / book_mid
        } else {
            0.0
        };
        let perm_cost = self.cfg.gamma * (fill_qty / total);
        let var_penalty = 0.01 * (self.current_sigma / sigma_ref).powi(2) * q_frac.powi(2);
        let inv_pressure = 0.0002 * q_frac.powi(2);

        let reward = if fill_qty > 1e-9 {
            (fill_price - book_mid) * fill_qty / (self.arrival_price.max(1e-12) * total)
        } else {
            -(var_penalty + inv_pressure) * 0.0001
        };

        // Termination.
        let inv_done = self.remaining <= 0.001 * self.cfg.total_inventory;
        let time_done = self.step_idx >= self.cfg.total_steps;
        let cf_exhausted = self.cf_lob.as_ref().map(|cf| cf.is_exhausted()).unwrap_or(false);
        let mut done = inv_done || time_done || cf_exhausted;
        let mut total_reward_adj = 0.0;
        let mut forced_liq_qty = 0.0;

        if (time_done || cf_exhausted) && self.remaining > 0.001 * self.cfg.total_inventory {
            // Same fix as historical-mode step(): route the forced dump through the real
            // adjusted book instead of pricing it at a flat, size-independent penalty.
            forced_liq_qty = self.remaining;
            let mid_at_end = self.current_mid.max(1e-12);

            let liquidated_notional = if let Some(cf) = self.cf_lob.as_ref() {
                let mut book = cf.current_adjusted_book();
                let fr = book.fill_market_sell(forced_liq_qty);
                let mut notional = fr.filled_qty * fr.avg_fill_price;
                if fr.residual_qty > 1e-9 {
                    let worst_price = if fr.filled_qty > 1e-12 {
                        fr.avg_fill_price
                    } else {
                        mid_at_end
                    };
                    let punitive_price = (worst_price - self.spread_ref.max(1e-6) * 5.0).max(1e-6);
                    notional += fr.residual_qty * punitive_price;
                }
                notional
            } else {
                forced_liq_qty * mid_at_end
            };
            let liquidated_at = if forced_liq_qty > 1e-12 {
                liquidated_notional / forced_liq_qty
            } else {
                mid_at_end
            };

            let forced_perm = self.cfg.gamma * (forced_liq_qty / total);
            self.total_filled += forced_liq_qty;
            self.total_notional_received += liquidated_notional;
            let slippage_cost =
                (mid_at_end - liquidated_at) * forced_liq_qty / (self.arrival_price.max(1e-12) * total);
            total_reward_adj -= slippage_cost + forced_perm;
            self.remaining = 0.0;
            done = true;
        }

        let depth_reduction_pct = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.total_residual_impact() * 100.0)
            .unwrap_or(0.0);
        let permanent_impact_pct = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.permanent_impact() * 100.0)
            .unwrap_or(0.0);
        let adjusted_depth = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.current_adjusted_book().total_bid_depth())
            .unwrap_or(0.0);

        let mut info = json!({
            "step": self.step_idx,
            "remaining": self.remaining,
            "fill_qty": fill_qty,
            "fill_price": fill_price,
            "book_mid": book_mid,
            "btc_scale": self.cfg.btc_target / total,
            "is_contribution": -(fill_cost + perm_cost),
            "regime": self.regime_label(),
            "depth_reduction_pct": depth_reduction_pct,
            "permanent_impact_pct": permanent_impact_pct,
            "adjusted_book_depth": adjusted_depth,
            "real_book_depth": real_depth,
        });

        if done {
            let ideal = self.arrival_price * self.cfg.total_inventory;
            let received = self.total_notional_received;
            let is_pct = if ideal.abs() > 1e-12 {
                (ideal - received) / ideal * 100.0
            } else {
                0.0
            };
            if let Some(obj) = info.as_object_mut() {
                obj.insert("total_IS_pct".to_string(), json!(is_pct));
                if forced_liq_qty > 0.0 {
                    obj.insert("forced_liquidation_qty".to_string(), json!(forced_liq_qty));
                }
            }
        }

        let state = self.build_state();
        StepResult {
            state,
            reward: reward + total_reward_adj,
            done,
            info,
        }
    }

    /// Step the price simulator (synthetic) or pull the next snapshot (historical).
    /// Counterfactual mode advances through `cf_lob` inside `step_counterfactual`.
    fn advance_market(&mut self) {
        match self.cfg.mode {
            RlMode::Counterfactual => return,
            RlMode::Synthetic => {
                let stepped: Option<(f64, f64)> = self.sim.as_mut().map(|sim| {
                    let new_price = sim.step();
                    (new_price, sim.volatility())
                });
                if let Some((new_price, new_sigma)) = stepped {
                    let prev_mid = self.current_mid.max(1e-12);
                    let log_ret = (new_price / prev_mid).ln();
                    self.push_return(log_ret);
                    self.current_mid = new_price;
                    self.current_sigma = new_sigma;
                    let mut book = synthetic_book(new_price, self.current_sigma, self.cfg.total_inventory);
                    apply_adversarial_deficit(&mut book, self.adversarial.as_ref());
                    self.current_book = Some(book);
                    self.price_history.push_back(new_price);
                }
            }
            RlMode::Historical => {
                let snap = self.lob_replay.as_mut().and_then(|r| r.next_snapshot());
                if let Some(mut book) = snap {
                    let new_raw_mid = book.mid_price().unwrap_or(self.current_mid);
                    // Log-returns computed from raw BTC prices so vol estimate is correct.
                    let prev_mid = self.current_mid.max(1e-12);
                    let log_ret = (new_raw_mid / prev_mid).ln();
                    self.push_return(log_ret);
                    self.current_mid = new_raw_mid;
                    // Apply adversarial deficit before rescaling (deficit is in raw qty units).
                    apply_adversarial_deficit(&mut book, self.adversarial.as_ref());
                    // Rescale to simulation scale so fills and state dims are mode-consistent.
                    // Uses self.price_factor, fixed once at episode arrival (reset_historical) --
                    // NOT recomputed from new_raw_mid here, which would renormalise every
                    // snapshot's mid back to ≈100 and erase real intra-episode price drift
                    // (the bug that made forced-liquidation IS evaluate to ≈0% regardless of
                    // actual market movement).
                    let btc_scale = (self.cfg.btc_target / self.cfg.total_inventory).max(1e-12);
                    let qty_factor = 1.0 / btc_scale;
                    self.current_book = Some(book.rescale_full(self.price_factor, qty_factor));
                    self.price_history.push_back(new_raw_mid);
                    self.current_sigma = self.realised_vol();
                }
                // else: replay exhausted — keep current book (final step will force liquidate)
            }
        }
        // Trim history.
        while self.price_history.len() > 16 {
            self.price_history.pop_front();
        }
    }

    fn push_return(&mut self, r: f64) {
        if r.is_finite() {
            self.return_buffer.push_back(r);
            while self.return_buffer.len() > 50 {
                self.return_buffer.pop_front();
            }
        }
    }

    /// Annualised realised volatility from the rolling log-return buffer.
    fn realised_vol(&self) -> f64 {
        if self.return_buffer.len() < 5 {
            return self.cfg.sigma_ref;
        }
        let n = self.return_buffer.len() as f64;
        let mean = self.return_buffer.iter().sum::<f64>() / n;
        let var = self.return_buffer.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let per_step = var.sqrt();
        // Annualise: multiply by √steps_per_year. Here total_steps ≈ steps_per_episode.
        // Use total_steps as a coarse annualisation factor.
        per_step * (self.cfg.total_steps as f64).sqrt()
    }

    /// Build the 8-dim observation vector.
    fn build_state(&self) -> Vec<f64> {
        let book = self.current_book.as_ref();
        let mid = book.and_then(|b| b.mid_price()).unwrap_or(self.current_mid).max(1e-12);
        let half_spread = book.and_then(|b| b.half_spread()).unwrap_or(self.spread_ref);
        let depth_1pct = book.map(|b| b.bid_depth_up_to(mid * 0.99)).unwrap_or(0.0);

        let total_inv = self.cfg.total_inventory.max(1.0);
        let total_steps = self.cfg.total_steps.max(1) as f64;
        let steps_remaining = (self.cfg.total_steps as i64 - self.step_idx as i64).max(0) as f64;

        let s0 = (self.remaining / total_inv).clamp(0.0, 1.0);
        let s1 = (steps_remaining / total_steps).clamp(0.0, 1.0);
        let s2 = (self.current_sigma / self.cfg.sigma_ref.max(1e-9)).tanh();
        let s3 = (half_spread / self.spread_ref.max(1e-9)).tanh();
        let s4 = (depth_1pct / self.depth_ref.max(1.0)).tanh();
        // OFI from aggTrades for the 30-second window ending at this snapshot.
        // 0.0 in synthetic mode (no trades loaded) and when --agg-trades is not supplied.
        let s5 = {
            let idx = self.ofi_start + self.step_idx;
            self.ofi_series.get(idx).copied().unwrap_or(0.0)
        };
        let s6 = {
            let drift = if self.price_history.len() >= 11 {
                let now = *self.price_history.back().unwrap();
                let then = self.price_history[self.price_history.len() - 11];
                if then.abs() > 1e-12 { (now - then) / then } else { 0.0 }
            } else {
                0.0
            };
            (drift / self.cfg.sigma_ref.max(1e-9)).tanh()
        };
        let s7 = (self.last_slippage / self.spread_ref.max(1e-9)).tanh();

        // Capacity 16 up front (the max possible width, time-features included) so
        // the pushes below never trigger a grow-reallocation -- exactly one
        // allocation per call instead of the 1-2 extra reallocations `vec![...]`
        // followed by growing pushes used to incur (measured via dhat-heap,
        // docs/system_design/phase_s_baseline.md).
        let mut state = Vec::with_capacity(16);
        state.push(s0);
        state.push(s1);
        state.push(s2);
        state.push(s3);
        state.push(s4);
        state.push(s5);
        state.push(s6);
        state.push(s7);
        if let Some(cf) = &self.cf_lob {
            let s8 = (cf.permanent_impact() / 0.5).tanh();
            let s9 = (cf.temporary_impact_total() / 0.3).tanh();
            state.push(s8);
            state.push(s9);
        } else {
            state.push(0.0); // [8] permanent impact fraction — zero in non-counterfactual modes
            state.push(0.0); // [9] temporary impact fraction — zero in non-counterfactual modes
        }
        // [10] vol regime: 0.0=low, 0.5=normal, 1.0=high
        // [11] liq regime: 0.0=thin, 0.5=normal, 1.0=deep
        let (vol_regime, liq_regime) = self.regime_dims();
        state.push(vol_regime);
        state.push(liq_regime);

        if !self.cfg.no_time_features {
            // [12]-[15]: intraday time-of-day features. Binance BTC-USDT liquidity
            // varies dramatically by UTC hour (thin 00:00-04:00, deep at US open);
            // without these an agent cannot learn "it's 2am, wait."
            //
            // current_timestamp_ms comes from the real Binance snapshot timestamp in
            // historical/counterfactual modes. Synthetic mode has no real wall-clock
            // time, so current_book.timestamp defaults to 0 -> time_of_day is a
            // constant 0.0, which is harmless (synthetic training has no intraday
            // structure to learn from regardless).
            let current_timestamp_ms =
                self.current_book.as_ref().map(|b| b.timestamp).unwrap_or(0);
            let seconds_since_midnight =
                (current_timestamp_ms % 86_400_000) as f64 / 1000.0;
            let time_of_day = seconds_since_midnight / 86_400.0;

            // Reuses the existing rolling realised-vol estimate as the "intraday"
            // signal and sigma_ref as the "daily" baseline -- no separate 5-min /
            // daily vol trackers exist, and both are already tracked per-step.
            let intraday_vol_ratio =
                self.current_sigma / (self.cfg.sigma_ref.max(1e-9) + 1e-10);

            state.push(time_of_day);
            state.push((intraday_vol_ratio - 1.0).tanh());
            state.push((2.0 * std::f64::consts::PI * time_of_day).sin());
            state.push((2.0 * std::f64::consts::PI * time_of_day).cos());
        }

        state
    }

    /// Discrete-bucketed (vol_regime, liq_regime) used as state dims [10] and [11].
    ///
    /// vol_regime ∈ {0.0, 0.5, 1.0} via current_sigma vs sigma_ref (×0.7, ×1.3)
    /// liq_regime ∈ {0.0, 0.5, 1.0} via current_half_spread / spread_ref (>2.0, <0.5)
    fn regime_dims(&self) -> (f64, f64) {
        let sigma_ref = self.cfg.sigma_ref.max(1e-9);
        let vol_regime = if self.current_sigma < sigma_ref * 0.7 {
            0.0
        } else if self.current_sigma > sigma_ref * 1.3 {
            1.0
        } else {
            0.5
        };

        let spread_ref = self.spread_ref.max(1e-9);
        let half_spread = self
            .current_book
            .as_ref()
            .and_then(|b| b.half_spread())
            .unwrap_or(spread_ref);
        let spread_ratio = half_spread / spread_ref;
        // Larger spread = thinner book.
        let liq_regime = if spread_ratio > 2.0 {
            0.0 // thin
        } else if spread_ratio < 0.5 {
            1.0 // deep
        } else {
            0.5 // normal
        };

        (vol_regime, liq_regime)
    }

    fn regime_label(&self) -> String {
        let vol_label = if self.current_sigma > self.cfg.sigma_ref * 1.5 {
            "high_vol"
        } else if self.current_sigma < self.cfg.sigma_ref * 0.5 {
            "low_vol"
        } else {
            "normal"
        };
        let spread_label = {
            let hs = self
                .current_book
                .as_ref()
                .and_then(|b| b.half_spread())
                .unwrap_or(self.spread_ref);
            if hs > self.spread_ref * 1.5 {
                "wide"
            } else if hs < self.spread_ref * 0.5 {
                "tight"
            } else {
                "normal"
            }
        };
        format!("{vol_label}/{spread_label}")
    }
}

// ---------------------------------------------------------------------------
// Batched env (Phase S, ADR-006) — N instances per process, sharing one
// MarketDataCache, transported via Arrow RecordBatch over a Unix domain
// socket instead of JSON-over-stdio. Additive: the default (no --batch-size)
// code path above is completely unchanged.
// ---------------------------------------------------------------------------

/// N `RlEnv` instances stepped together, sharing one `MarketDataCache` so a
/// date replayed by multiple instances in the batch is parsed once, not N
/// times (see ADR-006 and `MarketDataCache`'s doc comment).
pub struct BatchedRlEnv {
    envs: Vec<RlEnv>,
    n_state_dims: usize,
}

/// Column-oriented result of one `reset_batch`/`step_batch` call: `states` is
/// row-major N×D (state dims cast to f32 -- the wire/observation precision the
/// Python/torch side trains at; internal computation stays f64), `rewards`/
/// `dones` are length N.
pub struct BatchStepResult {
    pub states: Vec<f32>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub n: usize,
    pub d: usize,
}

impl BatchedRlEnv {
    pub fn new(cfg_template: EnvConfig, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be >= 1");
        let cache = Rc::new(RefCell::new(MarketDataCache::default()));
        let n_state_dims = if cfg_template.no_time_features { 12 } else { 16 };
        let envs = (0..batch_size)
            .map(|_| RlEnv::new_with_shared_cache(cfg_template.clone(), Rc::clone(&cache)))
            .collect();
        Self { envs, n_state_dims }
    }

    pub fn batch_size(&self) -> usize {
        self.envs.len()
    }

    pub fn reset_batch(&mut self, seeds: &[u64]) -> BatchStepResult {
        assert_eq!(seeds.len(), self.envs.len(), "seeds.len() must equal batch_size");
        self.collect(|env, i| env.reset(Some(seeds[i])))
    }

    pub fn step_batch(&mut self, actions: &[usize]) -> BatchStepResult {
        assert_eq!(actions.len(), self.envs.len(), "actions.len() must equal batch_size");
        self.collect(|env, i| env.step(actions[i]))
    }

    /// Reset only the given subset of instances (identified by index into the
    /// batch), each with its own seed, leaving every other instance's
    /// in-progress episode untouched. Required for the SB3 `VecEnv` contract:
    /// when instances finish episodes at different step counts (the normal
    /// case whenever `fixed_steps=False`, i.e. real training with randomised
    /// per-episode horizons -- not the fixed-horizon eval configuration Phase
    /// 0's harness uses), only the just-finished sub-envs auto-reset, not the
    /// whole batch. Returns a result sized to `indices.len()`, not
    /// `batch_size` -- the caller splices these rows into the right batch
    /// positions itself (see `BatchedStrataExecVecEnv.step_wait` in
    /// rl/batched_vec_env.py).
    pub fn reset_indices(&mut self, indices: &[usize], seeds: &[u64]) -> BatchStepResult {
        assert_eq!(indices.len(), seeds.len(), "indices.len() must equal seeds.len()");
        let d = self.n_state_dims;
        let mut states = Vec::with_capacity(indices.len() * d);
        let mut rewards = Vec::with_capacity(indices.len());
        let mut dones = Vec::with_capacity(indices.len());
        for (k, &idx) in indices.iter().enumerate() {
            let r = self.envs[idx].reset(Some(seeds[k]));
            debug_assert_eq!(r.state.len(), d);
            states.extend(r.state.iter().map(|&x| x as f32));
            rewards.push(r.reward as f32);
            dones.push(r.done);
        }
        BatchStepResult { states, rewards, dones, n: indices.len(), d }
    }

    fn collect(&mut self, mut f: impl FnMut(&mut RlEnv, usize) -> StepResult) -> BatchStepResult {
        let n = self.envs.len();
        let d = self.n_state_dims;
        let mut states = Vec::with_capacity(n * d);
        let mut rewards = Vec::with_capacity(n);
        let mut dones = Vec::with_capacity(n);
        for i in 0..n {
            let r = f(&mut self.envs[i], i);
            debug_assert_eq!(r.state.len(), d, "instance {i} produced a {}-dim state, expected {d}", r.state.len());
            states.extend(r.state.iter().map(|&x| x as f32));
            rewards.push(r.reward as f32);
            dones.push(r.done);
        }
        BatchStepResult { states, rewards, dones, n, d }
    }
}

// ---------------------------------------------------------------------------
// Arrow encode/decode for the batched transport
// ---------------------------------------------------------------------------

mod batch_arrow {
    use super::BatchStepResult;
    use arrow::array::{ArrayRef, BooleanArray, Float32Array, Int32Array, UInt32Array, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc as StdArc;

    /// Column names for state dims: "s0", "s1", ..., "s{d-1}".
    fn state_col_name(i: usize) -> String {
        format!("s{i}")
    }

    pub fn result_schema(d: usize) -> Schema {
        let mut fields: Vec<Field> = (0..d)
            .map(|i| Field::new(state_col_name(i), DataType::Float32, false))
            .collect();
        fields.push(Field::new("reward", DataType::Float32, false));
        fields.push(Field::new("done", DataType::Boolean, false));
        Schema::new(fields)
    }

    /// Encode a `BatchStepResult` (row-major N×D states + reward/done) into an
    /// Arrow `RecordBatch` (N rows, D+2 columns: s0..s{D-1}, reward, done).
    /// This is the "zero-copy on decode" side of ADR-006: the Python side reads
    /// each column as a `pyarrow` buffer-view via `to_numpy(zero_copy_only=True)`,
    /// no per-element parsing.
    pub fn encode_result(r: &BatchStepResult) -> RecordBatch {
        let schema = result_schema(r.d);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(r.d + 2);
        for col in 0..r.d {
            let values: Vec<f32> = (0..r.n).map(|row| r.states[row * r.d + col]).collect();
            columns.push(StdArc::new(Float32Array::from(values)));
        }
        columns.push(StdArc::new(Float32Array::from(r.rewards.clone())));
        columns.push(StdArc::new(BooleanArray::from(r.dones.clone())));
        RecordBatch::try_new(StdArc::new(schema), columns)
            .expect("encode_result: column lengths must match schema by construction")
    }

    // The real client is Python (via pyarrow, see rl/batched_environment.py),
    // which builds request RecordBatches itself -- these Rust-side encoders
    // exist only so Rust tests can act as a client without a Python
    // dependency. #[cfg(test)] reflects that honestly rather than suppressing
    // an unused-code warning on functions no production path calls.
    #[cfg(test)]
    pub fn request_schema_step() -> Schema {
        Schema::new(vec![Field::new("action", DataType::Int32, false)])
    }

    #[cfg(test)]
    pub fn request_schema_reset() -> Schema {
        Schema::new(vec![Field::new("seed", DataType::UInt64, false)])
    }

    #[cfg(test)]
    pub fn encode_step_request(actions: &[usize]) -> RecordBatch {
        let values: Vec<i32> = actions.iter().map(|&a| a as i32).collect();
        RecordBatch::try_new(
            StdArc::new(request_schema_step()),
            vec![StdArc::new(Int32Array::from(values))],
        )
        .expect("encode_step_request: schema mismatch")
    }

    #[cfg(test)]
    pub fn encode_reset_request(seeds: &[u64]) -> RecordBatch {
        RecordBatch::try_new(
            StdArc::new(request_schema_reset()),
            vec![StdArc::new(UInt64Array::from(seeds.to_vec()))],
        )
        .expect("encode_reset_request: schema mismatch")
    }

    pub fn decode_step_request(batch: &RecordBatch) -> Vec<usize> {
        let col = batch
            .column_by_name("action")
            .expect("step request batch missing 'action' column")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("'action' column must be Int32");
        (0..col.len()).map(|i| col.value(i) as usize).collect()
    }

    pub fn decode_reset_request(batch: &RecordBatch) -> Vec<u64> {
        let col = batch
            .column_by_name("seed")
            .expect("reset request batch missing 'seed' column")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("'seed' column must be UInt64");
        (0..col.len()).map(|i| col.value(i)).collect()
    }

    /// Indexed-reset request schema: `{index: uint32, seed: uint64}`, one row
    /// per instance to reset. Distinguished from a full reset_batch request
    /// (which has a "seed" column but no "index" column) by the presence of
    /// "index" -- checked first by the server dispatch.
    #[cfg(test)]
    pub fn request_schema_reset_indices() -> Schema {
        Schema::new(vec![
            Field::new("index", DataType::UInt32, false),
            Field::new("seed", DataType::UInt64, false),
        ])
    }

    #[cfg(test)]
    pub fn encode_reset_indices_request(indices: &[usize], seeds: &[u64]) -> RecordBatch {
        let idx_values: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        RecordBatch::try_new(
            StdArc::new(request_schema_reset_indices()),
            vec![
                StdArc::new(UInt32Array::from(idx_values)),
                StdArc::new(UInt64Array::from(seeds.to_vec())),
            ],
        )
        .expect("encode_reset_indices_request: schema mismatch")
    }

    pub fn decode_reset_indices_request(batch: &RecordBatch) -> (Vec<usize>, Vec<u64>) {
        let idx_col = batch
            .column_by_name("index")
            .expect("reset_indices request batch missing 'index' column")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("'index' column must be UInt32");
        let seed_col = batch
            .column_by_name("seed")
            .expect("reset_indices request batch missing 'seed' column")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("'seed' column must be UInt64");
        let indices = (0..idx_col.len()).map(|i| idx_col.value(i) as usize).collect();
        let seeds = (0..seed_col.len()).map(|i| seed_col.value(i)).collect();
        (indices, seeds)
    }
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

fn parse_cli() -> EnvConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = EnvConfig::default();

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
    cfg
}

// ---------------------------------------------------------------------------
// stdin/stdout loop
// ---------------------------------------------------------------------------

fn write_msg(out: &mut impl Write, msg: &OutMsg) -> io::Result<()> {
    // Write straight to the output stream instead of materialising an
    // intermediate String via serde_json::to_string -- one fewer allocation
    // per step on the hot path (measured via dhat-heap,
    // docs/system_design/phase_s_baseline.md).
    serde_json::to_writer(&mut *out, msg)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    out.write_all(b"\n")?;
    out.flush()
}

fn step_result_to_msg(r: StepResult) -> OutMsg {
    OutMsg { state: r.state, reward: r.reward, done: r.done, info: r.info }
}

/// Scans argv for `--batch-size N --socket-path PATH`, the two batch-mode
/// flags that live outside `EnvConfig` (they're process-level, not
/// per-instance). Absent -> the default single-instance JSON-over-stdio path
/// runs unchanged.
fn parse_batch_args() -> Option<(usize, String)> {
    let args: Vec<String> = std::env::args().collect();
    let mut batch_size: Option<usize> = None;
    let mut socket_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--batch-size" => {
                if let Some(v) = args.get(i + 1) {
                    batch_size = v.parse::<usize>().ok();
                    i += 1;
                }
            }
            "--socket-path" => {
                if let Some(v) = args.get(i + 1) {
                    socket_path = Some(v.clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    match (batch_size, socket_path) {
        (Some(n), Some(path)) if n > 0 => Some((n, path)),
        _ => None,
    }
}

/// Serve batched reset/step requests over a Unix domain socket using Arrow
/// RecordBatch IPC (ADR-006). One request = one embedded Arrow IPC stream
/// (schema + one batch + EOS marker) distinguished by column name ("seed" =
/// reset, "action" = step); one response = the same framing, encoding
/// `BatchStepResult`. Accepts exactly one client connection, matching how a
/// single training process drives one batched env.
/// Read one length-prefixed message: a 4-byte little-endian length followed
/// by exactly that many bytes. Returns `None` on EOF/error (client
/// disconnected). See the module doc above `read_framed_message` callers for
/// why this framing exists instead of writing Arrow IPC stream bytes
/// directly to the socket.
fn read_framed_message(stream: &mut impl io::Read) -> Option<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).ok()?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).ok()?;
    Some(buf)
}

fn write_framed_message(stream: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}

/// Serialize one RecordBatch to a self-contained Arrow IPC stream buffer
/// (schema + one batch + EOS marker) in memory.
fn encode_batch_to_bytes(batch: &arrow::record_batch::RecordBatch) -> Vec<u8> {
    use arrow::ipc::writer::StreamWriter;
    let mut buf = Vec::new();
    {
        let mut writer =
            StreamWriter::try_new(&mut buf, &batch.schema()).expect("StreamWriter::try_new");
        writer.write(batch).expect("StreamWriter::write");
        writer.finish().expect("StreamWriter::finish");
    }
    buf
}

fn run_batch_server(cfg_template: EnvConfig, batch_size: usize, socket_path: &str) {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;
    use std::os::unix::net::UnixListener;

    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path)
        .unwrap_or_else(|e| panic!("failed to bind batch socket '{socket_path}': {e}"));
    eprintln!("[rl-env batch] listening on {socket_path}, batch_size={batch_size}");

    let mut batched = BatchedRlEnv::new(cfg_template, batch_size);

    let (mut stream, _addr) = match listener.accept() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[rl-env batch] accept failed: {e}");
            return;
        }
    };

    // NOTE ON FRAMING: each message is length-prefixed (4-byte LE length +
    // exactly that many raw bytes of a self-contained Arrow IPC stream
    // buffer), rather than writing Arrow IPC stream bytes directly to the
    // live socket. pyarrow's high-level stream reader/writer over a shared
    // persistent connection was found (via the Python client's integration
    // test) to read ahead past one message's EOS marker into the next
    // message's bytes, corrupting the second and subsequent exchanges --
    // Arrow's stream format is designed for "the whole file/stream is one
    // Arrow stream," not for interleaving independent request/response
    // messages on one continuous byte channel. Framing each message
    // explicitly and parsing it from an in-memory buffer (bounded, no
    // over-read possible) is the standard fix.
    loop {
        let Some(msg_bytes) = read_framed_message(&mut stream) else {
            break; // client disconnected
        };
        let reader = match StreamReader::try_new(Cursor::new(msg_bytes), None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[rl-env batch] malformed request: {e}");
                break;
            }
        };
        let mut got_batch = None;
        for batch in reader {
            match batch {
                Ok(b) => got_batch = Some(b),
                Err(e) => {
                    eprintln!("[rl-env batch] malformed request: {e}");
                    break;
                }
            }
        }
        let Some(request) = got_batch else { break };

        let result = if request.column_by_name("index").is_some()
            && request.column_by_name("seed").is_some()
        {
            let (indices, seeds) = batch_arrow::decode_reset_indices_request(&request);
            batched.reset_indices(&indices, &seeds)
        } else if request.column_by_name("seed").is_some() {
            let seeds = batch_arrow::decode_reset_request(&request);
            batched.reset_batch(&seeds)
        } else if request.column_by_name("action").is_some() {
            let actions = batch_arrow::decode_step_request(&request);
            batched.step_batch(&actions)
        } else {
            eprintln!("[rl-env batch] request has neither 'seed'/'index' nor 'action' column");
            break;
        };

        let response = batch_arrow::encode_result(&result);
        let response_bytes = encode_batch_to_bytes(&response);
        if write_framed_message(&mut stream, &response_bytes).is_err() {
            break;
        }
    }
    let _ = std::fs::remove_file(socket_path);
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let cfg = parse_cli();

    if let Some((batch_size, socket_path)) = parse_batch_args() {
        run_batch_server(cfg, batch_size, &socket_path);
        return;
    }

    let mut env = RlEnv::new(cfg);

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    // Reused across iterations (cleared, not reallocated) instead of
    // BufRead::lines(), which allocates a fresh String per line -- one fewer
    // allocation per step on the input side of the hot path (measured via
    // dhat-heap, docs/system_design/phase_s_baseline.md).
    let mut line_buf = String::with_capacity(256);
    loop {
        line_buf.clear();
        match stdin.read_line(&mut line_buf) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }
        let trimmed = line_buf.trim();
        if trimmed.is_empty() {
            continue;
        }
        let msg: InMsg = match serde_json::from_str(trimmed) {
            Ok(m) => m,
            Err(e) => {
                let err = OutMsg {
                    state: vec![],
                    reward: 0.0,
                    done: false,
                    info: json!({"error": format!("invalid JSON: {e}")}),
                };
                let _ = write_msg(&mut stdout, &err);
                continue;
            }
        };

        let result = if msg.reset {
            env.reset(msg.seed)
        } else if let Some(a) = msg.action {
            env.step(a)
        } else {
            StepResult {
                state: vec![],
                reward: 0.0,
                done: false,
                info: json!({"error": "expected 'reset' or 'action'"}),
            }
        };

        if write_msg(&mut stdout, &step_result_to_msg(result)).is_err() {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Bit-for-bit hash of one full episode's trajectory: every state vector
    /// (hashed via f64::to_bits(), not float comparison, so this is a true
    /// bit-identity check with no tolerance) plus the terminal reward/inventory.
    /// Two runs with the same seed must produce the same hash, deterministically,
    /// or something in the state machine (RNG usage, HashMap iteration order,
    /// uninitialised memory, the new per-date cache, etc.) is non-deterministic.
    fn run_episode_and_hash(env: &mut RlEnv, seed: u64) -> u64 {
        let mut hasher = DefaultHasher::new();
        let reset_result = env.reset(Some(seed));
        for bit in reset_result.state.iter().map(|f| f.to_bits()) {
            bit.hash(&mut hasher);
        }
        let mut done = reset_result.done;
        let mut action_cycle = 0usize;
        while !done {
            // Deterministic pseudo-varied action sequence (not RNG-driven) so the
            // trajectory actually exercises different code paths across steps,
            // while remaining identical run-to-run for a fixed seed.
            let action = action_cycle % ACTION_FRACTIONS.len();
            action_cycle += 1;
            let r = env.step(action);
            for bit in r.state.iter().map(|f| f.to_bits()) {
                bit.hash(&mut hasher);
            }
            r.reward.to_bits().hash(&mut hasher);
            r.done.hash(&mut hasher);
            done = r.done;
        }
        hasher.finish()
    }

    #[test]
    fn synthetic_episode_hash_identical_across_100_runs_same_seed() {
        let seed = 12345;
        let mut hashes = Vec::with_capacity(100);
        for _ in 0..100 {
            let mut env = make_env();
            hashes.push(run_episode_and_hash(&mut env, seed));
        }
        let first = hashes[0];
        for (i, h) in hashes.iter().enumerate() {
            assert_eq!(
                *h, first,
                "run {i} produced a different trajectory hash than run 0 for the \
                 same seed -- the environment is not deterministic"
            );
        }
    }

    #[test]
    #[cfg(feature = "alloc-timing")]
    #[ignore = "manual benchmark; run with --features alloc-timing -- --ignored --nocapture"]
    fn step_allocation_time_share_of_wall_time() {
        // Follow-up to the dhat-heap allocation-COUNT measurement in
        // docs/system_design/phase_s_baseline.md: 15.01 allocs/step is a count,
        // not a cost. This measures what fraction of a step iteration's actual
        // wall time is spent inside the allocator, now that reset()'s ~239ms
        // no longer dominates -- run with:
        //   cargo test --release --features alloc-timing \
        //     step_allocation_time_share_of_wall_time -- --nocapture --ignored
        // (--ignored because this needs real 2024-01-15 data and is a
        // benchmark, not a correctness assertion; it prints a number rather
        // than asserting a threshold, since "how much allocation costs" is a
        // measurement to report, not a pass/fail gate.)
        let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
        if !std::path::Path::new(lob_path).exists() {
            eprintln!("skipping: {lob_path} not present in this checkout");
            return;
        }

        let cfg = EnvConfig {
            mode: RlMode::Historical,
            total_inventory: 1_000_000.0,
            total_steps: HISTORICAL_DEFAULT_STEPS,
            seed: 42,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: Some(lob_path.to_string()),
            btc_target: 50_000.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: true,
            fixed_size: true,
            no_time_features: true,
        };
        let mut env = RlEnv::new(cfg);

        // Warm the per-date cache (excluded from timing -- this measurement is
        // about the steady-state step() cost, not the already-measured reset()
        // cost).
        env.reset(Some(1));
        env.reset(Some(2));

        const N: usize = 2000;
        let incoming_line = "{\"action\": 0}";
        alloc_timing::reset();
        let wall_t0 = std::time::Instant::now();
        for i in 0..N {
            // Full main()-loop scope, calling the REAL production functions
            // (not reimplementations): parse the incoming action message
            // (serde_json::from_str::<InMsg>, same as main()'s hot loop -- an
            // earlier version of this benchmark skipped this and undercounted
            // allocations relative to the dhat-heap measurement in
            // phase_s_baseline.md, which does include the input side) ->
            // step() (includes build_state) -> step_result_to_msg -> write_msg.
            // io::sink() discards bytes without buffering/allocating, isolating
            // step()+serialization cost from actual socket/pipe I/O syscall
            // latency, which is not allocation.
            let in_msg: InMsg = serde_json::from_str(incoming_line).unwrap();
            let r = env.step(in_msg.action.unwrap_or(0));
            let out_msg = step_result_to_msg(r);
            write_msg(&mut io::sink(), &out_msg).unwrap();
            if i % 500 == 0 {
                // Historical mode terminates after HISTORICAL_DEFAULT_STEPS;
                // reset periodically (not timed as part of the loop average
                // in the sense that matters -- reset() cost is already
                // measured separately and is now ~1.2ms cached, negligible
                // next to N=2000 steps, but exclude it from the allocator
                // share calculation for a clean "steady-state step()" number).
                alloc_timing::reset();
                env.reset(Some((i as u64) % 5 + 1));
                alloc_timing::reset();
            }
        }
        let wall_elapsed = wall_t0.elapsed();
        let (alloc_nanos, alloc_calls) = alloc_timing::snapshot();

        let wall_ns = wall_elapsed.as_nanos() as u64;
        let pct = 100.0 * alloc_nanos as f64 / wall_ns as f64;

        println!("N={N} steps");
        println!("total wall time:      {:.3} ms ({:.1} ns/step)", wall_elapsed.as_secs_f64() * 1000.0, wall_ns as f64 / N as f64);
        println!("time inside allocator: {:.3} ms ({:.1} ns/step, {} calls, {:.2} calls/step)",
            alloc_nanos as f64 / 1e6, alloc_nanos as f64 / N as f64, alloc_calls, alloc_calls as f64 / N as f64);
        println!("allocator share of step wall time: {pct:.1}%");
        println!("NOTE: Instant::now() has ~20-40ns overhead on macOS, called twice per \
                   allocation call here -- this is an order-of-magnitude measurement, not \
                   a precise one.");
    }

    #[test]
    fn historical_lob_episode_hash_identical_across_20_runs_same_seed() {
        // The "Phase S LOB hash" determinism guard called out explicitly by the
        // Phase S plan: real LobReplay data (not synthetic), same seed, run
        // repeatedly, must produce a bit-identical trajectory every time. This is
        // also the guard that would catch a bug in the new per-date MarketDataCache
        // (e.g. Arc-shared snapshot data accidentally mutated or racily reused).
        let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
        if !std::path::Path::new(lob_path).exists() {
            eprintln!("skipping: {lob_path} not present in this checkout");
            return;
        }
        let seed = 7;
        let make_hist_env = || {
            let cfg = EnvConfig {
                mode: RlMode::Historical,
                total_inventory: 1_000_000.0,
                total_steps: HISTORICAL_DEFAULT_STEPS,
                seed: 42,
                sigma_ref: 0.20,
                gamma: 0.005,
                adversarial: false,
                lob_data_path: Some(lob_path.to_string()),
                btc_target: 50_000.0,
                adv_btc: 0.0,
                agg_trades_path: None,
                fixed_steps: true,
                fixed_size: true,
                no_time_features: true,
            };
            RlEnv::new(cfg)
        };

        let mut hashes = Vec::with_capacity(20);
        for _ in 0..20 {
            let mut env = make_hist_env();
            hashes.push(run_episode_and_hash(&mut env, seed));
        }
        let first = hashes[0];
        for (i, h) in hashes.iter().enumerate() {
            assert_eq!(
                *h, first,
                "run {i} produced a different historical-LOB trajectory hash than \
                 run 0 for the same seed"
            );
        }
    }

    #[test]
    fn synthetic_episode_hash_differs_across_seeds() {
        // Sanity check on the hash itself: it must not be a constant that would
        // make the "identical across runs" test trivially/vacuously pass.
        let mut env_a = make_env();
        let mut env_b = make_env();
        let hash_a = run_episode_and_hash(&mut env_a, 1);
        let hash_b = run_episode_and_hash(&mut env_b, 2);
        assert_ne!(hash_a, hash_b, "different seeds must produce different trajectories");
    }

    fn make_env() -> RlEnv {
        let cfg = EnvConfig {
            mode: RlMode::Synthetic,
            total_inventory: 1_000_000.0,
            total_steps: 100,
            seed: 42,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: None,
            btc_target: 500.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: false,
            fixed_size: false,
            no_time_features: true,
        };
        RlEnv::new(cfg)
    }

    #[test]
    fn rl_env_reset_returns_twelve_dimensional_state() {
        let mut env = make_env();
        let r = env.reset(Some(42));
        assert_eq!(
            r.state.len(),
            12,
            "state vector must have exactly 12 dimensions (8 base + 2 impact + 2 regime)"
        );
        assert!(!r.done);
        assert_eq!(r.reward, 0.0);
    }

    #[test]
    fn reset_on_missing_lob_file_terminates_episode_gracefully_instead_of_panicking() {
        // Phase S graceful-degradation test: a missing/corrupt historical data file
        // must force-terminate the episode with a defined penalty, not crash the
        // process (which would kill every other in-flight episode / the whole
        // training run).
        let cfg = EnvConfig {
            mode: RlMode::Historical,
            total_inventory: 1_000_000.0,
            total_steps: 100,
            seed: 42,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: Some("TradeData/BookDepth/does-not-exist-fixture.csv".to_string()),
            btc_target: 500.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: true,
            fixed_size: true,
            no_time_features: true,
        };
        let mut env = RlEnv::new(cfg);
        let r = env.reset(Some(1));

        assert!(r.done, "episode must be marked done, not left half-initialised");
        assert_eq!(
            r.reward, DATA_INGESTION_FAILURE_PENALTY,
            "must apply the defined ingestion-failure penalty, not a normal step reward"
        );
        assert_eq!(
            r.state.len(),
            12,
            "state vector must still be the correct width so the Python side doesn't \
             choke on a malformed observation"
        );
        let err = r.info.get("error").and_then(|v| v.as_str());
        assert!(err.is_some(), "info must explain what went wrong: {:?}", r.info);
        assert_eq!(
            r.info.get("terminated_reason").and_then(|v| v.as_str()),
            Some("data_ingestion_failure")
        );
    }

    #[test]
    fn reset_on_lob_file_with_zero_valid_snapshots_terminates_gracefully() {
        // Distinct failure mode from "file missing": the file opens and has a valid
        // header, but every row fails to parse (e.g. a truncated/corrupted download),
        // leaving zero snapshots -- LobReplay::from_csv returns Err in this case too,
        // and it must be handled the same way, not panic.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "rl_env_zero_snapshot_fixture_{}.csv",
            std::process::id()
        ));
        std::fs::write(&path, "timestamp,percentage,depth,notional\nnot-a-timestamp,-1,1.0,1.0\n")
            .expect("failed to write test fixture");

        let cfg = EnvConfig {
            mode: RlMode::Historical,
            total_inventory: 1_000_000.0,
            total_steps: 100,
            seed: 7,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: Some(path.to_string_lossy().to_string()),
            btc_target: 500.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: true,
            fixed_size: true,
            no_time_features: true,
        };
        let mut env = RlEnv::new(cfg);
        let r = env.reset(Some(1));

        let _ = std::fs::remove_file(&path);

        assert!(r.done);
        assert_eq!(r.reward, DATA_INGESTION_FAILURE_PENALTY);
        assert_eq!(
            r.info.get("terminated_reason").and_then(|v| v.as_str()),
            Some("data_ingestion_failure")
        );
    }

    #[test]
    fn rl_env_action_zero_does_not_change_inventory() {
        let mut env = make_env();
        env.reset(Some(42));
        let before = env.remaining;
        let r = env.step(0);
        assert!(
            (env.remaining - before).abs() < 1e-9,
            "action=0 must not change inventory: before={before} after={}",
            env.remaining
        );
        assert!(r.reward < 0.0, "wait action still incurs inventory pressure");
    }

    #[test]
    fn rl_env_action_nineteen_liquidates_all_remaining() {
        let mut env = make_env();
        env.reset(Some(42));
        let r = env.step(19);
        // The synthetic book has finite depth, so a 1.0×inventory order may have residual.
        // The intent is that action 19 ATTEMPTS to liquidate all remaining; whatever fills,
        // the remaining inventory must shrink dramatically. Check < 1% of initial.
        assert!(
            env.remaining < 0.01 * 1_000_000.0,
            "action=19 must fill ~all inventory; remaining={}",
            env.remaining
        );
        assert!(r.info.get("fill_qty").is_some());
    }

    #[test]
    fn rl_env_done_when_inventory_exhausted() {
        let mut env = make_env();
        env.reset(Some(42));
        let r = env.step(19);
        assert!(
            r.done,
            "after fully liquidating, done must be true (remaining = {})",
            env.remaining
        );
    }

    #[test]
    fn rl_env_done_when_steps_exhausted() {
        let cfg = EnvConfig {
            mode: RlMode::Synthetic,
            total_inventory: 1_000_000.0,
            total_steps: 5, // very short horizon
            seed: 7,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: None,
            btc_target: 500.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: false,
            fixed_size: false,
            no_time_features: true,
        };
        let mut env = RlEnv::new(cfg);
        env.reset(None);

        // Wait 4 times — stays under termination.
        for _ in 0..4 {
            let r = env.step(0);
            assert!(!r.done, "should not be done before exhausting steps");
        }
        // Fifth step should trigger time termination and forced liquidation.
        let r = env.step(0);
        assert!(r.done, "done must be true at the step horizon");
        assert!(
            r.info.get("forced_liquidation_qty").is_some(),
            "step-exhausted termination must report forced_liquidation_qty",
        );
    }

    #[test]
    fn rl_env_reward_negative_for_adverse_fill() {
        let mut env = make_env();
        env.reset(Some(42));
        // Action 9 → 5% of remaining (50k units). Synthetic book has half-spread > 0
        // so the fill is below mid → fill_cost > 0 → reward strictly negative.
        let r = env.step(9);
        assert!(
            r.reward < 0.0,
            "adverse fill must produce negative reward; got {}",
            r.reward
        );
        // is_contribution should also be ≤ 0 (sum of fill_cost + perm_cost, negated).
        let isc = r.info.get("is_contribution").and_then(|v| v.as_f64()).unwrap_or(0.0);
        assert!(isc <= 0.0, "is_contribution must be ≤ 0 for a sell, got {isc}");
    }

    #[test]
    fn rl_env_state_all_values_in_valid_range() {
        let mut env = make_env();
        env.reset(Some(99));
        // Walk 20 steps with a mix of actions and assert every state value is in range.
        let actions = [0, 9, 0, 5, 1, 0, 12, 3, 0, 7, 11, 0, 2, 6, 0, 8, 0, 4, 10, 0];
        for a in actions {
            let r = env.step(a);
            assert_eq!(r.state.len(), 12);
            // [0] inventory fraction in [0,1]
            assert!(r.state[0] >= 0.0 && r.state[0] <= 1.0, "s0 out of range: {}", r.state[0]);
            // [1] time fraction in [0,1]
            assert!(r.state[1] >= 0.0 && r.state[1] <= 1.0, "s1 out of range: {}", r.state[1]);
            // [2..=4] are tanh() of non-negative ratios → in [0, 1)
            for k in 2..=4 {
                assert!(
                    r.state[k] >= 0.0 && r.state[k] < 1.0,
                    "s{k} (tanh of non-negative ratio) out of range: {}",
                    r.state[k]
                );
            }
            // [5] OFI in [-1,1]; always 0.0 in synthetic mode (no agg-trades loaded)
            assert!(r.state[5] >= -1.0 && r.state[5] <= 1.0, "s5 (OFI) out of range: {}", r.state[5]);
            // [6..=7] are signed tanh() → in (-1, 1)
            for k in 6..=7 {
                assert!(
                    r.state[k] > -1.0 && r.state[k] < 1.0,
                    "s{k} (signed tanh) out of range: {}",
                    r.state[k]
                );
            }
            // [8..=9] impact dims: 0.0 in synthetic/historical modes
            assert_eq!(r.state[8], 0.0, "s8 (perm impact) must be 0 in synthetic mode");
            assert_eq!(r.state[9], 0.0, "s9 (temp impact) must be 0 in synthetic mode");
            // [10..=11] regime dims take discrete values {0.0, 0.5, 1.0}.
            for k in 10..=11 {
                let v = r.state[k];
                assert!(
                    v == 0.0 || v == 0.5 || v == 1.0,
                    "s{k} (regime dim) must be in {{0.0, 0.5, 1.0}}, got {v}"
                );
            }
            if r.done {
                break;
            }
        }
    }

    fn make_env_with_time_features() -> RlEnv {
        let cfg = EnvConfig {
            mode: RlMode::Synthetic,
            total_inventory: 1_000_000.0,
            total_steps: 100,
            seed: 42,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: None,
            btc_target: 500.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: false,
            fixed_size: false,
            no_time_features: false,
        };
        RlEnv::new(cfg)
    }

    #[test]
    fn time_features_produce_16_dims() {
        let mut env = make_env_with_time_features();
        let r = env.reset(Some(42));
        assert_eq!(
            r.state.len(),
            16,
            "with time features active, state must have 16 dimensions (12 v5 + 4 time)"
        );
    }

    #[test]
    fn no_time_features_produces_12_dims() {
        let mut env = make_env();
        let r = env.reset(Some(42));
        assert_eq!(
            r.state.len(),
            12,
            "--no-time-features must keep the v5-compatible 12-dim state vector"
        );
    }

    #[test]
    fn time_of_day_in_range_zero_to_one() {
        let mut env = make_env_with_time_features();
        let r = env.reset(Some(42));
        let time_of_day = r.state[12];
        assert!(
            (0.0..=1.0).contains(&time_of_day),
            "time_of_day must be in [0,1], got {time_of_day}"
        );
    }

    #[test]
    fn intraday_vol_ratio_dim_carries_no_information_beyond_s2_no_lookahead() {
        // Phase 2 acceptance criterion (Global Rule 1 + 7): an EXECUTABLE
        // leakage guard for state dim [13] (intraday_vol_ratio), not just
        // the Phase -1 audit's prose verdict. build_state() computes:
        //   s2  = tanh(current_sigma / sigma_ref)
        //   s13 = tanh(current_sigma / sigma_ref - 1.0)
        // i.e. s13 is a pure deterministic function of the SAME
        // current_sigma/sigma_ref ratio s2 already exposes -- it introduces
        // no new data source, so it cannot leak anything s2 doesn't already
        // (and s2's current_sigma is a bounded rolling-window estimator,
        // never a full-episode/future aggregate -- the classic "daily_vol"
        // lookahead bug this guard exists to catch would show up here as
        // s13 diverging from a pure function of s2 at some step).
        let mut env = make_env_with_time_features();
        let mut r = env.reset(Some(7));
        for _ in 0..20 {
            let s2 = r.state[2];
            let s13 = r.state[13];
            // Recover the ratio from s2, then recompute s13 from it.
            let ratio = s2.clamp(-0.999999, 0.999999).atanh();
            let expected_s13 = (ratio - 1.0).tanh();
            assert!(
                (s13 - expected_s13).abs() < 1e-9,
                "s13 ({s13}) must be a pure function of s2's ratio ({s2} -> ratio {ratio}), \
                 got expected {expected_s13} -- any divergence means s13 is drawing on a \
                 data source s2 doesn't already causally have"
            );
            if r.done {
                break;
            }
            r = env.step(5);
        }
    }

    #[test]
    fn cyclical_encoding_bounded() {
        let mut env = make_env_with_time_features();
        let r = env.reset(Some(42));
        assert!(
            r.state[14].abs() <= 1.0 + 1e-6,
            "sin(time_of_day) component out of range: {}",
            r.state[14]
        );
        assert!(
            r.state[15].abs() <= 1.0 + 1e-6,
            "cos(time_of_day) component out of range: {}",
            r.state[15]
        );
    }

    #[test]
    fn sin_cos_identity() {
        let mut env = make_env_with_time_features();
        let r = env.reset(Some(42));
        let identity = r.state[14].powi(2) + r.state[15].powi(2);
        assert!(
            (identity - 1.0).abs() < 0.001,
            "sin^2 + cos^2 must be ~1.0, got {identity}"
        );
    }

    // -----------------------------------------------------------------------
    // Batched env (Phase S, ADR-006)
    // -----------------------------------------------------------------------

    fn synthetic_cfg_template() -> EnvConfig {
        EnvConfig {
            mode: RlMode::Synthetic,
            total_inventory: 1_000_000.0,
            total_steps: 50,
            seed: 42,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: None,
            btc_target: 500.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: true,
            fixed_size: true,
            no_time_features: true,
        }
    }

    #[test]
    fn batched_env_reset_and_step_match_n_single_instances() {
        // Correctness guard: BatchedRlEnv must produce EXACTLY the same numbers
        // as N independent RlEnv instances stepped one at a time -- batching is
        // a performance change, not a semantics change.
        let n = 4;
        let seeds: Vec<u64> = vec![1, 2, 3, 4];
        let actions: Vec<usize> = vec![0, 5, 10, 19];

        let mut batched = BatchedRlEnv::new(synthetic_cfg_template(), n);
        let reset_result = batched.reset_batch(&seeds);
        let step_result = batched.step_batch(&actions);

        for i in 0..n {
            let mut single = RlEnv::new(synthetic_cfg_template());
            let single_reset = single.reset(Some(seeds[i]));
            let single_step = single.step(actions[i]);

            let batch_reset_row = &reset_result.states[i * reset_result.d..(i + 1) * reset_result.d];
            let expected_reset: Vec<f32> = single_reset.state.iter().map(|&x| x as f32).collect();
            assert_eq!(
                batch_reset_row, expected_reset.as_slice(),
                "instance {i}: batched reset state != single-instance reset state"
            );

            let batch_step_row = &step_result.states[i * step_result.d..(i + 1) * step_result.d];
            let expected_step: Vec<f32> = single_step.state.iter().map(|&x| x as f32).collect();
            assert_eq!(
                batch_step_row, expected_step.as_slice(),
                "instance {i}: batched step state != single-instance step state"
            );
            assert_eq!(
                step_result.rewards[i], single_step.reward as f32,
                "instance {i}: batched reward != single-instance reward"
            );
        }
    }

    #[test]
    fn reset_indices_only_resets_specified_instances() {
        // Correctness guard for the SB3 VecEnv auto-reset-on-done contract:
        // reset_indices must reset ONLY the specified instances and leave
        // every other instance's in-progress episode completely untouched
        // (same remaining inventory, same step_idx) -- a batched training
        // wrapper that reset the whole batch when only one instance finished
        // would silently truncate every other instance's episode.
        let n = 4;
        let mut batched = BatchedRlEnv::new(synthetic_cfg_template(), n);
        batched.reset_batch(&[10, 20, 30, 40]);
        // Advance all 4 by a few steps so they have observable in-progress state.
        batched.step_batch(&[5, 5, 5, 5]);
        batched.step_batch(&[5, 5, 5, 5]);
        let before = batched.step_batch(&[5, 5, 5, 5]);

        // Reset only instances 1 and 3.
        let reset_result = batched.reset_indices(&[1, 3], &[99, 100]);
        assert_eq!(reset_result.n, 2);

        // Instances 1 and 3 must match what a fresh reset(seed) produces.
        let mut fresh1 = RlEnv::new(synthetic_cfg_template());
        let fresh1_state = fresh1.reset(Some(99));
        let expected1: Vec<f32> = fresh1_state.state.iter().map(|&x| x as f32).collect();
        assert_eq!(&reset_result.states[0..reset_result.d], expected1.as_slice());

        let mut fresh3 = RlEnv::new(synthetic_cfg_template());
        let fresh3_state = fresh3.reset(Some(100));
        let expected3: Vec<f32> = fresh3_state.state.iter().map(|&x| x as f32).collect();
        assert_eq!(&reset_result.states[reset_result.d..2 * reset_result.d], expected3.as_slice());

        // Instances 0 and 2 must be UNTOUCHED: stepping them further must
        // continue their existing (3-steps-in) trajectory, not a fresh episode.
        let after = batched.step_batch(&[5, 5, 5, 5]);
        let mut continued0 = RlEnv::new(synthetic_cfg_template());
        continued0.reset(Some(10));
        continued0.step(5);
        continued0.step(5);
        continued0.step(5);
        let expected0_step4 = continued0.step(5);
        let expected0: Vec<f32> = expected0_step4.state.iter().map(|&x| x as f32).collect();
        assert_eq!(
            &after.states[0 * after.d..1 * after.d], expected0.as_slice(),
            "instance 0 (not reset) must continue its pre-existing trajectory unchanged"
        );

        let mut continued2 = RlEnv::new(synthetic_cfg_template());
        continued2.reset(Some(30));
        continued2.step(5);
        continued2.step(5);
        continued2.step(5);
        let expected2_step4 = continued2.step(5);
        let expected2: Vec<f32> = expected2_step4.state.iter().map(|&x| x as f32).collect();
        assert_eq!(
            &after.states[2 * after.d..3 * after.d], expected2.as_slice(),
            "instance 2 (not reset) must continue its pre-existing trajectory unchanged"
        );
        let _ = before; // used only to advance state; no direct assertion needed
    }

    #[test]
    fn batch_arrow_round_trip_preserves_values() {
        // Encode a BatchStepResult to Arrow, decode the request columns back,
        // and confirm nothing is lost/reordered in the RecordBatch conversion
        // itself, independent of the socket transport.
        let result = BatchStepResult {
            states: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 3 rows x 2 dims
            rewards: vec![-0.5, -0.25, -0.75],
            dones: vec![false, true, false],
            n: 3,
            d: 2,
        };
        let batch = batch_arrow::encode_result(&result);
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 4); // s0, s1, reward, done

        let seeds = vec![10u64, 20, 30];
        let req = batch_arrow::encode_reset_request(&seeds);
        assert_eq!(batch_arrow::decode_reset_request(&req), seeds);

        let actions = vec![0usize, 5, 19];
        let req = batch_arrow::encode_step_request(&actions);
        assert_eq!(batch_arrow::decode_step_request(&req), actions);
    }

    #[test]
    fn batch_socket_server_round_trip() {
        // End-to-end: spin up run_batch_server on a real Unix socket in a
        // background thread, connect as a client using the same length-prefixed
        // framing a real client (Python's BatchedStrataExecEnv) uses, and
        // confirm reset_batch then step_batch round-trip correctly across
        // MULTIPLE exchanges on one persistent connection. This specific
        // scenario (2nd+ message on a shared connection) is what caught the
        // "pyarrow reads ahead past the stream boundary" bug that motivated
        // the length-prefixed framing in the first place.
        use arrow::ipc::reader::StreamReader;
        use std::io::Cursor;
        use std::os::unix::net::UnixStream;

        let socket_path = std::env::temp_dir().join(format!(
            "rl_env_batch_test_{}_{}.sock",
            std::process::id(),
            42 // fixed per-test suffix; unique per test fn via the temp_dir cleanup below
        ));
        let socket_path_str = socket_path.to_string_lossy().to_string();
        let _ = std::fs::remove_file(&socket_path);

        let server_path = socket_path_str.clone();
        let handle = std::thread::spawn(move || {
            run_batch_server(synthetic_cfg_template(), 3, &server_path);
        });

        // Give the server a moment to bind before connecting.
        let mut stream = None;
        for _ in 0..50 {
            if let Ok(s) = UnixStream::connect(&socket_path_str) {
                stream = Some(s);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let mut stream = stream.expect("failed to connect to batch server socket");

        let request_response = |stream: &mut UnixStream, req: arrow::record_batch::RecordBatch| {
            let bytes = encode_batch_to_bytes(&req);
            write_framed_message(stream, &bytes).unwrap();
            let resp_bytes = read_framed_message(stream).expect("server closed connection");
            let reader = StreamReader::try_new(Cursor::new(resp_bytes), None).unwrap();
            let batches: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
            assert_eq!(batches.len(), 1);
            batches.into_iter().next().unwrap()
        };

        // -- reset_batch --
        let seeds = vec![1u64, 2, 3];
        let response = request_response(&mut stream, batch_arrow::encode_reset_request(&seeds));
        assert_eq!(response.num_rows(), 3);
        assert_eq!(
            response.schema().field(0).name(),
            "s0",
            "response columns must be s0..s{{d-1}}, reward, done"
        );

        // -- step_batch (2nd message on the same connection) --
        let actions = vec![0usize, 1, 2];
        let response = request_response(&mut stream, batch_arrow::encode_step_request(&actions));
        assert_eq!(response.num_rows(), 3);

        // -- a 3rd message, to be extra sure framing holds beyond just "2" --
        let actions = vec![5usize, 6, 7];
        let response = request_response(&mut stream, batch_arrow::encode_step_request(&actions));
        assert_eq!(response.num_rows(), 3);

        // -- reset_indices (partial reset over the wire, the SB3 VecEnv
        // auto-reset path) -- 4th message, only 2 of the 3 instances --
        let response = request_response(
            &mut stream,
            batch_arrow::encode_reset_indices_request(&[0, 2], &[111, 222]),
        );
        assert_eq!(response.num_rows(), 2, "partial reset must return exactly len(indices) rows");

        drop(stream); // closes the connection; server loop exits on next read
        let _ = handle.join();
        let _ = std::fs::remove_file(&socket_path);
    }

    // ── Phase 2R regression tests: reward loophole + price_factor rescaling ──

    #[test]
    fn forced_liquidation_is_not_reward_superior_to_trading() {
        // Regression test for the reward loophole behind the v6 training collapse
        // (Phase 2R incident): forced liquidation used to be priced at a flat
        // `gamma * (forced_liq_qty/total)` penalty (0.005 for liquidating 100%),
        // completely decoupled from the real cost of dumping that size into the
        // book. Against real historical LOB depth with a large-relative-to-depth
        // order, cumulative execution slippage clearly exceeds that flat 0.005 cap
        // -- so "never trade, eat the flat penalty" was reward-superior to actually
        // executing. The RL agent found and converged on exactly this shortcut.
        //
        // Invariant under the fix: spreading execution across the episode must
        // score a HIGHER (less negative) cumulative reward than never trading and
        // being forced to dump 100% of inventory in one shot at the horizon.
        //
        // Uses real historical LOB data (not synthetic) because the synthetic
        // book's depth is generous enough relative to a 1M-unit order that
        // trading slippage never gets close to gamma=0.005 either way -- the loophole
        // only actually bites at realistic size-vs-depth ratios, which is exactly
        // what happened in the real v6 training run this test guards against.
        let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
        if !std::path::Path::new(lob_path).exists() {
            eprintln!("skipping: {lob_path} not present in this checkout");
            return;
        }
        let make_cfg = || EnvConfig {
            mode: RlMode::Historical,
            total_inventory: 5_000_000.0,
            total_steps: HISTORICAL_DEFAULT_STEPS,
            seed: 42,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: Some(lob_path.to_string()),
            btc_target: 50_000.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: true,
            fixed_size: true,
            no_time_features: true,
        };

        // Never-trade policy: action 0 every step, forced-liquidated at the horizon.
        let mut never_trade_env = RlEnv::new(make_cfg());
        never_trade_env.reset(Some(1));
        let mut never_trade_reward = 0.0;
        loop {
            let r = never_trade_env.step(0);
            never_trade_reward += r.reward;
            if r.done {
                assert!(
                    r.info.get("forced_liquidation_qty").is_some(),
                    "never-trade policy must end in forced liquidation of the full size"
                );
                break;
            }
        }

        // Spread-execution policy: sell 15% of *remaining* inventory each step
        // (ACTION_FRACTIONS[12] = 0.15) -- winds down gradually instead of dumping
        // everything at once.
        let mut trading_env = RlEnv::new(make_cfg());
        trading_env.reset(Some(1));
        let mut trading_reward = 0.0;
        loop {
            let r = trading_env.step(12);
            trading_reward += r.reward;
            if r.done {
                break;
            }
        }

        assert!(
            trading_reward > never_trade_reward,
            "trading policy (reward={trading_reward}) must score better than the \
             never-trade policy (reward={never_trade_reward}) -- if this regresses, \
             the forced-liquidation penalty is once again cheaper than real execution"
        );
    }

    #[test]
    fn historical_price_factor_is_fixed_at_episode_start_not_recomputed_per_step() {
        // The bug: advance_market()'s historical branch used to recompute
        // `price_factor = 100.0 / new_raw_mid` fresh on every snapshot, which
        // renormalises the rescaled book's mid back to ≈100 (== arrival) at every
        // step regardless of real price movement. The fix fixes price_factor once
        // at reset_historical and reuses it for the whole episode.
        let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
        if !std::path::Path::new(lob_path).exists() {
            eprintln!("skipping: {lob_path} not present in this checkout");
            return;
        }
        let cfg = EnvConfig {
            mode: RlMode::Historical,
            total_inventory: 1_000_000.0,
            total_steps: HISTORICAL_DEFAULT_STEPS,
            seed: 42,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: Some(lob_path.to_string()),
            btc_target: 50_000.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: true,
            fixed_size: true,
            no_time_features: true,
        };
        let mut env = RlEnv::new(cfg);
        env.reset(Some(42));
        let factor_at_reset = env.price_factor;

        for _ in 0..50 {
            let r = env.step(0); // never trade -- pure market observation
            if r.done {
                break;
            }
        }
        assert_eq!(
            env.price_factor, factor_at_reset,
            "price_factor must stay fixed for the whole episode, not be recomputed per snapshot"
        );

        // With the factor fixed, the rescaled book's mid should drift away from the
        // arrival price (100.0) as the real BTC price moves over the day, instead of
        // being renormalised back to ~100 at every step.
        let mid_after = env
            .current_book
            .as_ref()
            .and_then(|b| b.mid_price())
            .unwrap_or(100.0);
        assert!(
            (mid_after - 100.0).abs() > 1e-6,
            "rescaled book mid should reflect real intraday price drift, not stay \
             pinned at 100.0 (got {mid_after})"
        );
    }

    #[test]
    fn forced_liquidation_is_pct_reflects_real_execution_not_pinned_zero() {
        // The bug this guards against: forced liquidation used to price the entire
        // remaining size at `current_book.mid_price()` without ever crossing the
        // book, and (compounded by the price_factor bug) that mid was always ≈
        // arrival -- so a fully-forced-liquidated episode scored total_IS_pct ≈ 0%
        // regardless of real market movement or spread cost. This is exactly what
        // produced the `-0.0000%` rows in the Phase 2 ablation.
        let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
        if !std::path::Path::new(lob_path).exists() {
            eprintln!("skipping: {lob_path} not present in this checkout");
            return;
        }
        let cfg = EnvConfig {
            mode: RlMode::Historical,
            total_inventory: 1_000_000.0,
            total_steps: 10, // short horizon so we hit forced liquidation quickly
            seed: 42,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: Some(lob_path.to_string()),
            btc_target: 50_000.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: true,
            fixed_size: true,
            no_time_features: true,
        };
        let mut env = RlEnv::new(cfg);
        env.reset(Some(42));

        let mut last = None;
        for _ in 0..10 {
            last = Some(env.step(0)); // never trade -- entire inventory force-liquidated
        }
        let r = last.unwrap();
        assert!(r.done);
        assert!(
            r.info.get("forced_liquidation_qty").is_some(),
            "expected forced liquidation at the horizon"
        );
        let is_pct = r
            .info
            .get("total_IS_pct")
            .and_then(|v| v.as_f64())
            .expect("done step must report total_IS_pct");
        assert!(
            is_pct.abs() > 1e-4,
            "forced-liquidation IS must be non-zero -- dumping 100% of inventory into \
             the book must cross real depth/spread, not price at a static mid \
             (got {is_pct})"
        );
    }
}
