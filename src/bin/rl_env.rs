// rl_env — Long-running RL environment server.
//
// Python controls this process via line-delimited JSON over stdin/stdout.
// One JSON object per line in both directions.
//
//   reset:  {"reset": true, "seed": 42}
//   action: {"action": <0..=19>}
//
// The response is a JSON object {state, reward, done, info}.

use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use strata_exec::market::agg_trades::AggTradesDay;
use strata_exec::market::adversarial::{FrontRunner, LiquidityWithdrawer};
use strata_exec::market::counterfactual_lob::{CounterfactualLob, CounterfactualLobConfig};
use strata_exec::market::garch::GarchSimulator;
use strata_exec::market::gbm::PriceSimulator;
use strata_exec::market::lob_generator::{generate_lob, LobGeneratorParams};
use strata_exec::market::lob_replay::LobReplay;
use strata_exec::market::order_book::OrderBook;

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
    agg_trades: Option<AggTradesDay>,
    /// Precomputed OFI per snapshot, indexed across the full replay (not offset-adjusted).
    ofi_series: Vec<f64>,
    /// Snapshot index of episode start; ofi_series[ofi_start + step_idx] = OFI at step.
    ofi_start: usize,
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
        Self {
            cfg,
            sim: None,
            lob_replay: None,
            current_book: None,
            current_mid: 0.0,
            arrival_price: 0.0,
            spread_ref: 1.0,
            depth_ref: 1.0,
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
        }
    }

    /// Reset the environment for a new episode.
    pub fn reset(&mut self, seed_override: Option<u64>) -> StepResult {
        let seed = seed_override.unwrap_or(self.cfg.seed);
        self.step_idx = 0;
        self.remaining = self.cfg.total_inventory;
        self.price_history.clear();
        self.return_buffer.clear();
        self.last_slippage = 0.0;
        self.total_notional_received = 0.0;
        self.total_filled = 0.0;
        self.current_sigma = self.cfg.sigma_ref;

        match self.cfg.mode {
            RlMode::Synthetic => self.reset_synthetic(seed),
            RlMode::Historical => self.reset_historical(seed),
            RlMode::Counterfactual => self.reset_counterfactual(seed),
        }

        // Reward-scale diagnostic (printed to stderr at each reset).
        // wait_at_full = -(var_penalty + inv_pressure) × 0.0001; at q=1, sigma=sigma_ref.
        let wait_at_full = -(0.01_f64 + 0.0002_f64) * 0.0001;
        eprintln!("Reward scale (new): wait_at_full_inv={:.2e}", wait_at_full);

        let state = self.build_state();
        let mut info = json!({
            "step": 0_usize,
            "remaining": self.remaining,
            "arrival_price": self.arrival_price,
            "action_9_fraction": ACTION_FRACTIONS[9],
        });
        if let Some(cf) = &self.cf_lob {
            let btc_scale = self.cfg.btc_target / self.cfg.total_inventory;
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

    fn reset_historical(&mut self, seed: u64) {
        let path = self
            .cfg
            .lob_data_path
            .clone()
            .expect("historical mode requires --lob-data");
        let mut replay = LobReplay::from_csv(&path).expect("failed to load --lob-data CSV");
        let n = replay.len();
        // Auto-align total_steps to replay length.
        if self.cfg.total_steps == DEFAULT_TOTAL_STEPS
            || self.cfg.total_steps == HISTORICAL_DEFAULT_STEPS
        {
            self.cfg.total_steps = n;
        }

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
        self.depth_ref = (depth_sum / count.max(1) as f64).max(1.0);

        // Random start within the first third of the day.
        // Different seeds → different arrival prices → non-zero IS variance across episodes.
        let max_start = (replay.len() / 3).max(1);
        let start_offset = (seed as usize) % max_start;
        replay.seek(start_offset);

        // Pull the snapshot at start_offset as the initial book.
        let first = replay.next_snapshot().expect("LobReplay must yield ≥1 snapshot");
        let raw_btc_mid = first.mid_price().unwrap_or(100.0);
        self.spread_ref = first.half_spread().unwrap_or(0.01).max(1e-9);
        // current_mid stays at the raw BTC price so log-returns between consecutive
        // raw-BTC snapshots are computed correctly for the volatility estimator.
        self.current_mid = raw_btc_mid;
        self.current_book = Some(first);
        self.price_history.push_back(self.current_mid);
        // FIX A: IS reward formula uses arrival_price as the normalisation denominator.
        // Set to simulation scale (100.0) so reward magnitudes are mode-independent.
        self.arrival_price = 100.0;

        // Precompute OFI series from aggTrades before replay is moved into self.
        if let Some(agg_path) = self.cfg.agg_trades_path.clone() {
            let timestamps = replay.snapshot_timestamps();
            match AggTradesDay::from_csv(&agg_path) {
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
    }

    fn reset_counterfactual(&mut self, seed: u64) {
        let path = self
            .cfg
            .lob_data_path
            .clone()
            .expect("counterfactual mode requires --lob-data");
        let mut replay = LobReplay::from_csv(&path).expect("failed to load --lob-data CSV");
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

        let first = replay.next_snapshot().expect("LobReplay must yield ≥1 snapshot");
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
            match AggTradesDay::from_csv(&agg_path) {
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
            let cf_replay = LobReplay::from_csv(&path).expect("failed to load --lob-data CSV for cf_lob");
            self.cf_lob = Some(CounterfactualLob::new(cf_replay, cf_config));
        }
        let cf = self.cf_lob.as_mut().unwrap();
        cf.reset();
        cf.seek(start_offset);
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
            // Forced liquidation at last mid price; cost = permanent impact only.
            forced_liq_qty = self.remaining;
            let liquidated_at = self.current_mid;
            self.total_filled += forced_liq_qty;
            self.total_notional_received += forced_liq_qty * liquidated_at;
            let forced_perm = self.cfg.gamma * (forced_liq_qty / total);
            total_reward_adj -= forced_perm;
            self.remaining = 0.0;
            done = true;
        }

        let mut info = json!({
            "step": self.step_idx,
            "remaining": self.remaining,
            "fill_qty": fill_qty,
            "fill_price": fill_price,
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
            forced_liq_qty = self.remaining;
            let forced_perm = self.cfg.gamma * (forced_liq_qty / total);
            self.total_filled += forced_liq_qty;
            self.total_notional_received += forced_liq_qty * self.current_mid;
            total_reward_adj -= forced_perm;
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
                    let new_mid = book.mid_price().unwrap_or(self.current_mid);
                    let prev_mid = self.current_mid.max(1e-12);
                    let log_ret = (new_mid / prev_mid).ln();
                    self.push_return(log_ret);
                    self.current_mid = new_mid;
                    apply_adversarial_deficit(&mut book, self.adversarial.as_ref());
                    self.current_book = Some(book);
                    self.price_history.push_back(new_mid);
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

        let mut state = vec![s0, s1, s2, s3, s4, s5, s6, s7];
        if let Some(cf) = &self.cf_lob {
            let s8 = (cf.permanent_impact() / 0.5).tanh();
            let s9 = (cf.temporary_impact_total() / 0.3).tanh();
            state.push(s8);
            state.push(s9);
        } else {
            state.push(0.0); // [8] permanent impact fraction — zero in non-counterfactual modes
            state.push(0.0); // [9] temporary impact fraction — zero in non-counterfactual modes
        }
        state
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
    let s = serde_json::to_string(msg).expect("OutMsg must serialise");
    writeln!(out, "{s}")?;
    out.flush()
}

fn step_result_to_msg(r: StepResult) -> OutMsg {
    OutMsg { state: r.state, reward: r.reward, done: r.done, info: r.info }
}

fn main() {
    let cfg = parse_cli();
    let mut env = RlEnv::new(cfg);

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
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
        };
        RlEnv::new(cfg)
    }

    #[test]
    fn rl_env_reset_returns_ten_dimensional_state() {
        let mut env = make_env();
        let r = env.reset(Some(42));
        assert_eq!(r.state.len(), 10, "state vector must have exactly 10 dimensions");
        assert!(!r.done);
        assert_eq!(r.reward, 0.0);
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
            assert_eq!(r.state.len(), 10);
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
            if r.done {
                break;
            }
        }
    }
}
