//! The single-instance RL environment: state, construction, and the
//! adversarial-agent state it carries.

mod execution;
mod reset;
mod state;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

use serde_json::Value;

use strata_exec::market::adversarial::{FrontRunner, LiquidityWithdrawer};
use strata_exec::market::agg_trades::AggTradesDay;
use strata_exec::market::counterfactual_lob::CounterfactualLob;
use strata_exec::market::gbm::PriceSimulator;
use strata_exec::market::lob_generator::{generate_lob, LobGeneratorParams};
use strata_exec::market::lob_replay::LobReplay;
use strata_exec::market::order_book::OrderBook;

use crate::config::*;
use crate::market_cache::MarketDataCache;

/// Build a synthetic LOB whose total depth is sized to the configured inventory.
pub(crate) fn synthetic_book(mid: f64, sigma: f64, total_inventory: f64) -> OrderBook {
    let params = LobGeneratorParams {
        num_levels: SYNTHETIC_NUM_LEVELS,
        base_quantity: (total_inventory / SYNTHETIC_DEPTH_SCALE).max(1_000.0),
        ..LobGeneratorParams::default()
    };
    generate_lob(mid, sigma, &params)
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
    pub(crate) current_book: Option<OrderBook>,
    current_mid: f64,

    arrival_price: f64,
    pub(crate) spread_ref: f64,
    depth_ref: f64,
    /// Historical-mode price rescale factor (100.0 / arrival raw-BTC mid), fixed once
    /// at `reset_historical` and reused for every snapshot in the episode. Must NOT be
    /// recomputed per-step -- doing so renormalises every book back to mid≈100 and
    /// erases real intra-episode price drift (see `advance_market`).
    pub(crate) price_factor: f64,

    step_idx: usize,
    pub(crate) remaining: f64,

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
    pub(crate) cf_lob: Option<CounterfactualLob>,

    /// Loaded aggTrades for OFI computation (historical/counterfactual modes).
    agg_trades: Option<Arc<AggTradesDay>>,
    /// Precomputed OFI per snapshot, indexed across the full replay (not offset-adjusted).
    ofi_series: Vec<f64>,
    /// Snapshot index of episode start; ofi_series[ofi_start + step_idx] = OFI at step.
    ofi_start: usize,
    /// Precomputed traded BTC volume per snapshot interval, same indexing as
    /// ofi_series (offset by ofi_start). Exposed per-step in info as `interval_volume`
    /// so the Phase 2R VWAP benchmark can weight the price path by real volume.
    volume_series: Vec<f64>,

    /// Parsed-per-date cache shared across resets within this process (see
    /// `MarketDataCache`). `Rc<RefCell<_>>` (not a plain field) so a batched env's
    /// N instances can share ONE cache instead of each parsing independently --
    /// single-threaded (one process, sequential stepping), so RefCell is sufficient.
    market_cache: Rc<RefCell<MarketDataCache>>,

    /// Snapshot of `cfg.total_steps` at construction time. The per-episode
    /// `cfg.total_steps` is rebuilt from this each reset (random or fixed).
    base_total_steps: usize,
    /// Protocol v3: requested start position for the episode being reset.
    start_frac: Option<f64>,
    /// Protocol v3: (start_offset, max_start) chosen for the current episode.
    episode_start: Option<(usize, usize)>,
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
pub(crate) struct AdversarialState {
    withdrawer: LiquidityWithdrawer,
    front_runner: FrontRunner,
    last_front_runner_qty: f64,
    last_depth_withdrawn: f64,
    last_pre_fill_depth: f64,
}

/// When a fresh book is sourced (synthetic regeneration or new historical snapshot),
/// carry over the LiquidityWithdrawer's outstanding per-level deficit so the
/// adversary's effect persists across step boundaries.
pub(crate) fn apply_adversarial_deficit(book: &mut OrderBook, adv: Option<&AdversarialState>) {
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
    pub fn new_with_shared_cache(
        cfg: EnvConfig,
        market_cache: Rc<RefCell<MarketDataCache>>,
    ) -> Self {
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
            volume_series: Vec::new(),
            market_cache,
            base_total_steps,
            start_frac: None,
            episode_start: None,
            base_total_inventory,
            episode_size_fraction: 1.0,
        }
    }
}
