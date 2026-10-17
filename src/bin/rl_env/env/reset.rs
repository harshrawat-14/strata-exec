//! Episode reset for the synthetic, historical and counterfactual modes.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::json;

use strata_exec::market::counterfactual_lob::{CounterfactualLob, CounterfactualLobConfig};
use strata_exec::market::garch::GarchSimulator;
use strata_exec::market::order_book::OrderBook;

use crate::config::*;
use crate::env::*;

impl RlEnv {
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
                self.cfg.total_steps = rng.gen_range(RANDOM_HORIZON_MIN..=RANDOM_HORIZON_MAX);
            }
            if !self.cfg.fixed_size {
                let size_fraction: f64 = rng.gen_range(RANDOM_SIZE_MIN..=RANDOM_SIZE_MAX);
                self.episode_size_fraction = size_fraction;
                self.cfg.total_inventory = (self.base_total_inventory * size_fraction).max(1.0);
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
            let adv_btc = if self.cfg.adv_btc > 0.0 {
                self.cfg.adv_btc
            } else {
                240_000.0
            };
            if let Some(obj) = info.as_object_mut() {
                obj.insert("btc_scale".to_string(), json!(btc_scale));
                obj.insert("adv_btc".to_string(), json!(adv_btc));
                obj.insert("depth_ref".to_string(), json!(cf.depth_ref));
            }
        }
        StepResult {
            state,
            reward: 0.0,
            done: false,
            info,
        }
    }

    pub(crate) fn reset_synthetic(&mut self, seed: u64) {
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

    pub(crate) fn reset_historical(&mut self, seed: u64) -> Result<(), String> {
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
                    self.volume_series = agg.build_volume_series(&timestamps);
                    self.ofi_start = start_offset;
                    self.agg_trades = Some(agg);
                }
                Err(e) => {
                    eprintln!("AggTrades load error (historical): {e}");
                    self.agg_trades = None;
                    self.ofi_series.clear();
                    self.volume_series.clear();
                    self.ofi_start = 0;
                }
            }
        } else {
            self.agg_trades = None;
            self.ofi_series.clear();
            self.volume_series.clear();
            self.ofi_start = 0;
        }

        self.lob_replay = Some(replay);
        self.sim = None;
        Ok(())
    }

    pub(crate) fn reset_counterfactual(&mut self, seed: u64) -> Result<(), String> {
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
        // Phase 2R follow-up fix: spread_ref MUST be computed on the RESCALED
        // (simulation-scale, ~100) book, not the raw BTC-dollar-scale snapshot --
        // historical mode's reset_historical already does this correctly
        // (scaled_first.half_spread()), but this counterfactual path used the raw
        // `first` directly. That scale mismatch (spread_ref ~$20-50 raw vs. the
        // ~0.03 sim-scale half_spread it's later compared against every step) made
        // state dim s3 collapse toward 0 after the first step (only briefly, at
        // t=0, matching by coincidence since current_book was also still raw then)
        // AND made the forced-liquidation punitive-price formula
        // (worst_price - 5*spread_ref) go deeply negative, flooring at 1e-6 --
        // pricing ~80-90% of any forced liquidation at an effectively-zero price
        // rather than a genuinely punitive-but-nonzero one. Found while
        // quantifying the calibrated-impact vs. punitive-extrapolation split of
        // the forced-liquidation reward (Task 3 follow-up).
        let real_mid_for_spread = first.mid_price().unwrap_or(100.0);
        let spread_price_factor = 100.0 / real_mid_for_spread.max(1e-9);
        let spread_qty_factor = 1.0 / btc_scale;
        let scaled_first_for_spread = first.rescale_full(spread_price_factor, spread_qty_factor);
        self.spread_ref = scaled_first_for_spread
            .half_spread()
            .unwrap_or(0.01)
            .max(1e-9);
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
                    self.volume_series = agg.build_volume_series(&timestamps);
                    self.ofi_start = start_offset;
                    self.agg_trades = Some(agg);
                }
                Err(e) => {
                    eprintln!("AggTrades load error (counterfactual): {e}");
                    self.agg_trades = None;
                    self.ofi_series.clear();
                    self.volume_series.clear();
                    self.ofi_start = 0;
                }
            }
        } else {
            self.agg_trades = None;
            self.ofi_series.clear();
            self.volume_series.clear();
            self.ofi_start = 0;
        }

        self.lob_replay = None;
        self.sim = None;

        // Build CounterfactualLob on the first episode; reuse and explicitly reset on
        // subsequent resets so accumulated impact from the previous episode never carries over.
        if self.cf_lob.is_none() {
            let cf_replay = self.market_cache.borrow_mut().load_lob(&path)?;
            // Phase 2R Task 2: calibrate impact_coefficient/sigma from this date's
            // real aggTrades + LOB snapshot mids, instead of the hardcoded
            // (impact_coefficient=0.1, sigma=1.0) defaults, when aggTrades data
            // is available. Falls back to the defaults (unchanged prior
            // behavior) if aggTrades were not supplied or calibration yields no
            // valid trades (e.g. none bracket the snapshot window).
            let cf_config = if let Some(agg) = self.agg_trades.as_ref() {
                let all_timestamps = cf_replay.snapshot_timestamps();
                let all_mids: Vec<f64> = cf_replay
                    .snapshots_arc()
                    .iter()
                    .map(|s| s.to_order_book().mid_price().unwrap_or(0.0))
                    .collect();
                let daily_volume_btc = adv_btc; // same denominator the sim will use
                let calibrated = CounterfactualLobConfig::calibrated(
                    cf_config,
                    agg,
                    &all_timestamps,
                    &all_mids,
                    daily_volume_btc,
                );
                eprintln!(
                    "  Calibrated impact: Y={:.4} (default 0.1), sigma={:.4} (default 1.0)",
                    calibrated.impact_coefficient, calibrated.sigma
                );
                calibrated
            } else {
                cf_config
            };
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
}
