//! Market advance, volatility estimate, regime signals and the state vector.

use crate::config::*;
use crate::env::*;

impl RlEnv {
    /// Traded BTC volume in the current step's snapshot interval (0.0 in
    /// synthetic mode or when --agg-trades is absent). Same index basis as the
    /// s5 OFI dim (ofi_start + step_idx). Exposed in info for VWAP weighting.
    pub(crate) fn interval_volume_at_step(&self) -> f64 {
        let idx = self.ofi_start + self.step_idx;
        self.volume_series.get(idx).copied().unwrap_or(0.0)
    }

    /// Step the price simulator (synthetic) or pull the next snapshot (historical).
    /// Counterfactual mode advances through `cf_lob` inside `step_counterfactual`.
    pub(crate) fn advance_market(&mut self) {
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
                    let mut book =
                        synthetic_book(new_price, self.current_sigma, self.cfg.total_inventory);
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

    pub(crate) fn push_return(&mut self, r: f64) {
        if r.is_finite() {
            self.return_buffer.push_back(r);
            while self.return_buffer.len() > 50 {
                self.return_buffer.pop_front();
            }
        }
    }

    /// Annualised realised volatility from the rolling log-return buffer.
    pub(crate) fn realised_vol(&self) -> f64 {
        if self.return_buffer.len() < 5 {
            return self.cfg.sigma_ref;
        }
        let n = self.return_buffer.len() as f64;
        let mean = self.return_buffer.iter().sum::<f64>() / n;
        let var = self
            .return_buffer
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / n;
        let per_step = var.sqrt();
        // Annualise against the FIXED TRADING_STEPS_PER_YEAR, never
        // cfg.total_steps (the per-episode horizon, randomised 200-1440
        // during training) — see that constant's doc comment. The previous
        // `(cfg.total_steps as f64).sqrt()` was off by ~19x (sqrt(2880) vs
        // sqrt(2880*365)) AND varied current_sigma with episode length.
        per_step * TRADING_STEPS_PER_YEAR.sqrt()
    }

    /// Build the 8-dim observation vector.
    pub(crate) fn build_state(&self) -> Vec<f64> {
        let book = self.current_book.as_ref();
        let mid = book
            .and_then(|b| b.mid_price())
            .unwrap_or(self.current_mid)
            .max(1e-12);
        let half_spread = book
            .and_then(|b| b.half_spread())
            .unwrap_or(self.spread_ref);
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
                if then.abs() > 1e-12 {
                    (now - then) / then
                } else {
                    0.0
                }
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
        // docs/performance/baseline-profiling.md).
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
        // [10] vol regime: tanh(ln(current_sigma/sigma_ref)); 0=at reference,
        //      positive=more volatile than reference, negative=calmer.
        // [11] liq regime: tanh(ln(half_spread/spread_ref)); 0=at reference,
        //      positive=wider spread than reference (thin), negative=deep.
        // See regime_dims()'s doc comment for the Phase 3 fix this replaced.
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
            let current_timestamp_ms = self.current_book.as_ref().map(|b| b.timestamp).unwrap_or(0);
            let seconds_since_midnight = (current_timestamp_ms % 86_400_000) as f64 / 1000.0;
            let time_of_day = seconds_since_midnight / 86_400.0;

            // Reuses the existing rolling realised-vol estimate as the "intraday"
            // signal and sigma_ref as the "daily" baseline -- no separate 5-min /
            // daily vol trackers exist, and both are already tracked per-step.
            let intraday_vol_ratio = self.current_sigma / (self.cfg.sigma_ref.max(1e-9) + 1e-10);

            state.push(time_of_day);
            state.push((intraday_vol_ratio - 1.0).tanh());
            state.push((2.0 * std::f64::consts::PI * time_of_day).sin());
            state.push((2.0 * std::f64::consts::PI * time_of_day).cos());
        }

        state
    }

    /// Continuous regime signals used as state dims [10] and [11].
    ///
    /// Phase 3 fix (docs/proposals/research-roadmap.md, sections C/D) replacing the prior
    /// 3-way hard-bucketed version. That version was measured to be pinned
    /// at LOW/"normal" on every real date this project has data for, calm
    /// or crash alike — not because of a lookahead leak (there is none:
    /// `sigma_ref` is a config-time constant, never derived from the
    /// episode's own future), but because `sigma_ref` was hardcoded at 0.20
    /// (a plausible BTC vol level) while `current_sigma`'s annualisation
    /// was separately wrong by ~19x, landing every real date's ratio inside
    /// the "normal" band by simple arithmetic. Both are fixed above
    /// (`CALIBRATED_SIGMA_REF`, `TRADING_STEPS_PER_YEAR`); fixing only one
    /// of the two would move every date to the OTHER dead rail instead —
    /// see the derivation comment on `CALIBRATED_SIGMA_REF`.
    ///
    /// Both signals are `tanh(ln(ratio))`: 0.0 at the reference, positive =
    /// more stressed than reference (higher vol / wider spread), negative =
    /// calmer. Log-space is deliberate, not just "more continuous than a
    /// bucket": a raw-ratio `tanh(ratio)` saturates hard at the ratios now
    /// reachable with a correct reference (crash-day vol ratios reach
    /// ~2.8-4.1x sigma_ref; tanh(4.1) = 0.9994 alone, which would erase the
    /// 2024-03-05 vs 2024-08-05 severity difference this project's severity
    /// analysis depends on). In log space they stay separated:
    /// tanh(ln 2.8) ≈ 0.774 vs tanh(ln 4.1) ≈ 0.886.
    ///
    /// NOTE for anyone diffing against the old encoding (e.g.
    /// `rl/phase2r_urgency_diagnosis.py`'s "s11: liq_regime (discrete
    /// 0/.5/1)" comment, now stale): the liquidity sign convention changed.
    /// Old: 0.0=thin(bad), 1.0=deep(good). New: positive=thin/wider spread
    /// (bad), matching vol's positive=stressed convention, so both dims
    /// agree on which sign means "worse."
    pub(crate) fn regime_dims(&self) -> (f64, f64) {
        let sigma_ref = self.cfg.sigma_ref.max(1e-9);
        let vol_ratio = (self.current_sigma.max(1e-9) / sigma_ref).max(1e-9);
        let vol_signal = vol_ratio.ln().tanh();

        let spread_ref = self.spread_ref.max(1e-9);
        let half_spread = self
            .current_book
            .as_ref()
            .and_then(|b| b.half_spread())
            .unwrap_or(spread_ref);
        let liq_ratio = (half_spread.max(1e-9) / spread_ref).max(1e-9);
        let liq_signal = liq_ratio.ln().tanh();

        (vol_signal, liq_signal)
    }

    pub(crate) fn regime_label(&self) -> String {
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
