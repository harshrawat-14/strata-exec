// CounterfactualLob — a real order book replayed, adjusted for our own market impact.
//
// The real book is replayed snapshot by snapshot; our selling shifts the WHOLE
// book's prices down by the impact I_t, leaving its depth as recorded:
//
//     I_t = Y * sigma_d * sqrt(D_t / V)
//     D_t = pi * Q_t + (1 - pi) * sum_s q_s * exp(-rho * (t - s))
//
// Q_t is everything sold so far, q_s each sale; pi (0.3) is the share of impact
// that never recovers, the rest decays at rho (half-life ~400 steps at 30 s).
// sigma_d is the date's daily volatility and V its daily volume. Impact is a
// concave function of the accumulated (decayed) quantity -- the square-root law
// for a metaorder -- not a sum of per-slice square roots, which would make
// slicing an order into many pieces absurdly expensive.
//
// Y is fitted on real signed order flow (rl/experiments/impact_fit_signed_flow.py:
// 30 s mid returns regressed on sign(Q) * sigma_d * sqrt(|Q| / V)), frozen at the
// training-date median like lambda. Era 1 instead (a) fitted Y by attributing
// every 30 s price move to every trade -- a volatility measure, not impact -- and
// (b) applied it as a cut in bid DEPTH, so our selling never moved the price.
//
// LobReplay is owned and advanced here; do not advance it externally.

use crate::market::lob_replay::LobReplay;
use crate::market::order_book::OrderBook;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct CounterfactualLobConfig {
    /// Fraction of impact that is permanent (OW-style temporal split: 0.3).
    pub permanent_fraction: f64,
    /// Fraction of impact that is temporary (= 1 - permanent_fraction).
    pub temporary_fraction: f64,
    /// Recovery speed for temporary impact. rho=5 with dt=1/2880 → half-life ≈ 400 steps.
    pub rho: f64,
    /// Time per step (1.0 / total_steps).
    pub dt: f64,
    /// Simulation units → BTC conversion (e.g. 50_000 BTC / 1_000_000 units = 0.05).
    pub btc_scale: f64,
    /// Daily volume V in BTC (the impact law's denominator). Set per date by
    /// `calibrated()` from the date's trades; 240k is a placeholder.
    pub adv_btc: f64,
    /// Impact coefficient Y in `I = Y * sigma * sqrt(D / V)`.
    pub impact_coefficient: f64,
    /// Daily volatility sigma_d (a fraction, not annualised). Set per date by
    /// `calibrated()`; 0.02 is a placeholder.
    pub sigma: f64,
}

impl Default for CounterfactualLobConfig {
    fn default() -> Self {
        Self {
            permanent_fraction: 0.3,
            temporary_fraction: 0.7,
            rho: 5.0,
            dt: 1.0 / 2880.0,
            btc_scale: 0.05,
            adv_btc: 240_000.0,
            impact_coefficient: CALIBRATED_IMPACT_Y,
            sigma: 0.02,
        }
    }
}

/// Impact coefficient Y: median over the 12 v3 training dates of the
/// signed-flow fit at 30 s (range 0.91-1.14; all 25 dates 0.91-1.33), on
/// Bybit BTCUSDT 2024 (rl/results/impact_fit_signed_flow.csv). Frozen.
pub const CALIBRATED_IMPACT_Y: f64 = 1.0715;

impl CounterfactualLobConfig {
    /// The base config with the date's own daily volatility (from the
    /// snapshot mids) and daily volume (from its trades). Y stays at the
    /// base value: it is fitted once, across training dates, not per date.
    pub fn calibrated(
        base: CounterfactualLobConfig,
        agg: Option<&crate::market::agg_trades::AggTradesDay>,
        snapshot_mids: &[f64],
    ) -> Self {
        let sigma = daily_volatility(snapshot_mids);
        let volume: f64 = agg
            .map(|a| a.trades.iter().map(|t| t.quantity_btc).sum())
            .unwrap_or(0.0);
        CounterfactualLobConfig {
            sigma: if sigma > 1e-9 { sigma } else { base.sigma },
            adv_btc: if volume > 0.0 { volume } else { base.adv_btc },
            ..base
        }
    }
}

/// Daily volatility from one day's snapshot mids: stdev of consecutive log
/// returns times sqrt(number of returns).
pub fn daily_volatility(mids: &[f64]) -> f64 {
    let r: Vec<f64> = mids
        .windows(2)
        .filter(|w| w[0] > 1e-9 && w[1] > 1e-9)
        .map(|w| (w[1] / w[0]).ln())
        .collect();
    if r.len() < 2 {
        return 0.0;
    }
    let n = r.len() as f64;
    let mean = r.iter().sum::<f64>() / n;
    let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (var * n).sqrt()
}

// ---------------------------------------------------------------------------
// CounterfactualLob
// ---------------------------------------------------------------------------

pub struct CounterfactualLob {
    replay: LobReplay,
    config: CounterfactualLobConfig,

    /// Total BTC sold so far this episode (its permanent share never decays).
    pub(crate) sold_btc: f64,

    /// (time_when_traded, btc) for each sale, decayed for the temporary share.
    temporary_impacts: Vec<(f64, f64)>,

    current_step: usize,

    /// Total bid depth of the first snapshot — calibration reference.
    pub depth_ref: f64,

    /// The current Binance snapshot (pre-impact).
    current_book: Option<OrderBook>,

    /// Fixed simulation mid price (100.0). Real BTC prices are normalised to this.
    simulation_mid_price: f64,

    /// Price rescale factor (simulation_mid_price / arrival raw mid), fixed once
    /// per episode (at new()/reset()/seek()) and reused for every snapshot in
    /// that episode. Phase 2R Task 3 fix: this used to be recomputed fresh from
    /// the CURRENT book's mid on every rescale_book() call, which renormalises
    /// every snapshot back to mid==simulation_mid_price and silently discards
    /// all real intraday price drift -- the counterfactual path's analogue of
    /// the historical-mode price_factor bug fixed in commit 5630e5d, which was
    /// never applied here. This is WHY dim 6 (short-term drift) was dead during
    /// counterfactual (training) mode: the rescaled mid never actually moved.
    /// SCALE (expressing prices as % deviations from arrival, comparable across
    /// dates) is preserved; PATH (real relative movement day-over-day) is now
    /// preserved too -- these were conflated before and are separated here.
    price_factor: f64,
}

impl CounterfactualLob {
    /// Read access to the (possibly calibrated) config in force for this
    /// instance. Exists so callers outside this module -- e.g. `rl_env.rs`'s
    /// `--print-calibration` -- can read back the per-date fitted
    /// impact_coefficient/sigma/adv_btc without duplicating the calibration
    /// logic that produced them (see `CounterfactualLobConfig::calibrated`).
    pub fn config(&self) -> &CounterfactualLobConfig {
        &self.config
    }

    pub fn new(mut replay: LobReplay, config: CounterfactualLobConfig) -> Self {
        // Peek at first snapshot, compute depth_ref in simulation units, then rewind.
        let depth_ref = {
            let first = replay.next_snapshot();
            replay.reset();
            first
                .map(|b| {
                    let real_mid = b.mid_price().unwrap_or(100.0);
                    let price_factor = 100.0 / real_mid;
                    let qty_factor = 1.0 / config.btc_scale;
                    b.rescale_full(price_factor, qty_factor).total_bid_depth()
                })
                .unwrap_or(1_000_000.0)
        };

        // Load first snapshot as current book (cursor now at 1).
        let current_book = replay.next_snapshot();
        let simulation_mid_price = 100.0;
        let price_factor = Self::price_factor_for(&current_book, simulation_mid_price);

        Self {
            replay,
            config,
            sold_btc: 0.0,
            temporary_impacts: Vec::new(),
            current_step: 0,
            depth_ref,
            current_book,
            simulation_mid_price,
            price_factor,
        }
    }

    /// Compute the arrival-anchored price_factor from a book's raw mid. Shared by
    /// new()/reset()/seek() so every episode fixes its own arrival price once.
    fn price_factor_for(book: &Option<OrderBook>, simulation_mid_price: f64) -> f64 {
        let real_mid = book
            .as_ref()
            .and_then(|b| b.mid_price())
            .unwrap_or(simulation_mid_price);
        simulation_mid_price / real_mid.max(1e-9)
    }

    /// Adjusted book at the current step: the real snapshot rescaled to
    /// simulation coordinates, then shifted down by our accumulated impact.
    pub fn current_adjusted_book(&self) -> OrderBook {
        let raw_book = match self.current_book.clone() {
            Some(b) => b,
            None => return OrderBook::new(vec![], vec![]),
        };
        let scaled_book = self.rescale_book(raw_book);
        let residual = self.total_residual_impact();
        self.apply_impact_to_book(scaled_book, residual)
    }

    /// Record that `trade_qty_units` simulation units were sold, then advance snapshot.
    pub fn record_trade_and_advance(&mut self, trade_qty_units: f64) {
        let trade_btc = trade_qty_units.max(0.0) * self.config.btc_scale;
        if trade_btc > 0.0 {
            self.sold_btc += trade_btc;
            let t = self.current_step as f64 * self.config.dt;
            self.temporary_impacts.push((t, trade_btc));
        }
        // Drop sales whose temporary weight has decayed to nothing.
        let now = self.current_step as f64 * self.config.dt;
        let rho = self.config.rho;
        self.temporary_impacts
            .retain(|(t, _)| (-rho * (now - t)).exp() > 1e-12);

        self.current_book = self.replay.next_snapshot();
        self.current_step += 1;
    }

    /// Advance to the next snapshot without recording a trade.
    pub fn advance_without_trade(&mut self) {
        self.current_book = self.replay.next_snapshot();
        self.current_step += 1;
    }

    /// True once the replay is exhausted (no more snapshots).
    pub fn is_exhausted(&self) -> bool {
        self.current_book.is_none()
    }

    /// Reset to the beginning of the replay, clearing all accumulated impact.
    pub fn reset(&mut self) {
        self.replay.reset();
        self.sold_btc = 0.0;
        self.temporary_impacts.clear();
        self.current_step = 0;
        self.current_book = self.replay.next_snapshot();
        // Re-anchor price_factor to THIS episode's arrival snapshot (Phase 2R
        // Task 3) so real drift over the episode is preserved relative to its
        // own arrival, not silently flattened every step.
        self.price_factor = Self::price_factor_for(&self.current_book, self.simulation_mid_price);
        debug_assert_eq!(self.sold_btc, 0.0, "reset() must clear sold_btc");
        debug_assert!(
            self.temporary_impacts.is_empty(),
            "reset() must clear temporary_impacts"
        );
        debug_assert_eq!(self.current_step, 0, "reset() must zero current_step");
    }

    /// Seek to a specific snapshot offset, clearing all accumulated impact.
    /// Clamps to the last valid index. Sets current_book to the snapshot at offset.
    pub fn seek(&mut self, offset: usize) {
        self.replay.seek(offset);
        self.sold_btc = 0.0;
        self.temporary_impacts.clear();
        self.current_step = 0;
        self.current_book = self.replay.next_snapshot();
        // Re-anchor price_factor to THIS episode's arrival snapshot (Phase 2R
        // Task 3) -- each episode's own start_offset is its own arrival point.
        self.price_factor = Self::price_factor_for(&self.current_book, self.simulation_mid_price);
        debug_assert_eq!(self.sold_btc, 0.0, "seek() must clear sold_btc");
        debug_assert!(
            self.temporary_impacts.is_empty(),
            "seek() must clear temporary_impacts"
        );
        debug_assert_eq!(self.current_step, 0, "seek() must zero current_step");
    }

    /// Update the sim-units → BTC scale factor used by the impact model.
    ///
    /// When the per-episode inventory varies (so `btc_scale = btc_target / inventory`
    /// changes), call this after `reset()` / `seek()` so the next episode's impact
    /// calculations use the correct conversion.
    pub fn set_btc_scale(&mut self, btc_scale: f64) {
        self.config.btc_scale = btc_scale.max(1e-12);
    }

    /// Total number of snapshots in the underlying replay.
    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    /// Price impact (fraction of price) of an effective sold quantity in BTC.
    fn impact_of(&self, btc: f64) -> f64 {
        if btc <= 0.0 {
            return 0.0;
        }
        let c = &self.config;
        (c.impact_coefficient * c.sigma * (btc / c.adv_btc.max(1e-9)).sqrt()).min(0.95)
    }

    /// Effective sold quantity D_t: the permanent share of everything sold
    /// plus the decayed temporary share of each sale.
    fn effective_sold_btc(&self) -> f64 {
        let now = self.current_step as f64 * self.config.dt;
        let rho = self.config.rho;
        let decayed: f64 = self
            .temporary_impacts
            .iter()
            .map(|(t, btc)| btc * (-rho * (now - t)).exp())
            .sum();
        self.config.permanent_fraction * self.sold_btc + self.config.temporary_fraction * decayed
    }

    /// Current price impact I_t (fraction of price, at most 0.95).
    pub fn total_residual_impact(&self) -> f64 {
        self.impact_of(self.effective_sold_btc())
    }

    /// The part of the impact that never recovers.
    pub fn permanent_impact(&self) -> f64 {
        self.impact_of(self.config.permanent_fraction * self.sold_btc)
    }

    /// The part of the impact that will still recover.
    pub fn temporary_impact_total(&self) -> f64 {
        (self.total_residual_impact() - self.permanent_impact()).max(0.0)
    }

    /// Total bid depth of the current snapshot in simulation units (pre-impact).
    pub fn real_total_bid_depth(&self) -> f64 {
        match &self.current_book {
            Some(b) => {
                let qty_factor = 1.0 / self.config.btc_scale;
                b.bid_levels().iter().map(|l| l.quantity * qty_factor).sum()
            }
            None => 0.0,
        }
    }

    // ── Internals ────────────────────────────────────────────────────────────

    /// Convert a raw Binance snapshot (real BTC prices/quantities) to simulation
    /// coordinates: prices → scaled relative to the EPISODE'S ARRIVAL price
    /// (fixed once in price_factor, not recomputed per snapshot -- see the
    /// price_factor field doc), quantities → simulation units. Using the fixed
    /// factor means a snapshot whose real price has moved away from arrival
    /// rescales to a mid away from simulation_mid_price (100.0) by the same
    /// proportion -- real drift survives the rescale instead of being
    /// renormalised back to 100 every step.
    fn rescale_book(&self, raw_book: OrderBook) -> OrderBook {
        let qty_factor = 1.0 / self.config.btc_scale;
        raw_book.rescale_full(self.price_factor, qty_factor)
    }

    /// Shift every price down by the impact; depth stays as recorded.
    fn apply_impact_to_book(&self, book: OrderBook, impact_fraction: f64) -> OrderBook {
        if impact_fraction <= 0.0 {
            return book;
        }
        book.rescale_full(1.0 - impact_fraction, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::lob_replay::tests::{write_fixture, FIXTURE_CSV};

    /// Small volume makes impact numerically visible at unit-test trade sizes.
    /// With V=20, Y=1, sigma=0.2:
    ///   - 1 000 units × 0.0005 = 0.5 BTC  → impact ≈ 3.2% (< 5%)
    ///   - 200 000 units × 0.0005 = 100 BTC → impact ≈ 45% (> 15%)
    fn test_config() -> CounterfactualLobConfig {
        CounterfactualLobConfig {
            adv_btc: 20.0,
            impact_coefficient: 1.0,
            sigma: 0.2,
            btc_scale: 0.0005, // pin so test numeric thresholds stay valid
            ..Default::default()
        }
    }

    fn make_replay() -> LobReplay {
        let path = write_fixture(FIXTURE_CSV);
        LobReplay::from_csv(&path).expect("fixture must load")
    }

    fn make_cf_lob() -> CounterfactualLob {
        CounterfactualLob::new(make_replay(), test_config())
    }

    // ── price_factor drift restoration (Phase 2R Task 3) ────────────────────

    #[test]
    fn price_factor_is_fixed_at_episode_arrival_not_recomputed_per_snapshot() {
        let mut cf = make_cf_lob();
        let factor_at_start = cf.price_factor;
        for _ in 0..3 {
            cf.advance_without_trade();
        }
        assert_eq!(
            cf.price_factor, factor_at_start,
            "price_factor must stay fixed for the whole episode, not be recomputed per snapshot"
        );
    }

    /// A fixture with genuine mid movement across rows (FIXTURE_CSV's mid is a
    /// constant 100.0 at every row by construction -- only spread width varies --
    /// so it can't exercise drift-restoration behavior; this one can).
    const DRIFTING_FIXTURE_CSV: &str = "\
timestamp,b0_p,b0_q,b1_p,b1_q,a0_p,a0_q,a1_p,a1_q
1000,99.9,10.0,99.8,20.0,100.1,15.0,100.2,25.0
1001,100.9,12.0,100.8,18.0,101.1,14.0,101.2,22.0
1002,101.9,8.0,101.8,16.0,102.1,10.0,102.2,20.0
1003,98.9,11.0,98.8,19.0,99.1,13.0,99.2,21.0
1004,97.9,9.0,97.8,17.0,98.1,12.0,98.2,23.0
";

    fn make_drifting_cf_lob() -> CounterfactualLob {
        let path = write_fixture(DRIFTING_FIXTURE_CSV);
        let replay = LobReplay::from_csv(&path).expect("fixture must load");
        CounterfactualLob::new(replay, test_config())
    }

    #[test]
    fn rescaled_mid_reflects_real_drift_not_pinned_to_100() {
        // Row 1 mid=100.0 (arrival); row 2 mid=101.0 (+1%); row 3 mid=102.0 (+2%).
        // With the OLD bug, current_adjusted_book().mid_price() would be
        // renormalised back to ~100.0 at every step regardless of this real
        // movement. With the fix, it must move away from 100.0 by the same
        // proportion.
        let mut cf = make_drifting_cf_lob();
        let arrival_mid = cf.current_adjusted_book().mid_price().unwrap();
        assert!(
            (arrival_mid - 100.0).abs() < 1e-6,
            "arrival step should rescale to ~100.0"
        );

        cf.advance_without_trade(); // now on row 2, real mid 101.0 (+1% vs arrival)
        let later_mid = cf.current_adjusted_book().mid_price().unwrap();
        assert!(
            (later_mid - 101.0).abs() < 1e-6,
            "rescaled mid must move to ~101.0 (the real +1% drift from arrival), \
             not stay pinned at 100.0 (got {later_mid})"
        );
    }

    #[test]
    fn seek_and_reset_each_re_anchor_price_factor_to_their_own_arrival() {
        // Each episode (each seek()/reset()) has its OWN arrival snapshot, so
        // price_factor must be recomputed at that point -- fixed ONCE PER EPISODE,
        // not fixed forever across the whole file.
        let mut cf = make_drifting_cf_lob();
        let factor_at_offset_0 = cf.price_factor; // arrival mid 100.0 -> factor 1.0

        cf.seek(2); // row 3, real mid 102.0 -> factor should be 100/102, not 1.0
        let factor_at_offset_2 = cf.price_factor;

        assert_ne!(
            factor_at_offset_0, factor_at_offset_2,
            "seeking to a different start offset must re-anchor price_factor to \
             THAT row's real mid, not keep reusing the first episode's factor"
        );

        // And after seeking, the newly-anchored factor still rescales ITS OWN
        // arrival (row 3) back to 100.0, per-episode as designed.
        let arrival_mid_after_seek = cf.current_adjusted_book().mid_price().unwrap();
        assert!((arrival_mid_after_seek - 100.0).abs() < 1e-6);
    }

    #[test]
    fn no_trade_means_no_impact() {
        let mut cf = make_cf_lob();
        for _ in 0..10 {
            cf.advance_without_trade();
        }
        assert_eq!(cf.total_residual_impact(), 0.0, "no trades → zero impact");
    }

    #[test]
    fn small_trade_causes_small_impact() {
        let mut cf = make_cf_lob();
        cf.record_trade_and_advance(1_000.0); // 0.5 BTC
        let impact = cf.total_residual_impact();
        assert!(impact > 0.0, "impact must be positive, got {impact}");
        assert!(
            impact < 0.05,
            "small trade impact must be < 5%, got {impact}"
        );
    }

    #[test]
    fn large_trade_causes_large_impact() {
        let mut cf = make_cf_lob();
        cf.record_trade_and_advance(200_000.0); // 100 BTC
        let impact = cf.total_residual_impact();
        assert!(
            impact > 0.15,
            "large trade impact must be > 15%, got {impact}"
        );
    }

    #[test]
    fn temporary_impact_decays_over_time() {
        let mut cf = make_cf_lob();
        cf.record_trade_and_advance(100_000.0); // 50 BTC
        let impact_step1 = cf.total_residual_impact();

        for _ in 0..499 {
            cf.advance_without_trade();
        }
        let impact_step500 = cf.total_residual_impact();

        assert!(
            impact_step500 < impact_step1,
            "impact must decay: step1={impact_step1:.6}, step500={impact_step500:.6}"
        );
    }

    #[test]
    fn permanent_impact_does_not_decay() {
        let mut cf = make_cf_lob();
        cf.record_trade_and_advance(100_000.0);
        let perm_initial = cf.permanent_impact();

        for _ in 0..2000 {
            cf.advance_without_trade();
        }
        assert!(
            (cf.permanent_impact() - perm_initial).abs() < 1e-12,
            "permanent impact must not decay: initial={perm_initial}, \
             after 2000 steps={}",
            cf.permanent_impact()
        );
    }

    #[test]
    fn cumulative_trades_increase_impact() {
        // Single 50k trade
        let mut cf1 = make_cf_lob();
        cf1.record_trade_and_advance(50_000.0);
        let impact_one = cf1.total_residual_impact();

        // Two 50k trades
        let mut cf2 = make_cf_lob();
        cf2.record_trade_and_advance(50_000.0);
        cf2.record_trade_and_advance(50_000.0);
        let impact_two = cf2.total_residual_impact();

        assert!(
            impact_two > impact_one,
            "two trades must cause more impact: one={impact_one:.6}, two={impact_two:.6}"
        );
    }

    // ── CounterfactualLobConfig::calibrated ─────────────────────────────────

    fn agg_with(quantities: &[f64]) -> crate::market::agg_trades::AggTradesDay {
        use crate::market::agg_trades::{AggTrade, AggTradesDay};
        AggTradesDay {
            trades: quantities
                .iter()
                .enumerate()
                .map(|(i, &q)| AggTrade {
                    timestamp_ms: 1_000 + i as u64,
                    price: 100.0,
                    quantity_btc: q,
                    is_sell_aggressor: i % 2 == 0,
                })
                .collect(),
            date: "2024-01-15".to_string(),
            skipped_rows: 0,
        }
    }

    #[test]
    fn calibrated_config_takes_the_dates_volume_and_volatility_but_not_y() {
        let mids = vec![100.0, 101.0, 100.5, 101.5];
        let agg = agg_with(&[2.0, 3.0, 5.0]);
        let c = CounterfactualLobConfig::calibrated(
            CounterfactualLobConfig::default(),
            Some(&agg),
            &mids,
        );
        assert_eq!(c.adv_btc, 10.0, "V must be the date's traded volume");
        assert!((c.sigma - daily_volatility(&mids)).abs() < 1e-15);
        assert!(c.sigma > 0.0);
        assert_eq!(
            c.impact_coefficient, CALIBRATED_IMPACT_Y,
            "Y is frozen, not per date"
        );
    }

    #[test]
    fn calibrated_config_falls_back_to_base_without_trades_or_movement() {
        let base = CounterfactualLobConfig::default();
        let c = CounterfactualLobConfig::calibrated(
            CounterfactualLobConfig::default(),
            None,
            &[100.0, 100.0, 100.0],
        );
        assert_eq!(c.adv_btc, base.adv_btc);
        assert_eq!(c.sigma, base.sigma);
    }

    #[test]
    fn daily_volatility_scales_per_step_volatility_by_sqrt_steps() {
        // Returns +u, -u, +u, -u (u = ln 1.01): mean 0, sample stdev u*sqrt(4/3),
        // times sqrt(4 returns).
        let mids = vec![100.0, 101.0, 100.0, 101.0, 100.0];
        let u = 1.01f64.ln();
        let expected = u * (4.0f64 / 3.0).sqrt() * 2.0;
        assert!((daily_volatility(&mids) - expected).abs() < 1e-12);
    }

    #[test]
    fn impact_is_concave_in_the_amount_sold() {
        // Square-root law on the accumulated quantity: 4x the size -> 2x impact.
        let mut a = make_cf_lob();
        a.record_trade_and_advance(10_000.0);
        let mut b = make_cf_lob();
        b.record_trade_and_advance(40_000.0);
        let ratio = b.total_residual_impact() / a.total_residual_impact();
        assert!((ratio - 2.0).abs() < 1e-3, "ratio {ratio}");
    }

    #[test]
    fn slicing_an_order_does_not_multiply_its_impact() {
        // Selling 40k units in 40 back-to-back slices must cost about the same
        // impact as one 40k sale (the temporary part decays a little between
        // slices), never ~sqrt(40) times more.
        let mut one = make_cf_lob();
        one.record_trade_and_advance(40_000.0);
        let mut sliced = make_cf_lob();
        for _ in 0..40 {
            sliced.record_trade_and_advance(1_000.0);
        }
        let (i1, i40) = (one.total_residual_impact(), sliced.total_residual_impact());
        assert!(
            i40 <= i1 * 1.0001 && i40 > 0.95 * i1,
            "one={i1} sliced={i40}"
        );
    }

    // ── Sanity checks requested for the Phase 2R Task 2 audit ──────────────

    #[test]
    fn impact_makes_a_sell_fill_worse_not_better() {
        // "Impact is positive and in the correct direction for a sell (pushes
        // price down)." Compare the average fill price of an IDENTICAL sell on
        // a fresh book vs. one that already has accumulated impact -- the
        // impacted fill must be at or below the unimpacted one, never above.
        let fresh = make_cf_lob();
        let unimpacted_fill = fresh
            .current_adjusted_book()
            .fill_market_sell(500.0)
            .avg_fill_price;

        let mut impacted = make_cf_lob();
        impacted.record_trade_and_advance(200_000.0); // large trade -> real impact
        let impacted_fill = impacted
            .current_adjusted_book()
            .fill_market_sell(500.0)
            .avg_fill_price;

        assert!(
            impacted_fill <= unimpacted_fill,
            "a sell against an impacted (price-shifted) book must fill at or below the \
             unimpacted price, never above: unimpacted={unimpacted_fill:.4}, impacted={impacted_fill:.4}"
        );
    }

    #[test]
    fn impact_decays_monotonically_toward_permanent_level() {
        // "Impact decays monotonically toward the permanent level over time."
        // rho=5, dt=1/2880 -> half-life = ln(2)/rho/dt ~= 400 steps, so
        // convergence needs several thousand steps, not a few hundred.
        let mut cf = make_cf_lob();
        cf.record_trade_and_advance(150_000.0);
        let perm = cf.permanent_impact();

        let mut samples = Vec::new();
        for _ in 0..15 {
            for _ in 0..2000 {
                cf.advance_without_trade();
            }
            samples.push(cf.total_residual_impact());
        }
        for w in samples.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "residual impact must decay monotonically: {w:?}"
            );
        }
        assert!(
            (samples.last().unwrap() - perm).abs() < 1e-3,
            "after many steps, residual impact must converge to the permanent level: \
             final={:.6}, permanent={:.6}",
            samples.last().unwrap(),
            perm
        );
    }

    #[test]
    fn temporary_component_vanishes_as_decay_time_grows() {
        // "Temporary component -> 0 as decay time -> infinity." Half-life ~400
        // steps (see above), so 30000 steps (~75 half-lives) drives it to
        // numerical zero with comfortable margin.
        let mut cf = make_cf_lob();
        cf.record_trade_and_advance(150_000.0);
        for _ in 0..30_000 {
            cf.advance_without_trade();
        }
        assert!(
            cf.temporary_impact_total() < 1e-8,
            "temporary impact must vanish after enough decay time, got {}",
            cf.temporary_impact_total()
        );
    }

    #[test]
    fn impact_never_improves_a_sell_price_no_free_lunch() {
        // "No-arbitrage: a constructed round-trip trade cannot be profitable."
        // NOTE: this simulator's impact model is sell-only (record_trade_and_advance
        // only ever pushes prices down; there is no buy-side mechanism), so a
        // literal sell-then-buy-back round trip is
        // not constructible with the current API -- there is nothing to buy back
        // INTO. The meaningful, testable form of "no free lunch" here is: a LARGER
        // self-trade must never leave a subsequent sell better off than a SMALLER
        // (or zero) self-trade, holding everything else fixed.
        //
        // Isolating "impact only" requires holding the underlying real snapshot
        // FIXED, but record_trade_and_advance always advances to the next real
        // snapshot as a side effect (real snapshot-to-snapshot price movement is
        // a separate, legitimate effect this test must not conflate with impact).
        // So: use independent fresh instances that each advance EXACTLY ONCE
        // (one record_trade_and_advance call) -- all land on the SAME next real
        // snapshot deterministically, so the only difference between them is the
        // impact from their one trade's size.
        let probe = |qty: f64| -> f64 {
            let mut cf = make_cf_lob();
            cf.record_trade_and_advance(qty);
            cf.current_adjusted_book()
                .fill_market_sell(200.0)
                .avg_fill_price
        };

        let no_trade = probe(0.0);
        let small = probe(20_000.0);
        let medium = probe(80_000.0);
        let large = probe(150_000.0);

        assert!(
            no_trade >= small - 1e-9 && small >= medium - 1e-9 && medium >= large - 1e-9,
            "a larger self-trade must never leave a subsequent sell better off: \
             no_trade={no_trade:.6} small={small:.6} medium={medium:.6} large={large:.6}"
        );
    }

    #[test]
    fn impact_capped_at_95_percent() {
        let mut cf = make_cf_lob();
        for _ in 0..20 {
            cf.record_trade_and_advance(1_000_000.0);
            let impact = cf.total_residual_impact();
            assert!(impact <= 0.95, "impact must not exceed 0.95, got {impact}");
        }
    }

    /// WHAT: After running episodes that accumulate impact, reset() must restore
    ///       the LOB to a pristine state — no permanent reduction, no temporary
    ///       impacts queued, and the first fill matches the un-impacted mid price.
    /// WHY: A known prior bug allowed permanent impact to bleed across
    ///      episodes, silently degrading every subsequent fill price. This test
    ///      pins reset() against that regression.
    #[test]
    fn counterfactual_reset_guarantees_fresh_start() {
        let mut cf = make_cf_lob();

        // Accumulate impact across 50 large-sell steps.
        for _ in 0..50 {
            cf.record_trade_and_advance(200_000.0);
        }
        assert!(
            cf.sold_btc > 0.0,
            "preconditions: impact should have accumulated after 50 large trades"
        );

        // Capture the un-impacted reference fill price BEFORE reset so the test
        // is independent of fixture-snapshot ordering quirks.
        let fresh = make_cf_lob();
        let baseline_fill = fresh
            .current_adjusted_book()
            .fill_market_sell(100.0)
            .avg_fill_price;

        // Reset and re-verify the canonical preconditions hold.
        cf.reset();
        assert_eq!(cf.sold_btc, 0.0, "after reset, sold_btc must be 0");
        assert!(
            cf.temporary_impacts.is_empty(),
            "after reset, temporary_impacts must be empty"
        );
        assert_eq!(cf.current_step, 0, "after reset, current_step must be 0");

        // A fresh fill on the post-reset book must match the baseline fill from a
        // never-used CounterfactualLob (within 1% of the reference mid).
        let reset_fill = cf
            .current_adjusted_book()
            .fill_market_sell(100.0)
            .avg_fill_price;
        let ref_mid = cf.current_adjusted_book().mid_price().unwrap_or(100.0);
        let pct_diff = ((reset_fill - baseline_fill).abs() / ref_mid.max(1e-9)) * 100.0;
        assert!(
            pct_diff < 1.0,
            "post-reset fill price {reset_fill:.4} should match baseline {baseline_fill:.4} \
             within 1% of mid {ref_mid:.4} (diff {pct_diff:.4}%)"
        );
    }

    #[test]
    fn impact_shifts_prices_down_and_leaves_depth_as_recorded() {
        let mut cf = make_cf_lob();
        let mut free = make_cf_lob();
        cf.record_trade_and_advance(200_000.0); // 100 BTC -- large impact
        free.advance_without_trade(); // same real snapshot, no impact
        let (hit, clean) = (cf.current_adjusted_book(), free.current_adjusted_book());
        let i = cf.total_residual_impact();
        assert!(i > 0.1);
        assert!((hit.mid_price().unwrap() - clean.mid_price().unwrap() * (1.0 - i)).abs() < 1e-9);
        assert!((hit.total_bid_depth() - clean.total_bid_depth()).abs() < 1e-9);
    }
}
