// CounterfactualLob — real Binance LOB snapshots adjusted for your own market impact.
//
// Glues together LobReplay (real snapshots) + impact model (Obizhaeva-Wang 2013):
//   - Permanent fraction: never recovers (30%)
//   - Temporary fraction: exponential decay (70%, half-life ~3.3 hours at rho=5)
//
// LobReplay is owned and advanced here; do not advance it externally.

use crate::market::lob_replay::LobReplay;
use crate::market::order_book::OrderBook;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct CounterfactualLobConfig {
    /// Fraction of impact that is permanent (Obizhaeva-Wang: 0.3).
    pub permanent_fraction: f64,
    /// Fraction of impact that is temporary (= 1 - permanent_fraction).
    pub temporary_fraction: f64,
    /// Recovery speed for temporary impact. rho=5 with dt=1/2880 → half-life ≈ 400 steps.
    pub rho: f64,
    /// Time per step (1.0 / total_steps).
    pub dt: f64,
    /// Simulation units → BTC conversion (e.g. 50_000 BTC / 1_000_000 units = 0.05).
    pub btc_scale: f64,
    /// Average daily volume in BTC. Controls participation-rate denominator.
    pub adv_btc: f64,
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
        }
    }
}

// ---------------------------------------------------------------------------
// CounterfactualLob
// ---------------------------------------------------------------------------

pub struct CounterfactualLob {
    replay: LobReplay,
    config: CounterfactualLobConfig,

    /// Accumulated permanent depth reduction (never decays), capped at 0.95.
    pub(crate) permanent_depth_reduction: f64,

    /// (time_when_traded, impact_fraction) — used to compute decayed residual.
    temporary_impacts: Vec<(f64, f64)>,

    current_step: usize,

    /// Total bid depth of the first snapshot — calibration reference.
    pub depth_ref: f64,

    /// The current Binance snapshot (pre-impact).
    current_book: Option<OrderBook>,

    /// Fixed simulation mid price (100.0). Real BTC prices are normalised to this.
    simulation_mid_price: f64,
}

impl CounterfactualLob {
    pub fn new(mut replay: LobReplay, config: CounterfactualLobConfig) -> Self {
        // Peek at first snapshot, compute depth_ref in simulation units, then rewind.
        let depth_ref = {
            let first = replay.next_snapshot();
            replay.reset();
            first.map(|b| {
                let real_mid = b.mid_price().unwrap_or(100.0);
                let price_factor = 100.0 / real_mid;
                let qty_factor = 1.0 / config.btc_scale;
                b.rescale_full(price_factor, qty_factor).total_bid_depth()
            }).unwrap_or(1_000_000.0)
        };

        // Load first snapshot as current book (cursor now at 1).
        let current_book = replay.next_snapshot();

        Self {
            replay,
            config,
            permanent_depth_reduction: 0.0,
            temporary_impacts: Vec::new(),
            current_step: 0,
            depth_ref,
            current_book,
            simulation_mid_price: 100.0,
        }
    }

    /// Adjusted book at the current step: real Binance snapshot rescaled to
    /// simulation coordinates, then depth-reduced by accumulated impact.
    pub fn current_adjusted_book(&self) -> OrderBook {
        let raw_book = match self.current_book.clone() {
            Some(b) => b,
            None => return OrderBook::new(vec![], vec![]),
        };
        let scaled_book = self.rescale_book(raw_book);
        let residual = self.total_residual_impact();
        self.apply_impact_to_book(scaled_book, residual)
    }

    /// Record that `trade_qty_units` simulation units were traded, then advance snapshot.
    pub fn record_trade_and_advance(&mut self, trade_qty_units: f64) {
        let trade_btc = trade_qty_units * self.config.btc_scale;
        let participation = trade_btc / self.config.adv_btc;
        // Square-root impact: sqrt(participation) * calibration_constant
        let impact_fraction = participation.sqrt() * 0.1;

        let perm = impact_fraction * self.config.permanent_fraction;
        let temp = impact_fraction * self.config.temporary_fraction;

        self.permanent_depth_reduction =
            (self.permanent_depth_reduction + perm).min(0.95);

        if temp > 1e-8 {
            let t = self.current_step as f64 * self.config.dt;
            self.temporary_impacts.push((t, temp));
        }

        // Prune impacts whose residual has decayed below 1e-6.
        // (1e-3 was too large for realistic ADV values where impact < 0.1%.)
        let now = self.current_step as f64 * self.config.dt;
        let rho = self.config.rho;
        self.temporary_impacts.retain(|(t, impact)| {
            impact * (-rho * (now - t)).exp() > 1e-6
        });

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
        self.permanent_depth_reduction = 0.0;
        self.temporary_impacts.clear();
        self.current_step = 0;
        self.current_book = self.replay.next_snapshot();
        debug_assert_eq!(self.permanent_depth_reduction, 0.0,
            "reset() must clear permanent_depth_reduction");
        debug_assert!(self.temporary_impacts.is_empty(),
            "reset() must clear temporary_impacts");
        debug_assert_eq!(self.current_step, 0,
            "reset() must zero current_step");
    }

    /// Seek to a specific snapshot offset, clearing all accumulated impact.
    /// Clamps to the last valid index. Sets current_book to the snapshot at offset.
    pub fn seek(&mut self, offset: usize) {
        self.replay.seek(offset);
        self.permanent_depth_reduction = 0.0;
        self.temporary_impacts.clear();
        self.current_step = 0;
        self.current_book = self.replay.next_snapshot();
        debug_assert_eq!(self.permanent_depth_reduction, 0.0,
            "seek() must clear permanent_depth_reduction");
        debug_assert!(self.temporary_impacts.is_empty(),
            "seek() must clear temporary_impacts");
        debug_assert_eq!(self.current_step, 0,
            "seek() must zero current_step");
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

    /// Total residual impact fraction at the current step (permanent + decayed temporary).
    pub fn total_residual_impact(&self) -> f64 {
        let now = self.current_step as f64 * self.config.dt;
        let rho = self.config.rho;
        let temporary_total: f64 = self
            .temporary_impacts
            .iter()
            .map(|(t, impact)| impact * (-rho * (now - t)).exp())
            .sum();
        (self.permanent_depth_reduction + temporary_total).min(0.95)
    }

    /// Permanent impact fraction (never decays).
    pub fn permanent_impact(&self) -> f64 {
        self.permanent_depth_reduction
    }

    /// Temporary-only residual at the current step.
    pub fn temporary_impact_total(&self) -> f64 {
        let now = self.current_step as f64 * self.config.dt;
        let rho = self.config.rho;
        self.temporary_impacts
            .iter()
            .map(|(t, impact)| impact * (-rho * (now - t)).exp())
            .sum()
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
    /// coordinates: prices → ~100.0, quantities → simulation units.
    fn rescale_book(&self, raw_book: OrderBook) -> OrderBook {
        let real_mid = raw_book.mid_price().unwrap_or(self.simulation_mid_price);
        let price_factor = self.simulation_mid_price / real_mid;
        let qty_factor = 1.0 / self.config.btc_scale;
        raw_book.rescale_full(price_factor, qty_factor)
    }

    fn apply_impact_to_book(&self, mut book: OrderBook, impact_fraction: f64) -> OrderBook {
        if impact_fraction < 1e-6 {
            return book;
        }
        book.scale_bid_depth(1.0 - impact_fraction);
        book
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::lob_replay::tests::{write_fixture, FIXTURE_CSV};

    /// Small ADV makes impact numerically visible at unit-test trade sizes.
    /// With adv=20 and constant=0.1:
    ///   - 1 000 units × 0.0005 = 0.5 BTC  → impact ≈ 1.6% (< 5%)
    ///   - 200 000 units × 0.0005 = 100 BTC → impact ≈ 22%  (> 15%)
    fn test_config() -> CounterfactualLobConfig {
        CounterfactualLobConfig {
            adv_btc: 20.0,
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

    #[test]
    fn no_trade_means_no_impact() {
        let mut cf = make_cf_lob();
        for _ in 0..10 {
            cf.advance_without_trade();
        }
        assert_eq!(
            cf.total_residual_impact(),
            0.0,
            "no trades → zero impact"
        );
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
        let perm_initial = cf.permanent_depth_reduction;

        for _ in 0..2000 {
            cf.advance_without_trade();
        }
        assert!(
            (cf.permanent_depth_reduction - perm_initial).abs() < 1e-12,
            "permanent impact must not decay: initial={perm_initial}, \
             after 2000 steps={}",
            cf.permanent_depth_reduction
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

    #[test]
    fn impact_capped_at_95_percent() {
        let mut cf = make_cf_lob();
        for _ in 0..20 {
            cf.record_trade_and_advance(1_000_000.0);
            let impact = cf.total_residual_impact();
            assert!(
                impact <= 0.95,
                "impact must not exceed 0.95, got {impact}"
            );
        }
    }

    /// WHAT: After running episodes that accumulate impact, reset() must restore
    ///       the LOB to a pristine state — no permanent reduction, no temporary
    ///       impacts queued, and the first fill matches the un-impacted mid price.
    /// WHY: A known prior bug allowed permanent_depth_reduction to bleed across
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
            cf.permanent_depth_reduction > 0.0,
            "preconditions: impact should have accumulated after 50 large trades"
        );

        // Capture the un-impacted reference fill price BEFORE reset so the test
        // is independent of fixture-snapshot ordering quirks.
        let mut fresh = make_cf_lob();
        let baseline_fill = fresh.current_adjusted_book().fill_market_sell(100.0).avg_fill_price;

        // Reset and re-verify the canonical preconditions hold.
        cf.reset();
        assert_eq!(
            cf.permanent_depth_reduction, 0.0,
            "after reset, permanent_depth_reduction must be 0"
        );
        assert!(
            cf.temporary_impacts.is_empty(),
            "after reset, temporary_impacts must be empty"
        );
        assert_eq!(cf.current_step, 0, "after reset, current_step must be 0");

        // A fresh fill on the post-reset book must match the baseline fill from a
        // never-used CounterfactualLob (within 1% of the reference mid).
        let reset_fill = cf.current_adjusted_book().fill_market_sell(100.0).avg_fill_price;
        let ref_mid = cf.current_adjusted_book().mid_price().unwrap_or(100.0);
        let pct_diff = ((reset_fill - baseline_fill).abs() / ref_mid.max(1e-9)) * 100.0;
        assert!(
            pct_diff < 1.0,
            "post-reset fill price {reset_fill:.4} should match baseline {baseline_fill:.4} \
             within 1% of mid {ref_mid:.4} (diff {pct_diff:.4}%)"
        );
    }

    #[test]
    fn adjusted_book_has_less_depth_than_real_book() {
        let mut cf = make_cf_lob();
        cf.record_trade_and_advance(200_000.0); // 100 BTC — large impact
        let real_depth = cf.real_total_bid_depth();
        let adjusted_depth = cf.current_adjusted_book().total_bid_depth();
        assert!(
            adjusted_depth < real_depth,
            "adjusted book depth must be less than real: \
             real={real_depth:.4}, adjusted={adjusted_depth:.4}"
        );
    }
}
