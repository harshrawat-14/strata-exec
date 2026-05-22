// AggTrades reader — Binance Vision aggTrades CSV format.
//
// Column layout (confirmed from diagnostic):
//   agg_trade_id, price, quantity, first_trade_id, last_trade_id,
//   transact_time, is_buyer_maker
//
// is_buyer_maker == "true"  → sell-aggressor (maker is the buyer, taker sold)
// is_buyer_maker == "false" → buy-aggressor  (taker bought)
// quantity is in BTC units.

use crate::market::vpin::calibrate_bucket_size;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub struct AggTrade {
    pub timestamp_ms: u64,
    pub price: f64,
    pub quantity_btc: f64,
    /// true when is_buyer_maker == "true" (sell-aggressor)
    pub is_sell_aggressor: bool,
}

pub struct AggTradesDay {
    pub trades: Vec<AggTrade>,
    /// Date string parsed from the file, e.g. "2024-01-15".
    pub date: String,
    pub skipped_rows: usize,
}

impl AggTradesDay {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    pub fn from_csv(path: &str) -> Result<Self, String> {
        let mut reader = csv::Reader::from_path(path)
            .map_err(|e| format!("cannot open '{}': {e}", path))?;

        // Locate required columns by name — robust to column reordering.
        let headers = reader
            .headers()
            .map_err(|e| format!("cannot read headers: {e}"))?
            .clone();

        let col = |name: &str| -> Result<usize, String> {
            headers
                .iter()
                .position(|h| h.trim() == name)
                .ok_or_else(|| format!("missing column '{name}'"))
        };

        let ts_col  = col("transact_time")?;
        let px_col  = col("price")?;
        let qty_col = col("quantity")?;
        let ibm_col = col("is_buyer_maker")?;

        let mut trades: Vec<AggTrade> = Vec::new();
        let mut skipped: usize = 0;

        for result in reader.records() {
            let rec = match result {
                Ok(r) => r,
                Err(_) => { skipped += 1; continue; }
            };

            let timestamp_ms = match rec.get(ts_col).and_then(|v| v.trim().parse::<u64>().ok()) {
                Some(v) => v,
                None => { skipped += 1; continue; }
            };
            let price = match rec.get(px_col).and_then(|v| v.trim().parse::<f64>().ok()) {
                Some(v) if v > 0.0 => v,
                _ => { skipped += 1; continue; }
            };
            let quantity_btc = match rec.get(qty_col).and_then(|v| v.trim().parse::<f64>().ok()) {
                Some(v) if v > 0.0 => v,
                _ => { skipped += 1; continue; }
            };
            let is_sell_aggressor = match rec.get(ibm_col) {
                Some(v) => v.trim().to_lowercase() == "true",
                None => { skipped += 1; continue; }
            };

            trades.push(AggTrade { timestamp_ms, price, quantity_btc, is_sell_aggressor });
        }

        // Derive date from the first trade's timestamp (unix ms → UTC date).
        let date = if let Some(first) = trades.first() {
            let secs = (first.timestamp_ms / 1000) as i64;
            // Format: "YYYY-MM-DD" using integer arithmetic (no extra crate needed).
            // chrono is already a dependency from lob_replay.
            use chrono::TimeZone;
            let dt = chrono::Utc.timestamp_opt(secs, 0).single()
                .unwrap_or_else(chrono::Utc::now);
            dt.format("%Y-%m-%d").to_string()
        } else {
            String::new()
        };

        let day = AggTradesDay { trades, date, skipped_rows: skipped };

        let total_btc = day.total_volume_btc();
        let bucket = day.calibrated_bucket_size(50);
        eprintln!(
            "AggTrades: {:>12} trades loaded, {:>12.1} BTC total volume",
            format_count(day.trades.len()),
            total_btc,
        );
        eprintln!(
            "VPIN bucket: {:>8.1} BTC (auto-calibrated, 50 buckets/day)",
            bucket,
        );

        Ok(day)
    }

    // -----------------------------------------------------------------------
    // Volume queries
    // -----------------------------------------------------------------------

    pub fn total_volume_btc(&self) -> f64 {
        self.trades.iter().map(|t| t.quantity_btc).sum()
    }

    pub fn calibrated_bucket_size(&self, target_buckets: usize) -> f64 {
        calibrate_bucket_size(self.total_volume_btc(), target_buckets)
    }

    /// Buy and sell BTC volume in the half-open window [start_ms, end_ms).
    ///
    /// Returns (buy_volume_btc, sell_volume_btc).
    /// Trades are assumed to be sorted by timestamp_ms (as they come from the CSV).
    /// Uses a linear scan; the caller is responsible for not calling this in O(n²)
    /// fashion — `build_vpin_series` advances a cursor instead.
    pub fn volume_in_window(&self, start_ms: u64, end_ms: u64) -> (f64, f64) {
        let mut buy = 0.0_f64;
        let mut sell = 0.0_f64;
        for t in &self.trades {
            if t.timestamp_ms >= start_ms && t.timestamp_ms < end_ms {
                if t.is_sell_aggressor {
                    sell += t.quantity_btc;
                } else {
                    buy += t.quantity_btc;
                }
            } else if t.timestamp_ms >= end_ms {
                break; // trades are sorted; no point scanning further
            }
        }
        (buy, sell)
    }

    // -----------------------------------------------------------------------
    // VPIN series (Step 2)
    // -----------------------------------------------------------------------

    /// Compute a VPIN value for each snapshot timestamp using real aggTrades volume.
    ///
    /// For each consecutive pair of snapshot timestamps [t_k, t_{k+1}):
    ///   - Sum buy_vol and sell_vol from aggTrades in that window.
    ///   - Feed into VpinCalculator (no tick-rule needed — direction is known).
    ///   - Record current_vpin() as the estimate for snapshot k.
    ///
    /// Returns exactly `snapshot_timestamps.len()` values.
    /// Snapshots before the first bucket completes return 0.0.
    pub fn build_vpin_series(
        &self,
        snapshot_timestamps: &[u64],
        bucket_size_btc: f64,
        window_buckets: usize,
    ) -> Vec<f64> {
        use crate::market::vpin::VpinCalculator;

        let n = snapshot_timestamps.len();
        if n == 0 {
            return Vec::new();
        }

        let mut calc = VpinCalculator::new(bucket_size_btc, window_buckets);
        let mut result = vec![0.0_f64; n];

        // Advance a single cursor through the trade list to avoid O(n²).
        let mut cursor: usize = 0;
        let trades = &self.trades;

        for k in 0..n {
            let t_start = snapshot_timestamps[k];
            let t_end = if k + 1 < n {
                snapshot_timestamps[k + 1]
            } else {
                u64::MAX // last snapshot: consume everything after it
            };

            // Accumulate buy/sell volume in [t_start, t_end).
            let mut buy_vol = 0.0_f64;
            let mut sell_vol = 0.0_f64;

            // Advance cursor past any trades before t_start (shouldn't happen for
            // monotone timestamps, but guards against gaps).
            while cursor < trades.len() && trades[cursor].timestamp_ms < t_start {
                cursor += 1;
            }
            // Accumulate trades in [t_start, t_end).
            let window_start = cursor;
            while cursor < trades.len() && trades[cursor].timestamp_ms < t_end {
                let t = &trades[cursor];
                if t.is_sell_aggressor {
                    sell_vol += t.quantity_btc;
                } else {
                    buy_vol += t.quantity_btc;
                }
                cursor += 1;
            }
            let _ = window_start; // suppress unused warning

            // Feed into VpinCalculator if there was any volume.
            // We split buy and sell into separate synthetic "trades" with a
            // fixed price direction so VpinCalculator's tick rule classifies them correctly.
            //
            // Trick: feed buy_vol at price=101 (prev=100) → uptick → buy classification.
            //        feed sell_vol at price=99  (prev=100) → downtick → sell classification.
            // This bypasses the tick rule entirely and lets us use the known aggressor side.
            if buy_vol > 1e-12 {
                calc.add_trade(101.0, buy_vol, 100.0);
            }
            if sell_vol > 1e-12 {
                calc.add_trade(99.0, sell_vol, 101.0);
            }

            result[k] = calc.current_vpin().unwrap_or(0.0);
        }

        result
    }

    // -----------------------------------------------------------------------
    // OFI series (Step 3)
    // -----------------------------------------------------------------------

    /// Compute a rolling OFI (Order Flow Imbalance) for each snapshot timestamp.
    ///
    /// For each snapshot at `t_k`, sums buy and sell BTC volume from aggTrades in
    /// the half-open window [t_k − window_ms, t_k) and returns:
    ///
    ///   OFI = (buy_vol − sell_vol) / (buy_vol + sell_vol)   ∈ [−1, 1]
    ///
    /// Returns 0.0 for windows with no trade volume.
    /// Returns exactly `snapshot_timestamps.len()` values.
    pub fn build_ofi_series(&self, snapshot_timestamps: &[u64], window_ms: u64) -> Vec<f64> {
        let n = snapshot_timestamps.len();
        if n == 0 {
            return Vec::new();
        }

        let mut result = vec![0.0_f64; n];
        let trades = &self.trades;

        for k in 0..n {
            let end_ms = snapshot_timestamps[k];
            let start_ms = end_ms.saturating_sub(window_ms);

            // Binary search for window boundaries — trades are sorted by timestamp.
            let left = trades.partition_point(|t| t.timestamp_ms < start_ms);
            let right = trades.partition_point(|t| t.timestamp_ms < end_ms);

            let mut buy = 0.0_f64;
            let mut sell = 0.0_f64;
            for t in &trades[left..right] {
                if t.is_sell_aggressor {
                    sell += t.quantity_btc;
                } else {
                    buy += t.quantity_btc;
                }
            }

            let total = buy + sell;
            result[k] = if total < 1e-12 { 0.0 } else { (buy - sell) / total };
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 7-row fixture: header + 6 data rows.
    // Columns: agg_trade_id, price, quantity, first_trade_id, last_trade_id,
    //          transact_time, is_buyer_maker
    const FIXTURE_CSV: &str = "\
agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
1,41735.0,0.100,100,100,1705276800000,false
2,41734.9,0.200,101,101,1705276810000,true
3,41735.5,0.050,102,102,1705276820000,false
4,41736.0,0.300,103,103,1705276830000,true
5,41735.0,0.150,104,104,1705276840000,false
6,41734.0,0.400,105,105,1705276850000,true
";

    // Malformed-row fixture: one row with a non-numeric quantity.
    const MALFORMED_CSV: &str = "\
agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
1,41735.0,0.100,100,100,1705276800000,false
2,41734.9,INVALID,101,101,1705276810000,true
3,41735.5,0.050,102,102,1705276820000,false
";

    fn load_fixture(csv: &str) -> AggTradesDay {
        let path = write_fixture(csv);
        AggTradesDay::from_csv(&path).expect("fixture must load")
    }

    fn write_fixture(content: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir()
            .join(format!("agg_trades_fixture_{}_{}.csv", std::process::id(), id))
            .to_str()
            .unwrap()
            .to_string();
        std::fs::write(&path, content).expect("fixture write must succeed");
        path
    }

    #[test]
    fn aggrades_loads_csv_from_fixture() {
        let day = load_fixture(FIXTURE_CSV);
        assert_eq!(day.trades.len(), 6, "must load all 6 data rows");
        assert_eq!(day.skipped_rows, 0);
    }

    #[test]
    fn sell_aggressor_when_is_buyer_maker_is_true() {
        let day = load_fixture(FIXTURE_CSV);
        // Row 2: is_buyer_maker = "true" → sell-aggressor
        assert!(
            day.trades[1].is_sell_aggressor,
            "is_buyer_maker='true' must set is_sell_aggressor=true"
        );
    }

    #[test]
    fn buy_aggressor_when_is_buyer_maker_is_false() {
        let day = load_fixture(FIXTURE_CSV);
        // Row 1: is_buyer_maker = "false" → buy-aggressor
        assert!(
            !day.trades[0].is_sell_aggressor,
            "is_buyer_maker='false' must set is_sell_aggressor=false"
        );
    }

    #[test]
    fn total_volume_sums_all_quantities() {
        let day = load_fixture(FIXTURE_CSV);
        // 0.100 + 0.200 + 0.050 + 0.300 + 0.150 + 0.400 = 1.200
        let expected = 1.200_f64;
        let got = day.total_volume_btc();
        assert!(
            (got - expected).abs() < 1e-9,
            "total_volume_btc expected {expected}, got {got}"
        );
    }

    #[test]
    fn volume_in_window_returns_correct_split() {
        let day = load_fixture(FIXTURE_CSV);
        // Window [1705276800000, 1705276830000) covers rows 1, 2, 3 (timestamps 000, 010, 020 sec).
        // Row 1: buy  0.100
        // Row 2: sell 0.200
        // Row 3: buy  0.050
        let (buy, sell) = day.volume_in_window(1_705_276_800_000, 1_705_276_830_000);
        assert!((buy - 0.150).abs() < 1e-9, "buy expected 0.150, got {buy}");
        assert!((sell - 0.200).abs() < 1e-9, "sell expected 0.200, got {sell}");
    }

    #[test]
    fn volume_in_window_excludes_trades_outside_range() {
        let day = load_fixture(FIXTURE_CSV);
        // Window [1705276860000, 1705276900000) is after all trades → both zero.
        let (buy, sell) = day.volume_in_window(1_705_276_860_000, 1_705_276_900_000);
        assert!(buy < 1e-12, "buy must be 0 outside range, got {buy}");
        assert!(sell < 1e-12, "sell must be 0 outside range, got {sell}");
    }

    #[test]
    fn malformed_rows_are_skipped_not_panicked() {
        let day = load_fixture(MALFORMED_CSV);
        // Row 2 has INVALID quantity → skipped. Rows 1 and 3 must load.
        assert_eq!(day.trades.len(), 2, "must load 2 valid rows, skip 1 malformed");
        assert_eq!(day.skipped_rows, 1, "must count 1 skipped row");
    }

    // ── build_vpin_series tests ──────────────────────────────────────────────

    /// Build a synthetic AggTradesDay from inline data (no CSV I/O).
    fn make_day(trades: Vec<AggTrade>) -> AggTradesDay {
        AggTradesDay { trades, date: "2024-01-15".to_string(), skipped_rows: 0 }
    }

    /// Generate `n` uniformly-spaced snapshot timestamps starting at `start_ms`
    /// with `interval_ms` between them.
    fn make_snapshots(start_ms: u64, interval_ms: u64, n: usize) -> Vec<u64> {
        (0..n).map(|i| start_ms + i as u64 * interval_ms).collect()
    }

    #[test]
    fn vpin_series_length_matches_snapshot_count() {
        let day = load_fixture(FIXTURE_CSV);
        let snaps = make_snapshots(1_705_276_800_000, 30_000, 10);
        let series = day.build_vpin_series(&snaps, 0.05, 5);
        assert_eq!(
            series.len(), 10,
            "series length must equal snapshot count: expected 10, got {}",
            series.len()
        );
    }

    #[test]
    fn vpin_series_values_are_bounded_zero_to_one() {
        let day = load_fixture(FIXTURE_CSV);
        let snaps = make_snapshots(1_705_276_800_000, 10_000, 10);
        let series = day.build_vpin_series(&snaps, 0.05, 5);
        for (i, &v) in series.iter().enumerate() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "VPIN at index {i} = {v} is outside [0, 1]"
            );
        }
    }

    #[test]
    fn vpin_series_has_nonzero_values_after_warmup_period() {
        // Build a day with many trades so the bucket fills quickly.
        let mut trades = Vec::new();
        for i in 0u64..200 {
            // Alternating buy/sell, 0.1 BTC each, every 100 ms.
            trades.push(AggTrade {
                timestamp_ms: 1_705_276_800_000 + i * 100,
                price: 41735.0,
                quantity_btc: 0.1,
                is_sell_aggressor: i % 2 == 0,
            });
        }
        let day = make_day(trades);
        // 200 trades × 0.1 BTC = 20 BTC total; bucket_size = 0.5 BTC → 40 buckets
        let snaps = make_snapshots(1_705_276_800_000, 1_000, 25);
        let series = day.build_vpin_series(&snaps, 0.5, 5);
        // At least some values after index 0 must be non-zero (bucket must fill).
        let any_nonzero = series.iter().any(|&v| v > 1e-12);
        assert!(any_nonzero, "no non-zero VPIN values found; bucket never filled");
    }

    #[test]
    fn high_one_sided_volume_produces_higher_vpin_than_balanced() {
        // Each snapshot window gets exactly 10 BTC of volume.
        // Balanced: 5 BTC buy + 5 BTC sell per window → imbalance per bucket = 0.
        // One-sided: 10 BTC buy + 0 BTC sell per window → imbalance per bucket = 1.
        // Bucket size = 10 BTC so each window fills exactly one bucket.
        // After filling 5 buckets, VPIN is the mean of the last 5 imbalances.

        let window_ms = 30_000_u64; // 30 seconds per snapshot
        let n_snaps = 20_usize;

        let mut balanced_trades = Vec::new();
        let mut onesided_trades = Vec::new();

        for i in 0u64..n_snaps as u64 {
            let t = 1_705_276_800_000 + i * window_ms;
            // Balanced: 5 BTC buy at t, 5 BTC sell at t+1ms
            balanced_trades.push(AggTrade { timestamp_ms: t,     price: 41735.0, quantity_btc: 5.0, is_sell_aggressor: false });
            balanced_trades.push(AggTrade { timestamp_ms: t + 1, price: 41735.0, quantity_btc: 5.0, is_sell_aggressor: true  });
            // One-sided: 10 BTC buy only
            onesided_trades.push(AggTrade { timestamp_ms: t, price: 41735.0, quantity_btc: 10.0, is_sell_aggressor: false });
        }

        let balanced = make_day(balanced_trades);
        let onesided = make_day(onesided_trades);
        let snaps = make_snapshots(1_705_276_800_000, window_ms, n_snaps);

        // bucket_size = 10 BTC, window = 5 buckets
        let b_series = balanced.build_vpin_series(&snaps, 10.0, 5);
        let o_series = onesided.build_vpin_series(&snaps, 10.0, 5);

        // Skip early zeros (pre-warmup). Compare means over non-zero entries.
        let b_avg: f64 = {
            let nonzero: Vec<f64> = b_series.iter().copied().filter(|&v| v > 1e-12).collect();
            if nonzero.is_empty() { 0.0 } else { nonzero.iter().sum::<f64>() / nonzero.len() as f64 }
        };
        let o_avg: f64 = {
            let nonzero: Vec<f64> = o_series.iter().copied().filter(|&v| v > 1e-12).collect();
            if nonzero.is_empty() { 0.0 } else { nonzero.iter().sum::<f64>() / nonzero.len() as f64 }
        };

        assert!(
            o_avg > b_avg + 0.5,
            "one-sided avg VPIN ({o_avg:.4}) must be significantly higher than balanced ({b_avg:.4})"
        );
    }

    #[test]
    fn vpin_series_empty_snapshots_returns_empty_vec() {
        let day = load_fixture(FIXTURE_CSV);
        let series = day.build_vpin_series(&[], 0.1, 5);
        assert!(series.is_empty(), "empty snapshot list must return empty series");
    }
}
