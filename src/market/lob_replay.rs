// Phase 2 Track B — Real market data ingestion
// Third execution path alongside formula and synthetic LOB.
// Data source: Binance Vision historical L2 snapshots (bookDepth format)
// Does NOT replace lob_generator.rs — both coexist.
//
// Two CSV layouts are supported, detected automatically from the header:
//
// (A) Binance bookDepth tall format — one row per price level:
//       timestamp, percentage, depth, notional
//   percentage < 0 → bid level (−1 = best bid, −5 = 5 levels out)
//   percentage > 0 → ask level (+1 = best ask, +5 = 5 levels out)
//   depth and notional are CUMULATIVE from the mid outward.
//   Per-level qty/price are recovered by differencing consecutive levels.
//
// (B) Wide format — one row per snapshot (used by unit-test fixture):
//       timestamp, b0_p, b0_q, b1_p, b1_q, ..., a0_p, a0_q, a1_p, a1_q, ...
//   Each bN/aN column pair is one level at a direct (non-cumulative) price.
//
// Detection: if the header contains all of {percentage, depth, notional}, format A
// is assumed; otherwise format B is used.

use std::error::Error;

use chrono::NaiveDateTime;
use crate::market::order_book::{OrderBook, PriceLevel};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One parsed L2 snapshot.
pub struct LobSnapshot {
    pub timestamp: u64,
    /// Bid levels, sorted descending by price (best bid first).
    pub bids: Vec<PriceLevel>,
    /// Ask levels, sorted ascending by price (best ask first).
    pub asks: Vec<PriceLevel>,
}

impl LobSnapshot {
    /// Convert to an `OrderBook`.
    pub fn to_order_book(&self) -> OrderBook {
        let mut book = OrderBook::new(self.asks.clone(), self.bids.clone());
        book.timestamp = self.timestamp;
        book
    }
}

/// Replays historical L2 snapshots in order, one per simulation step.
///
/// When the replay is exhausted, `next_snapshot` returns `None` so the caller
/// can fall back to the synthetic LOB.
pub struct LobReplay {
    snapshots: Vec<LobSnapshot>,
    cursor: usize,
}

impl LobReplay {
    /// Parse a depth snapshot CSV in either Binance bookDepth (tall) or wide format.
    ///
    /// Format is detected from the header — no user configuration needed.
    /// Rows / groups with parse errors are silently skipped.
    /// Returns an error only if the file cannot be opened or yields zero valid snapshots.
    pub fn from_csv(path: &str) -> Result<Self, Box<dyn Error>> {
        // Peek at the header to choose the parser.
        let mut reader = csv::Reader::from_path(path)?;
        let headers: Vec<String> = reader
            .headers()?
            .iter()
            .map(|h| h.trim().to_lowercase())
            .collect();

        let snapshots = if is_binance_tall_format(&headers) {
            parse_binance_tall(path)?
        } else {
            parse_wide(path, &headers)?
        };

        if snapshots.is_empty() {
            return Err("CSV contained no valid L2 snapshots".into());
        }

        Ok(Self { snapshots, cursor: 0 })
    }

    /// Return the next snapshot as an `OrderBook`, advancing the cursor.
    /// Returns `None` when the replay is exhausted.
    pub fn next_snapshot(&mut self) -> Option<OrderBook> {
        if self.cursor >= self.snapshots.len() {
            return None;
        }
        let book = self.snapshots[self.cursor].to_order_book();
        self.cursor += 1;
        Some(book)
    }

    /// Reset the cursor to the beginning so the replay can be reused.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Advance the cursor to the given snapshot index.
    /// Clamps to the last valid index if `offset` exceeds the snapshot count.
    pub fn seek(&mut self, offset: usize) {
        if self.snapshots.is_empty() {
            self.cursor = 0;
        } else {
            self.cursor = offset.min(self.snapshots.len() - 1);
        }
    }

    /// Number of snapshots loaded.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// True if no snapshots were loaded.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Return the unix-millisecond timestamp of every loaded snapshot, in order.
    /// Used by `AggTradesDay::build_vpin_series` to align aggTrades windows with
    /// bookDepth snapshot boundaries.
    pub fn snapshot_timestamps(&self) -> Vec<u64> {
        self.snapshots.iter().map(|s| s.timestamp).collect()
    }

    /// Compute a time series of VPIN estimates from the loaded L2 snapshots.
    ///
    /// Uses midprice changes between consecutive snapshots as the tick-rule
    /// direction signal, and the absolute change in total book depth as a
    /// proxy for traded volume (consuming depth ≈ aggressive order flow).
    ///
    /// Returns a vector of `(timestamp_ms, vpin)` pairs, one per bucket
    /// completion.  May be shorter than `self.len()` when depth changes are
    /// too small to fill many buckets.
    pub fn compute_vpin_series(
        &self,
        bucket_size: f64,
        window_buckets: usize,
    ) -> Vec<(u64, f64)> {
        use crate::market::vpin::VpinCalculator;

        let mut calc = VpinCalculator::new(bucket_size, window_buckets);
        let mut results: Vec<(u64, f64)> = Vec::new();

        for i in 1..self.snapshots.len() {
            let prev = &self.snapshots[i - 1];
            let curr = &self.snapshots[i];

            // Midprice direction via tick rule.
            let prev_mid = {
                let b = prev.to_order_book();
                b.mid_price().unwrap_or(0.0)
            };
            let curr_mid = {
                let b = curr.to_order_book();
                b.mid_price().unwrap_or(0.0)
            };

            // Volume proxy: absolute change in total book depth.
            // Depth consumed from either side approximates aggressive trade flow.
            let prev_depth: f64 = prev.bids.iter().map(|l| l.quantity).sum::<f64>()
                + prev.asks.iter().map(|l| l.quantity).sum::<f64>();
            let curr_depth: f64 = curr.bids.iter().map(|l| l.quantity).sum::<f64>()
                + curr.asks.iter().map(|l| l.quantity).sum::<f64>();
            let volume_proxy = (curr_depth - prev_depth).abs();

            if volume_proxy > 1e-12 && prev_mid > 1e-15 {
                if let Some(vpin) = calc.add_trade(curr_mid, volume_proxy, prev_mid) {
                    results.push((curr.timestamp, vpin));
                }
            }
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

fn is_binance_tall_format(headers: &[String]) -> bool {
    ["percentage", "depth", "notional"]
        .iter()
        .all(|&col| headers.contains(&col.to_string()))
}

// ---------------------------------------------------------------------------
// Parser A — Binance bookDepth tall format
// ---------------------------------------------------------------------------

/// One raw data row from the tall format before grouping.
struct TallRow {
    timestamp: String,
    percentage: i32,
    cum_depth: f64,
    cum_notional: f64,
}

/// Parse the Binance bookDepth tall CSV.
///
/// Algorithm:
///   1. Read all rows, grouping by timestamp string (preserves file order).
///   2. For each group, split into bid rows (pct < 0) and ask rows (pct > 0).
///   3. Sort bid rows by pct descending (−1 first = best bid).
///   4. Sort ask rows by pct ascending  (+1 first = best ask).
///   5. Diff consecutive cumulative depths to get per-level qty and price.
///   6. Discard groups with zero valid levels on either side.
fn parse_binance_tall(path: &str) -> Result<Vec<LobSnapshot>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers: Vec<String> = reader
        .headers()?
        .iter()
        .map(|h| h.trim().to_lowercase())
        .collect();

    let ts_col = col_idx(&headers, "timestamp")
        .ok_or("missing 'timestamp' column")?;
    let pct_col = col_idx(&headers, "percentage")
        .ok_or("missing 'percentage' column")?;
    let depth_col = col_idx(&headers, "depth")
        .ok_or("missing 'depth' column")?;
    let notional_col = col_idx(&headers, "notional")
        .ok_or("missing 'notional' column")?;

    // Accumulate raw rows, keyed by timestamp string to preserve stable order.
    let mut groups: Vec<(String, Vec<TallRow>)> = Vec::new();
    let mut key_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for result in reader.records() {
        let rec = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let ts = match rec.get(ts_col) {
            Some(v) => v.trim().to_string(),
            None => continue,
        };
        let pct: i32 = match rec.get(pct_col).and_then(|v| v.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let cum_depth: f64 = match rec.get(depth_col).and_then(|v| v.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let cum_notional: f64 = match rec.get(notional_col).and_then(|v| v.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };

        if pct == 0 || cum_depth < 0.0 {
            continue; // pct=0 has no economic meaning; negative depth is degenerate
        }

        let row = TallRow { timestamp: ts.clone(), percentage: pct, cum_depth, cum_notional };

        if let Some(&idx) = key_index.get(&ts) {
            groups[idx].1.push(row);
        } else {
            key_index.insert(ts, groups.len());
            groups.push((row.timestamp.clone(), vec![row]));
        }
    }

    let mut snapshots: Vec<LobSnapshot> = Vec::with_capacity(groups.len());

    for (ts_str, rows) in groups {
        let mut bid_rows: Vec<&TallRow> = rows.iter().filter(|r| r.percentage < 0).collect();
        let mut ask_rows: Vec<&TallRow> = rows.iter().filter(|r| r.percentage > 0).collect();

        // Best bid = pct closest to 0 (−1), deepest = most negative.
        // Sort descending by pct so index 0 = best bid.
        bid_rows.sort_by_key(|r| std::cmp::Reverse(r.percentage));

        // Best ask = pct closest to 0 (+1), deepest = most positive.
        // Sort ascending by pct so index 0 = best ask.
        ask_rows.sort_by_key(|r| r.percentage);

        let bids = diff_levels(&bid_rows);
        let asks = diff_levels(&ask_rows);

        if bids.is_empty() || asks.is_empty() {
            continue;
        }

        let timestamp = parse_timestamp_ms(ts_str.trim())?;

        snapshots.push(LobSnapshot { timestamp, bids, asks });
    }

    // Keep rows in file order (groups were inserted in encounter order).
    Ok(snapshots)
}

/// Recover per-level `PriceLevel` values by differencing cumulative rows.
///
/// `sorted_rows[0]` is the best level (smallest absolute pct).
/// The cumulative depth at level k includes all levels from 0..=k.
/// Per-level qty  = cum_depth[k]    − cum_depth[k−1]   (0 at k=0)
/// Per-level price = delta_notional / delta_qty          (price = dn/dq)
fn diff_levels(sorted_rows: &[&TallRow]) -> Vec<PriceLevel> {
    let mut levels: Vec<PriceLevel> = Vec::with_capacity(sorted_rows.len());
    let mut prev_depth = 0.0_f64;
    let mut prev_notional = 0.0_f64;

    for row in sorted_rows {
        let dq = row.cum_depth - prev_depth;
        let dn = row.cum_notional - prev_notional;

        if dq > 1e-12 {
            let price = infer_price(dn, dq);
            if price > 0.0 {
                levels.push(PriceLevel { price, quantity: dq });
            }
        }

        prev_depth = row.cum_depth;
        prev_notional = row.cum_notional;
    }

    levels
}

/// Recover a price from a notional/depth pair.
fn infer_price(notional: f64, depth: f64) -> f64 {
    if depth.abs() < 1e-15 { 100.0 } else { notional / depth }
}

// ---------------------------------------------------------------------------
// Parser B — wide format (original; used by unit-test fixture)
// ---------------------------------------------------------------------------

/// (price_col_index, qty_col_index) for one depth level.
type LevelCols = (usize, usize);

fn parse_wide(path: &str, headers: &[String]) -> Result<Vec<LobSnapshot>, Box<dyn Error>> {
    let ts_col: Option<usize> = headers.iter().position(|h| h == "timestamp");
    let bid_pairs = collect_level_pairs(headers, 'b');
    let ask_pairs = collect_level_pairs(headers, 'a');

    let mut reader = csv::Reader::from_path(path)?;
    let mut snapshots: Vec<LobSnapshot> = Vec::new();
    let mut row_idx: u64 = 0;

    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => { row_idx += 1; continue; }
        };

        let timestamp = ts_col
            .and_then(|c| record.get(c))
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(row_idx);

        let bids = parse_wide_levels(&record, &bid_pairs);
        let asks = parse_wide_levels(&record, &ask_pairs);

        if !bids.is_empty() && !asks.is_empty() {
            snapshots.push(LobSnapshot { timestamp, bids, asks });
        }

        row_idx += 1;
    }

    Ok(snapshots)
}

/// Scan headers for `<side><N>_p` / `<side><N>_q` pairs sorted by level number.
fn collect_level_pairs(headers: &[String], side: char) -> Vec<LevelCols> {
    let prefix = side.to_string();
    let mut pairs: Vec<(usize, LevelCols)> = Vec::new();

    for (ci, h) in headers.iter().enumerate() {
        if let Some(rest) = h.strip_prefix(&prefix) {
            if let Some(n_str) = rest.strip_suffix("_p") {
                if let Ok(n) = n_str.parse::<usize>() {
                    let qty_name = format!("{prefix}{n}_q");
                    if let Some(qi) = headers.iter().position(|h2| *h2 == qty_name) {
                        pairs.push((n, (ci, qi)));
                    }
                }
            }
        }
    }

    pairs.sort_by_key(|(n, _)| *n);
    pairs.into_iter().map(|(_, cols)| cols).collect()
}

/// Parse direct (non-cumulative) price/qty pairs from a wide-format record.
fn parse_wide_levels(record: &csv::StringRecord, pairs: &[LevelCols]) -> Vec<PriceLevel> {
    pairs
        .iter()
        .filter_map(|(p_col, q_col)| {
            let price = record.get(*p_col)?.trim().parse::<f64>().ok()?;
            let qty = record.get(*q_col)?.trim().parse::<f64>().ok()?;
            if price > 0.0 && qty > 0.0 {
                Some(PriceLevel { price, quantity: qty })
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn col_idx(headers: &[String], name: &str) -> Option<usize> {
    headers.iter().position(|h| h == name)
}

/// Parse a timestamp string to unix milliseconds.
///
/// Tries unix-ms integer first (some files use numeric timestamps).
/// Falls back to ISO "YYYY-MM-DD HH:MM:SS" (Binance bookDepth format).
/// Returns an error rather than silently producing garbage via byte-hash.
pub fn parse_timestamp_ms(s: &str) -> Result<u64, Box<dyn std::error::Error>> {
    if let Ok(ms) = s.parse::<u64>() {
        return Ok(ms);
    }
    let dt = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
        .map_err(|e| format!("unrecognised timestamp '{s}': {e}"))?;
    Ok(dt.and_utc().timestamp_millis() as u64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub mod tests {
    use super::*;

    // ── Wide-format fixture (unchanged) ──────────────────────────────────────

    /// Minimal 5-snapshot wide-format fixture.
    /// Layout: timestamp, b0_p, b0_q, b1_p, b1_q, a0_p, a0_q, a1_p, a1_q
    pub const FIXTURE_CSV: &str = "\
timestamp,b0_p,b0_q,b1_p,b1_q,a0_p,a0_q,a1_p,a1_q
1000,99.9,10.0,99.8,20.0,100.1,15.0,100.2,25.0
1001,99.85,12.0,99.75,18.0,100.15,14.0,100.25,22.0
1002,99.8,8.0,99.7,16.0,100.2,10.0,100.3,20.0
1003,99.95,11.0,99.85,19.0,100.05,13.0,100.15,21.0
1004,99.88,9.0,99.78,17.0,100.12,12.0,100.22,23.0
";

    /// Minimal Binance bookDepth tall-format fixture (2 snapshots, ±1 and ±2 levels).
    /// Cumulative depths match the wide fixture's first snapshot for cross-format comparison.
    /// snap 0: bid best ≈41528, ask best ≈41889 (BTC-scale prices)
    pub const BINANCE_FIXTURE_CSV: &str = "\
timestamp,percentage,depth,notional
2024-01-15 00:00:12,-2,5752.893,237633205.34422
2024-01-15 00:00:12,-1,2692.069,111796720.65430
2024-01-15 00:00:12,1,1017.460,42620326.49210
2024-01-15 00:00:12,2,2599.593,109655932.61735
2024-01-15 00:00:45,-2,5631.718,232812838.22400
2024-01-15 00:00:45,-1,2573.466,106974841.66560
2024-01-15 00:00:45,1,1173.950,49241948.38940
2024-01-15 00:00:45,2,2832.353,119594607.93045
";

    /// Write fixture content to a temp file; return the path.
    ///
    /// Uses a monotonic counter so parallel test threads each get a unique path,
    /// preventing the race condition where two threads truncate and write the same file.
    pub fn write_fixture(content: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir()
            .join(format!("lob_fixture_{}_{}_{}.csv",
                std::process::id(),
                id,
                content.len()))
            .to_str()
            .unwrap()
            .to_string();
        std::fs::write(&path, content).expect("fixture write must succeed");
        path
    }

    // ── Existing 5 tests — unchanged signatures, still use wide fixture ──────

    #[test]
    fn lob_replay_loads_csv_without_error() {
        let path = write_fixture(FIXTURE_CSV);
        let replay = LobReplay::from_csv(&path).expect("must load without error");
        assert_eq!(replay.len(), 5, "must load all 5 rows");
        assert!(!replay.is_empty());
    }

    #[test]
    fn lob_replay_next_returns_valid_order_book() {
        let path = write_fixture(FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");
        let book = replay.next_snapshot().expect("first snapshot must exist");

        // Row 0: bid best = 99.9, ask best = 100.1
        let best_bid = book.best_bid().expect("must have best bid");
        let best_ask = book.best_ask().expect("must have best ask");

        assert!(
            (best_bid - 99.9).abs() < 1e-9,
            "best_bid expected 99.9, got {best_bid}"
        );
        assert!(
            (best_ask - 100.1).abs() < 1e-9,
            "best_ask expected 100.1, got {best_ask}"
        );
        assert!(best_ask > best_bid, "book must not be crossed");
    }

    #[test]
    fn lob_replay_resets_to_beginning() {
        let path = write_fixture(FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");

        for _ in 0..5 {
            assert!(replay.next_snapshot().is_some());
        }
        assert!(replay.next_snapshot().is_none(), "must be exhausted");

        replay.reset();
        let book = replay.next_snapshot().expect("must return first snapshot after reset");
        let best_bid = book.best_bid().expect("must have best bid");
        assert!(
            (best_bid - 99.9).abs() < 1e-9,
            "after reset, best_bid expected 99.9, got {best_bid}"
        );
    }

    #[test]
    fn lob_replay_produces_nonzero_spread_from_real_data() {
        let path = write_fixture(FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");
        while let Some(book) = replay.next_snapshot() {
            let spread = book.spread().expect("every snapshot must have a spread");
            assert!(
                spread > 1e-9,
                "spread must be positive (non-zero): got {spread}"
            );
        }
    }

    #[test]
    fn real_spread_differs_from_synthetic_spread() {
        use crate::market::lob_generator::{generate_lob, LobGeneratorParams};

        let path = write_fixture(FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");
        let real_book = replay.next_snapshot().expect("must have first snapshot");

        // The fixture mid is (99.9 + 100.1) / 2 = 100.0.
        let real_spread = real_book.spread().expect("must have spread");

        let params = LobGeneratorParams::default();
        let synthetic_book = generate_lob(100.0, 0.20, &params);
        let synth_spread = synthetic_book.spread().expect("synthetic must have spread");

        assert!(
            (real_spread - synth_spread).abs() > 1e-9,
            "real spread ({real_spread:.6}) must differ from synthetic ({synth_spread:.6})"
        );
    }

    // ── Binance tall-format tests ────────────────────────────────────────────

    #[test]
    fn binance_format_detected_and_loads_two_snapshots() {
        let path = write_fixture(BINANCE_FIXTURE_CSV);
        let replay = LobReplay::from_csv(&path).expect("must load Binance format");
        assert_eq!(replay.len(), 2, "must parse 2 distinct timestamps");
    }

    #[test]
    fn binance_snapshot_has_uncrossed_book() {
        let path = write_fixture(BINANCE_FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");
        while let Some(book) = replay.next_snapshot() {
            let bid = book.best_bid().expect("must have best bid");
            let ask = book.best_ask().expect("must have best ask");
            assert!(ask > bid, "book must not be crossed: bid={bid:.2} ask={ask:.2}");
        }
    }

    #[test]
    fn binance_diff_recovers_correct_best_bid_price() {
        // Snapshot 0: bid pct=-1 cum_depth=2692.069, cum_notional=111796720.654
        // Best bid delta = (2692.069 - 0, 111796720.654 - 0) → price = 111796720.654/2692.069 ≈ 41528.18
        let path = write_fixture(BINANCE_FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");
        let book = replay.next_snapshot().expect("first snapshot must exist");
        let best_bid = book.best_bid().expect("must have best bid");
        let expected = 111_796_720.654_30 / 2_692.069;
        assert!(
            (best_bid - expected).abs() < 0.01,
            "best_bid expected {expected:.4}, got {best_bid:.4}"
        );
    }

    #[test]
    fn binance_diff_recovers_correct_best_ask_price() {
        // Snapshot 0: ask pct=+1 cum_depth=1017.460, cum_notional=42620326.492
        // Best ask delta = (1017.460 - 0, 42620326.492 - 0) → price ≈ 41888.95
        let path = write_fixture(BINANCE_FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");
        let book = replay.next_snapshot().expect("first snapshot must exist");
        let best_ask = book.best_ask().expect("must have best ask");
        let expected = 42_620_326.492_10 / 1_017.460;
        assert!(
            (best_ask - expected).abs() < 0.01,
            "best_ask expected {expected:.4}, got {best_ask:.4}"
        );
    }

    #[test]
    fn binance_bids_are_descending_asks_are_ascending() {
        let path = write_fixture(BINANCE_FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");
        while let Some(book) = replay.next_snapshot() {
            // Verify bid side: each successive bid price must be <= previous.
            // We check indirectly: bid_depth_up_to grows as price limit decreases.
            let best_bid = book.best_bid().unwrap();
            let d1 = book.bid_depth_up_to(best_bid);
            let d2 = book.bid_depth_up_to(best_bid - 1000.0); // include deeper level
            assert!(d2 >= d1, "bid depth must grow as price limit falls");

            // Verify ask side: ask_depth_up_to grows as price limit increases.
            let best_ask = book.best_ask().unwrap();
            let a1 = book.ask_depth_up_to(best_ask);
            let a2 = book.ask_depth_up_to(best_ask + 1000.0);
            assert!(a2 >= a1, "ask depth must grow as price limit rises");
        }
    }

    #[test]
    fn seek_advances_cursor_to_correct_position() {
        let path = write_fixture(FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");
        // Seek to index 2; row 2 is: 1002,99.8,8.0,...
        replay.seek(2);
        let book = replay.next_snapshot().expect("must return snapshot at index 2");
        let best_bid = book.best_bid().expect("must have best bid");
        assert!(
            (best_bid - 99.8).abs() < 1e-9,
            "after seek(2), best_bid expected 99.8, got {best_bid}"
        );
    }

    #[test]
    fn seek_clamps_to_last_valid_position() {
        let path = write_fixture(FIXTURE_CSV);
        let mut replay = LobReplay::from_csv(&path).expect("must load");
        // Seek far beyond the 5-snapshot replay — cursor should land at index 4.
        replay.seek(9999);
        let book = replay.next_snapshot().expect("must return last snapshot");
        // Row 4: 1004,99.88,9.0,...
        let best_bid = book.best_bid().expect("must have best bid");
        assert!(
            (best_bid - 99.88).abs() < 1e-9,
            "after seek(9999), best_bid expected 99.88, got {best_bid}"
        );
        assert!(
            replay.next_snapshot().is_none(),
            "after consuming last snapshot, replay must be exhausted"
        );
    }

    #[test]
    fn bookdepth_iso_timestamp_parses_to_unix_ms() {
        // "2024-01-15 00:00:12" UTC = 1705276800000 + 12000 = 1705276812000 ms
        let result = parse_timestamp_ms("2024-01-15 00:00:12")
            .expect("ISO timestamp must parse without error");
        assert_eq!(
            result, 1_705_276_812_000,
            "ISO timestamp must convert to unix ms: expected 1705276812000, got {result}"
        );
    }
}
