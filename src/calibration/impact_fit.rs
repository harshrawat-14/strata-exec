use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use crate::market::agg_trades::AggTradesDay;

/// Format of the expected historical market trades inside the CSV file.
#[derive(Debug, Deserialize, Clone)]
pub struct HistoricalTrade {
    pub timestamp_sec: f64,
    pub price_before: f64,
    pub price_after: f64,
    pub trade_size: f64,
}

/// A calibration engine to determine the empirical square-root market impact coefficient (Y).
///
/// Under the model: Impact Fraction = Y * sigma * sqrt(Q / V)
///
/// We solve for Y:
/// Y = ( (price_after - price_before).abs() / price_before ) / ( sigma * sqrt(Q / V) )
pub struct CalibrationEngine {
    pub daily_volume_estimate: f64,
    pub annualized_volatility: f64,
}

impl CalibrationEngine {
    /// Constructs a new calibration engine bounded by known macroscopic regimes.
    pub fn new(daily_volume_estimate: f64, annualized_volatility: f64) -> Self {
        Self {
            daily_volume_estimate,
            annualized_volatility,
        }
    }

    /// Iterates over historical trades to empirically derive the optimal `Y` impact scaler.
    pub fn calibrate(&self, trades: &[HistoricalTrade]) -> Option<f64> {
        if trades.is_empty() {
            return None;
        }

        let mut y_sum = 0.0;
        let mut valid_trades = 0;

        for trade in trades {
            // Safety bounds for empty or negative volume glitches
            if trade.trade_size <= 1e-6 || trade.price_before <= 1e-6 {
                continue;
            }

            // Delta Price relative fraction
            let diff = (trade.price_after - trade.price_before).abs();
            let impact_fraction = diff / trade.price_before;

            // Compute theoretical unscaled impact base -> sigma * sqrt(Q / V)
            let unscaled_impact_base =
                self.annualized_volatility * (trade.trade_size / self.daily_volume_estimate).sqrt();

            // Guard against division by zero
            if unscaled_impact_base > 1e-12 {
                let empirical_y = impact_fraction / unscaled_impact_base;
                y_sum += empirical_y;
                valid_trades += 1;
            }
        }

        if valid_trades == 0 {
            None
        } else {
            Some(y_sum / valid_trades as f64)
        }
    }

    /// Loads the structured `HistoricalTrade` models strictly off a target CSV payload on disk.
    pub fn load_csv(filepath: &str) -> Result<Vec<HistoricalTrade>, Box<dyn Error>> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let mut csv_reader = csv::Reader::from_reader(reader);

        let mut trades = Vec::new();
        for result in csv_reader.deserialize() {
            let record: HistoricalTrade = result?;
            trades.push(record);
        }

        Ok(trades)
    }
}

// ---------------------------------------------------------------------------
// Real-data bridge (Phase 2R Task 2): CalibrationEngine::calibrate needs
// HistoricalTrade { price_before, price_after, .. }, but this project's real
// market data is Binance aggTrades (a single trade price, no bracketing
// before/after) plus LOB snapshots (timestamped mids). Neither
// CounterfactualLob nor rl_env.rs ever bridged the two, which is WHY the
// calibration engine sat unwired for the RL training path even though it is
// wired (via --calibrate) into research_sim.rs's separate SquareRootImpact
// model. This closes that gap: derive price_before/price_after for each real
// trade from the nearest LOB snapshot mids bracketing its timestamp.
// ---------------------------------------------------------------------------

/// Build `HistoricalTrade` records from real aggTrades + LOB snapshot timestamps/mids,
/// using the nearest snapshot mid at or before the trade as `price_before` and the
/// nearest snapshot mid at or after the trade as `price_after`. Trades outside the
/// snapshot window (before the first or after the last) are skipped, since neither
/// bracket exists.
pub fn build_calibration_trades(
    agg: &AggTradesDay,
    snapshot_timestamps: &[u64],
    snapshot_mids: &[f64],
) -> Vec<HistoricalTrade> {
    assert_eq!(
        snapshot_timestamps.len(),
        snapshot_mids.len(),
        "snapshot_timestamps and snapshot_mids must be the same length"
    );
    if snapshot_timestamps.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for t in &agg.trades {
        // Index of the first snapshot at or after t.timestamp_ms.
        let after_idx = snapshot_timestamps.partition_point(|&ts| ts < t.timestamp_ms);
        if after_idx == 0 || after_idx >= snapshot_timestamps.len() {
            continue; // trade is before the first snapshot or after the last -- no bracket
        }
        let before_idx = after_idx - 1;
        let price_before = snapshot_mids[before_idx];
        let price_after = snapshot_mids[after_idx];
        if price_before <= 1e-9 || price_after <= 1e-9 {
            continue;
        }
        out.push(HistoricalTrade {
            timestamp_sec: t.timestamp_ms as f64 / 1000.0,
            price_before,
            price_after,
            trade_size: t.quantity_btc,
        });
    }
    out
}

/// Realized volatility from a snapshot mid-price series: stdev of consecutive
/// log-returns, annualized assuming `snapshots_per_day` snapshots span one
/// trading day (crypto trades 24/7, so this is a straight sqrt(N) scaling, no
/// trading-day-count adjustment). Matches the sigma convention CalibrationEngine
/// expects (comparable to `sigma_ref` ~0.20 used elsewhere in this codebase).
pub fn realized_volatility(snapshot_mids: &[f64], snapshots_per_day: f64) -> f64 {
    if snapshot_mids.len() < 2 {
        return 0.0;
    }
    let log_returns: Vec<f64> = snapshot_mids
        .windows(2)
        .filter_map(|w| {
            if w[0] > 1e-9 && w[1] > 1e-9 {
                Some((w[1] / w[0]).ln())
            } else {
                None
            }
        })
        .collect();
    if log_returns.len() < 2 {
        return 0.0;
    }
    let n = log_returns.len() as f64;
    let mean = log_returns.iter().sum::<f64>() / n;
    let var = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let per_snapshot_vol = var.sqrt();
    per_snapshot_vol * (snapshots_per_day * 365.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrate_empirical_algebra() {
        // Assume V = 1,000,000  and sigma = 0.05
        let engine = CalibrationEngine::new(1_000_000.0, 0.05);

        // A trade of size 10_000 -> sqrt(10_000 / 1_000_000) = sqrt(0.01) = 0.1
        // unscaled_base = 0.05 * 0.1 = 0.005
        //
        // Let's say we observed an empirical slippage from 100.0 up to 100.25
        // impact_fraction = 0.25 / 100.0 = 0.0025
        //
        // Y = 0.0025 / 0.005 = 0.50

        let trades = vec![HistoricalTrade {
            timestamp_sec: 1.0,
            price_before: 100.0,
            price_after: 100.25,
            trade_size: 10_000.0,
        }];

        let result_y = engine.calibrate(&trades).expect("calibration failed");
        assert!((result_y - 0.50).abs() < 1e-6);
    }

    #[test]
    fn test_skips_zero_trades() {
        let engine = CalibrationEngine::new(1_000_000.0, 0.05);
        let trades = vec![HistoricalTrade {
            timestamp_sec: 1.0,
            price_before: 100.0,
            price_after: 100.0,
            trade_size: 0.0, // should skip
        }];

        let result = engine.calibrate(&trades);
        assert!(result.is_none());
    }

    #[test]
    fn test_handles_multiple_trades() {
        let engine = CalibrationEngine::new(1_000_000.0, 0.05);
        let trades = vec![
            // Trade 1 yields Y = 0.5
            HistoricalTrade {
                timestamp_sec: 1.0,
                price_before: 100.0,
                price_after: 100.25,
                trade_size: 10_000.0,
            },
            // Trade 2 yields Y = 0.7  (let's say we bump the slippage observation to 0.35)
            // unscaled_base = 0.005,  impact_fraction = 0.35/100 = 0.0035 => Y = 0.7
            HistoricalTrade {
                timestamp_sec: 2.0,
                price_before: 100.0,
                price_after: 100.35,
                trade_size: 10_000.0,
            },
        ];

        let result_y = engine.calibrate(&trades).expect("calibration failed");
        // Mean of 0.5 and 0.7 is 0.6
        assert!((result_y - 0.60).abs() < 1e-6);
    }

    // ── build_calibration_trades / realized_volatility (Phase 2R Task 2) ──

    use crate::market::agg_trades::AggTrade;

    fn make_agg(trades: Vec<AggTrade>) -> AggTradesDay {
        AggTradesDay {
            trades,
            date: "2024-01-15".to_string(),
            skipped_rows: 0,
        }
    }

    #[test]
    fn build_calibration_trades_brackets_with_nearest_snapshots() {
        let agg = make_agg(vec![AggTrade {
            timestamp_ms: 1_500,
            price: 100.1,
            quantity_btc: 5.0,
            is_sell_aggressor: true,
        }]);
        let snapshot_timestamps = vec![1_000, 2_000, 3_000];
        let snapshot_mids = vec![100.0, 100.2, 100.4];

        let out = build_calibration_trades(&agg, &snapshot_timestamps, &snapshot_mids);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].price_before, 100.0,
            "must bracket with the snapshot at/before the trade"
        );
        assert_eq!(
            out[0].price_after, 100.2,
            "must bracket with the snapshot at/after the trade"
        );
        assert_eq!(out[0].trade_size, 5.0);
    }

    #[test]
    fn build_calibration_trades_skips_trades_outside_snapshot_window() {
        let agg = make_agg(vec![
            AggTrade {
                timestamp_ms: 500,
                price: 99.0,
                quantity_btc: 1.0,
                is_sell_aggressor: false,
            }, // before first
            AggTrade {
                timestamp_ms: 3_500,
                price: 101.0,
                quantity_btc: 1.0,
                is_sell_aggressor: false,
            }, // after last
        ]);
        let snapshot_timestamps = vec![1_000, 2_000, 3_000];
        let snapshot_mids = vec![100.0, 100.2, 100.4];

        let out = build_calibration_trades(&agg, &snapshot_timestamps, &snapshot_mids);
        assert!(
            out.is_empty(),
            "trades outside the snapshot window must be skipped, not bracketed with garbage"
        );
    }

    #[test]
    fn realized_volatility_is_zero_for_flat_price() {
        let mids = vec![100.0; 20];
        let vol = realized_volatility(&mids, 2880.0);
        assert_eq!(
            vol, 0.0,
            "a perfectly flat price series must have zero realized volatility"
        );
    }

    #[test]
    fn realized_volatility_is_positive_for_moving_price() {
        let mids: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 * 0.3).sin()).collect();
        let vol = realized_volatility(&mids, 2880.0);
        assert!(
            vol > 0.0,
            "a moving price series must have positive realized volatility, got {vol}"
        );
    }
}
