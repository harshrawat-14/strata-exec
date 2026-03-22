use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

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
            let unscaled_impact_base = self.annualized_volatility
                * (trade.trade_size / self.daily_volume_estimate).sqrt();

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
}
