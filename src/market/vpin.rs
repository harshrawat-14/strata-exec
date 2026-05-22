// VPIN — Volume-Synchronized Probability of Informed Trading.
// Reference: Easley, López de Prado, O'Hara (2012).
//
// Measures order-flow toxicity via volume imbalance within fixed-size volume
// buckets. High VPIN signals active informed trading (adverse selection risk).
//
// Algorithm:
//   1. Classify each trade as buy- or sell-initiated via the tick rule.
//   2. Accumulate into volume buckets of fixed total size.
//   3. On bucket completion, record imbalance = |buy_vol - sell_vol| / total_vol.
//   4. Rolling VPIN = mean of imbalance over the last `window_buckets` buckets.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Completed volume bucket statistics.
pub struct VpinBucket {
    pub total_volume: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    /// Normalised order-flow imbalance for this bucket: |buy − sell| / total.
    pub vpin: f64,
}

/// Rolling VPIN estimator using the tick rule for trade classification.
///
/// Maintains a sliding window of completed volume buckets and computes
/// the rolling mean of per-bucket order-flow imbalance.
pub struct VpinCalculator {
    /// Target total volume per bucket (e.g. 50_000 USDT).
    bucket_size: f64,
    /// Number of completed buckets in the rolling window.
    window_buckets: usize,
    /// Sealed per-bucket imbalances in chronological order (oldest first).
    buckets: VecDeque<f64>,
    // ── In-progress bucket accumulators ──────────────────────────────────
    current_buy_vol: f64,
    current_sell_vol: f64,
    current_total_vol: f64,
    // ── Tick-rule state ───────────────────────────────────────────────────
    /// Direction of the last classified trade: +1 = buy, −1 = sell, 0 = none yet.
    last_direction: i8,
}

impl VpinCalculator {
    pub fn new(bucket_size: f64, window_buckets: usize) -> Self {
        Self {
            bucket_size,
            window_buckets,
            buckets: VecDeque::new(),
            current_buy_vol: 0.0,
            current_sell_vol: 0.0,
            current_total_vol: 0.0,
            last_direction: 0,
        }
    }

    /// Feed a trade into the estimator using the tick rule for classification.
    ///
    /// Returns the updated rolling VPIN whenever a bucket is sealed; returns
    /// `None` on partial fills that have not yet completed a bucket.
    ///
    /// A single trade may fill multiple buckets if `volume >= 2 × bucket_size`.
    /// In that case the return value reflects the VPIN after the final sealed bucket.
    ///
    /// # Tick rule
    ///   price > prev_price  → buy-initiated
    ///   price < prev_price  → sell-initiated
    ///   price == prev_price → same direction as the previous classified trade
    pub fn add_trade(&mut self, price: f64, volume: f64, prev_price: f64) -> Option<f64> {
        let direction: i8 = if price > prev_price {
            1
        } else if price < prev_price {
            -1
        } else if self.last_direction != 0 {
            self.last_direction
        } else {
            1 // first trade with no prior context: default to buy
        };
        self.last_direction = direction;

        let mut remaining = volume;
        let mut result: Option<f64> = None;

        while remaining > 1e-15 {
            let space = self.bucket_size - self.current_total_vol;
            let fill = remaining.min(space);

            if direction > 0 {
                self.current_buy_vol += fill;
            } else {
                self.current_sell_vol += fill;
            }
            self.current_total_vol += fill;
            remaining -= fill;

            // Seal bucket when filled (allow for floating-point proximity).
            if self.current_total_vol >= self.bucket_size - 1e-12 {
                let total = self.current_total_vol;
                let imbalance = if total > 1e-15 {
                    (self.current_buy_vol - self.current_sell_vol).abs() / total
                } else {
                    0.0
                };

                self.buckets.push_back(imbalance);
                if self.buckets.len() > self.window_buckets {
                    self.buckets.pop_front();
                }

                // Reset in-progress bucket.
                self.current_buy_vol = 0.0;
                self.current_sell_vol = 0.0;
                self.current_total_vol = 0.0;

                let vpin = self.buckets.iter().sum::<f64>() / self.buckets.len() as f64;
                result = Some(vpin);
            }
        }

        result
    }

    /// Current rolling VPIN estimate.
    ///
    /// Returns `None` if no bucket has been completed yet.
    pub fn current_vpin(&self) -> Option<f64> {
        if self.buckets.is_empty() {
            None
        } else {
            Some(self.buckets.iter().sum::<f64>() / self.buckets.len() as f64)
        }
    }

    /// Returns `true` if the current VPIN estimate exceeds `threshold`.
    /// Returns `false` when no estimate is available yet (insufficient data).
    pub fn is_toxic(&self, threshold: f64) -> bool {
        self.current_vpin().map_or(false, |v| v > threshold)
    }
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/// Compute a VPIN bucket size from total daily volume.
///
/// Standard calibration: 50 buckets per day gives one VPIN estimate every
/// ~29 minutes on a typical day. Caller can override `target_buckets_per_day`
/// for higher or lower resolution.
pub fn calibrate_bucket_size(total_volume_btc: f64, target_buckets_per_day: usize) -> f64 {
    total_volume_btc / target_buckets_per_day as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: fill one complete bucket with a given buy fraction.
    // buy_fraction ∈ [0, 1]; total volume = bucket_size.
    fn fill_balanced_bucket(calc: &mut VpinCalculator, bucket_size: f64, buy_fraction: f64) {
        let buy = bucket_size * buy_fraction;
        let sell = bucket_size * (1.0 - buy_fraction);
        if buy > 1e-12 {
            // Uptick → buy
            calc.add_trade(101.0, buy, 100.0);
        }
        if sell > 1e-12 {
            // Downtick → sell
            calc.add_trade(99.0, sell, 101.0);
        }
    }

    #[test]
    fn vpin_is_none_before_first_bucket_completes() {
        let bucket_size = 1_000.0;
        let mut calc = VpinCalculator::new(bucket_size, 10);
        // Send a partial trade that does not fill the bucket.
        let result = calc.add_trade(101.0, bucket_size * 0.5, 100.0);
        assert!(result.is_none(), "partial bucket must not return a VPIN");
        assert!(
            calc.current_vpin().is_none(),
            "current_vpin must be None before any bucket completes"
        );
    }

    #[test]
    fn vpin_is_zero_when_buy_and_sell_perfectly_balanced() {
        let bucket_size = 1_000.0;
        let mut calc = VpinCalculator::new(bucket_size, 10);
        fill_balanced_bucket(&mut calc, bucket_size, 0.5);
        let vpin = calc.current_vpin().expect("bucket must be complete");
        assert!(
            vpin.abs() < 1e-12,
            "perfectly balanced buy/sell must yield VPIN = 0, got {vpin}"
        );
    }

    #[test]
    fn vpin_is_one_when_all_volume_is_buy_initiated() {
        let bucket_size = 1_000.0;
        let mut calc = VpinCalculator::new(bucket_size, 10);
        // Single uptick trade fills the entire bucket with buys.
        let vpin = calc
            .add_trade(101.0, bucket_size, 100.0)
            .expect("bucket must seal after a full-bucket buy");
        assert!(
            (vpin - 1.0).abs() < 1e-12,
            "all-buy bucket must yield VPIN = 1.0, got {vpin}"
        );
    }

    #[test]
    fn vpin_increases_after_one_sided_volume_surge() {
        let bucket_size = 1_000.0;
        let mut calc = VpinCalculator::new(bucket_size, 10);

        // First bucket: perfectly balanced → imbalance = 0.
        fill_balanced_bucket(&mut calc, bucket_size, 0.5);
        let vpin_after_balanced = calc.current_vpin().unwrap();

        // Second bucket: all buys → imbalance = 1.
        calc.add_trade(101.0, bucket_size, 100.0);
        let vpin_after_surge = calc.current_vpin().unwrap();

        assert!(
            vpin_after_surge > vpin_after_balanced,
            "VPIN must rise after a one-sided buy surge: before={vpin_after_balanced:.4} after={vpin_after_surge:.4}"
        );
    }

    #[test]
    fn tick_rule_classifies_uptick_as_buy() {
        let bucket_size = 1_000.0;
        let mut calc = VpinCalculator::new(bucket_size, 10);
        // price (101) > prev_price (100) → buy
        calc.add_trade(101.0, bucket_size, 100.0);
        let vpin = calc.current_vpin().expect("bucket must be complete");
        assert!(
            (vpin - 1.0).abs() < 1e-12,
            "uptick must classify as buy (VPIN=1), got {vpin}"
        );
    }

    #[test]
    fn tick_rule_classifies_downtick_as_sell() {
        let bucket_size = 1_000.0;
        let mut calc = VpinCalculator::new(bucket_size, 10);
        // price (99) < prev_price (100) → sell
        calc.add_trade(99.0, bucket_size, 100.0);
        let vpin = calc.current_vpin().expect("bucket must be complete");
        assert!(
            (vpin - 1.0).abs() < 1e-12,
            "downtick must classify as sell (VPIN still=1 since |0-1000|/1000=1), got {vpin}"
        );
    }

    #[test]
    fn vpin_stays_in_zero_one_range() {
        let bucket_size = 100.0;
        let mut calc = VpinCalculator::new(bucket_size, 20);
        // Alternate buy/sell trades of varying sizes.
        let mut prev_price = 100.0_f64;
        for i in 0..200usize {
            let price = if i % 3 == 0 { prev_price + 1.0 } else { prev_price - 0.5 };
            let volume = (i % 50 + 1) as f64 * 5.0;
            calc.add_trade(price, volume, prev_price);
            if let Some(v) = calc.current_vpin() {
                assert!(
                    v >= 0.0 && v <= 1.0,
                    "VPIN must stay in [0,1], got {v} at step {i}"
                );
            }
            prev_price = price;
        }
    }

    #[test]
    fn window_rolls_correctly_after_window_buckets_filled() {
        let bucket_size = 1_000.0;
        let window = 3;
        let mut calc = VpinCalculator::new(bucket_size, window);

        // Fill `window` balanced buckets → VPIN = 0.
        for _ in 0..window {
            fill_balanced_bucket(&mut calc, bucket_size, 0.5);
        }
        let vpin_all_balanced = calc.current_vpin().expect("window must be full");
        assert!(
            vpin_all_balanced.abs() < 1e-12,
            "all balanced buckets → VPIN = 0, got {vpin_all_balanced}"
        );

        // Add one all-buy bucket. The oldest balanced bucket drops out of the window.
        // Window now: [0, 0, 1] (keeping last 3) → mean = 1/3.
        calc.add_trade(101.0, bucket_size, 100.0);
        let vpin_after_surge = calc.current_vpin().expect("must have VPIN after roll");
        let expected = 1.0 / 3.0;
        assert!(
            (vpin_after_surge - expected).abs() < 1e-10,
            "after rolling window: expected VPIN = {expected:.4}, got {vpin_after_surge:.4}"
        );
    }

    #[test]
    fn bucket_size_calibrates_to_volume_divided_by_target() {
        let result = calibrate_bucket_size(240_059.0, 50);
        let expected = 240_059.0 / 50.0; // 4801.18
        assert!(
            (result - expected).abs() < 1e-9,
            "calibrate_bucket_size(240059, 50) must equal {expected:.4}, got {result:.4}"
        );
    }
}
