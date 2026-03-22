use std::collections::VecDeque;

/// Rolling-window volatility estimator using incremental variance.
///
/// Maintains O(1) amortised updates by tracking running `sum` and `sum_sq`
/// and adjusting them when the oldest return is evicted.
pub struct VolatilityEstimator {
    window: usize,
    returns: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl VolatilityEstimator {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            returns: VecDeque::with_capacity(window),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Push a new return value into the rolling window.
    ///
    /// If the window is full, the oldest value is evicted and the running
    /// sums are decremented accordingly.
    pub fn update(&mut self, return_value: f64) {
        if self.returns.len() == self.window {
            if let Some(old) = self.returns.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }

        self.returns.push_back(return_value);
        self.sum += return_value;
        self.sum_sq += return_value * return_value;
    }

    /// Returns the current standard deviation of the rolling window.
    ///
    /// Returns `None` when fewer than 2 data points are available,
    /// since variance is undefined for a single observation.
    pub fn current(&self) -> Option<f64> {
        let n = self.returns.len();
        if n < 2 {
            return None;
        }

        let n_f = n as f64;
        let mean = self.sum / n_f;
        let variance = (self.sum_sq / n_f) - (mean * mean);

        // Guard against negative variance from floating-point drift.
        Some(variance.max(0.0).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_estimator_returns_none() {
        let est = VolatilityEstimator::new(5);
        assert!(est.current().is_none());
    }

    #[test]
    fn single_point_returns_none() {
        let mut est = VolatilityEstimator::new(5);
        est.update(1.0);
        assert!(est.current().is_none());
    }

    #[test]
    fn small_window_computes_stddev() {
        let mut est = VolatilityEstimator::new(3);
        // values: [1.0, 2.0, 3.0]
        est.update(1.0);
        est.update(2.0);
        est.update(3.0);

        let stddev = est.current().expect("should have value");
        // mean = 2.0, variance = ((1+4+9)/3) - 4 = 14/3 - 4 = 2/3
        // stddev = sqrt(2/3) ≈ 0.8165
        let expected = (2.0_f64 / 3.0).sqrt();
        assert!((stddev - expected).abs() < 1e-10);
    }

    #[test]
    fn rolling_eviction_updates_correctly() {
        let mut est = VolatilityEstimator::new(3);
        est.update(1.0);
        est.update(2.0);
        est.update(3.0);
        // Window: [1, 2, 3]

        est.update(4.0);
        // Window: [2, 3, 4] — 1.0 evicted

        let stddev = est.current().expect("should have value");
        // mean = 3.0, variance = ((4+9+16)/3) - 9 = 29/3 - 9 = 2/3
        let expected = (2.0_f64 / 3.0).sqrt();
        assert!((stddev - expected).abs() < 1e-10);

        est.update(4.0);
        // Window: [3, 4, 4] — 2.0 evicted

        let stddev2 = est.current().expect("should have value");
        // mean = 11/3, variance = (9+16+16)/3 - (11/3)^2 = 41/3 - 121/9 = 2/9
        let expected2 = (2.0_f64 / 9.0).sqrt();
        assert!((stddev2 - expected2).abs() < 1e-10);
    }
}
