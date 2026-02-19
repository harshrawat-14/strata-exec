/// Stateless drift estimator for expected price movement.
pub struct DriftEstimator;

impl DriftEstimator {
    /// Estimate expected drift given volatility and a time delta.
    ///
    /// Returns `0.0` when either input is non-positive.
    pub fn expected_drift(volatility: f64, delta_time: f64) -> f64 {
        if volatility <= 0.0 || delta_time <= 0.0 {
            return 0.0;
        }

        volatility * delta_time.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_volatility_returns_zero() {
        assert_eq!(DriftEstimator::expected_drift(0.0, 1.0), 0.0);
    }

    #[test]
    fn negative_volatility_returns_zero() {
        assert_eq!(DriftEstimator::expected_drift(-0.5, 1.0), 0.0);
    }

    #[test]
    fn zero_delta_time_returns_zero() {
        assert_eq!(DriftEstimator::expected_drift(0.2, 0.0), 0.0);
    }

    #[test]
    fn negative_delta_time_returns_zero() {
        assert_eq!(DriftEstimator::expected_drift(0.2, -1.0), 0.0);
    }

    #[test]
    fn positive_values() {
        let result = DriftEstimator::expected_drift(0.2, 4.0);
        // 0.2 * sqrt(4.0) = 0.2 * 2.0 = 0.4
        assert!((result - 0.4).abs() < 1e-10);
    }

    #[test]
    fn numerical_correctness() {
        let vol = 0.15;
        let dt = 1.0 / 252.0; // ~one trading day
        let result = DriftEstimator::expected_drift(vol, dt);
        let expected = vol * (1.0_f64 / 252.0).sqrt();
        assert!((result - expected).abs() < 1e-12);
    }
}

