/// Reusable distribution statistics for Monte Carlo analysis.
///
/// Computes full risk-sensitive summary statistics from a sample of `f64`
/// values, including VaR, CVaR, skewness, and excess kurtosis.

/// Summary statistics computed from a sample of values.
#[derive(Debug, Clone)]
pub struct DistributionStats {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub p90: f64,
    pub p99: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    pub var_99: f64,
    pub cvar_99: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub max: f64,
    pub min: f64,
}

impl DistributionStats {
    /// Compute distribution statistics from a slice of samples.
    ///
    /// The input slice is **not** mutated; values are copied and sorted
    /// internally.  Returns `None` if the input is empty.
    ///
    /// ## Formulas
    ///
    /// - **VaR α** = `percentile(α)` of the cost distribution.
    /// - **CVaR α** = mean of all values ≥ VaR α (expected tail loss).
    /// - **Skewness** = `E[(X − μ)³] / σ³`  (population skewness).
    /// - **Kurtosis** = `E[(X − μ)⁴] / σ⁴ − 3`  (excess kurtosis; 0 for normal).
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();
        let nf = n as f64;
        let mean = samples.iter().sum::<f64>() / nf;

        // Single-pass moment accumulation (population moments).
        let (mut m2, mut m3, mut m4) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &x in samples {
            let d = x - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }

        let variance = m2 / nf;
        let std_dev = variance.sqrt();

        let skewness = if std_dev > 1e-15 {
            (m3 / nf) / (std_dev * std_dev * std_dev)
        } else {
            0.0
        };

        let kurtosis = if std_dev > 1e-15 {
            (m4 / nf) / (variance * variance) - 3.0
        } else {
            0.0
        };

        // Sort a copy for order statistics.
        let mut sorted: Vec<f64> = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = percentile_sorted(&sorted, 50.0);
        let p90 = percentile_sorted(&sorted, 90.0);
        let p99 = percentile_sorted(&sorted, 99.0);
        let var_95 = percentile_sorted(&sorted, 95.0);
        let var_99 = percentile_sorted(&sorted, 99.0);
        let cvar_95 = cvar_sorted(&sorted, 95.0);
        let cvar_99 = cvar_sorted(&sorted, 99.0);
        let max = sorted[n - 1];
        let min = sorted[0];

        Some(Self {
            count: n,
            mean,
            std_dev,
            median,
            p90,
            p99,
            var_95,
            cvar_95,
            var_99,
            cvar_99,
            skewness,
            kurtosis,
            max,
            min,
        })
    }
}

/// Compute the `p`-th percentile from a **pre-sorted** slice using
/// linear interpolation (the "inclusive" / NIST method).
///
/// `p` is in `[0, 100]`.  For a single element, all percentiles
/// return that element.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty());

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    // Rank index (0-based continuous).
    let rank = (p / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil().min((n - 1) as f64) as usize;
    let frac = rank - lo as f64;

    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Compute CVaR (Conditional Value-at-Risk) at level `alpha` from a
/// **pre-sorted** slice.
///
/// CVaR α = mean of all samples whose value ≥ the α-th percentile.
/// This is the expected shortfall in the tail.
fn cvar_sorted(sorted: &[f64], alpha: f64) -> f64 {
    debug_assert!(!sorted.is_empty());

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    let threshold = percentile_sorted(sorted, alpha);

    // Average all values at or above the threshold.
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in sorted {
        if v >= threshold {
            sum += v;
            count += 1;
        }
    }

    if count == 0 {
        threshold
    } else {
        sum / count as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_none() {
        assert!(DistributionStats::from_samples(&[]).is_none());
    }

    #[test]
    fn single_element() {
        let stats = DistributionStats::from_samples(&[42.0]).unwrap();
        assert!((stats.mean - 42.0).abs() < 1e-10);
        assert!((stats.median - 42.0).abs() < 1e-10);
        assert!((stats.p90 - 42.0).abs() < 1e-10);
        assert!((stats.p99 - 42.0).abs() < 1e-10);
        assert!((stats.var_95 - 42.0).abs() < 1e-10);
        assert!((stats.cvar_95 - 42.0).abs() < 1e-10);
        assert!((stats.max - 42.0).abs() < 1e-10);
        assert!((stats.std_dev).abs() < 1e-10);
        assert!((stats.skewness).abs() < 1e-10);
        assert!((stats.kurtosis).abs() < 1e-10);
    }

    #[test]
    fn known_distribution() {
        // 1..=100 → mean = 50.5, median = 50.5
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let stats = DistributionStats::from_samples(&samples).unwrap();

        assert!((stats.mean - 50.5).abs() < 1e-10, "mean = {}", stats.mean);
        assert!(
            (stats.median - 50.5).abs() < 1e-10,
            "median = {}",
            stats.median
        );
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 100.0).abs() < 1e-10);
        assert_eq!(stats.count, 100);

        // P90 of 1..100 with linear interp: rank = 0.9 * 99 = 89.1
        // = sorted[89] + 0.1 * (sorted[90] - sorted[89]) = 90 + 0.1 = 90.1
        assert!((stats.p90 - 90.1).abs() < 1e-10, "p90 = {}", stats.p90);

        // VaR 95: rank = 0.95 * 99 = 94.05 → 95 + 0.05 = 95.05
        assert!(
            (stats.var_95 - 95.05).abs() < 1e-10,
            "var_95 = {}",
            stats.var_95
        );
    }

    #[test]
    fn std_dev_correct() {
        let samples = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = DistributionStats::from_samples(&samples).unwrap();
        // mean = 5.0, variance = 4.0, std = 2.0
        assert!((stats.mean - 5.0).abs() < 1e-10);
        assert!(
            (stats.std_dev - 2.0).abs() < 1e-10,
            "std = {}",
            stats.std_dev
        );
    }

    #[test]
    fn percentile_interpolation() {
        // 5 elements: [10, 20, 30, 40, 50]
        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = DistributionStats::from_samples(&samples).unwrap();

        // median: rank = 0.5 * 4 = 2.0 → sorted[2] = 30
        assert!(
            (stats.median - 30.0).abs() < 1e-10,
            "median = {}",
            stats.median
        );
    }

    #[test]
    fn cvar_is_tail_mean() {
        // 10 elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        // VaR 90 = percentile(90) = rank=0.9*9=8.1 → 9+0.1=9.1
        // CVaR 90 = mean of {9.1-eligible: 10} → values ≥ 9.1 → just 10 → 10.0
        let samples: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let stats = DistributionStats::from_samples(&samples).unwrap();

        assert!(
            (stats.cvar_99 - 10.0).abs() < 1e-10,
            "cvar_99 = {}",
            stats.cvar_99
        );
    }

    #[test]
    fn symmetric_has_zero_skewness() {
        // Symmetric distribution → skewness ≈ 0
        let samples: Vec<f64> = (-50..=50).map(|x| x as f64).collect();
        let stats = DistributionStats::from_samples(&samples).unwrap();
        assert!(
            stats.skewness.abs() < 1e-10,
            "symmetric distribution should have skewness ≈ 0, got {}",
            stats.skewness
        );
    }

    #[test]
    fn normal_has_near_zero_excess_kurtosis() {
        // Uniform distribution has excess kurtosis = -1.2
        let samples: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
        let stats = DistributionStats::from_samples(&samples).unwrap();
        assert!(
            (stats.kurtosis - (-1.2)).abs() < 0.01,
            "uniform excess kurtosis should be ≈ -1.2, got {}",
            stats.kurtosis
        );
    }
}
