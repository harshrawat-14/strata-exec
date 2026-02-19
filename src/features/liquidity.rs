/// Point-in-time snapshot of a liquidity pool's reserves.
pub struct LiquiditySnapshot {
    reserve_in: f64,
    reserve_out: f64,
}

impl LiquiditySnapshot {
    pub fn new(reserve_in: f64, reserve_out: f64) -> Self {
        Self {
            reserve_in,
            reserve_out,
        }
    }

    /// Mid-market price derived from the constant-product invariant.
    ///
    /// Returns `None` when either reserve is non-positive.
    pub fn mid_price(&self) -> Option<f64> {
        if self.reserve_in <= 0.0 || self.reserve_out <= 0.0 {
            return None;
        }

        Some(self.reserve_out / self.reserve_in)
    }

    /// Geometric mean of reserves — a proxy for pool depth.
    ///
    /// Returns `0.0` when either reserve is non-positive.
    pub fn depth_metric(&self) -> f64 {
        if self.reserve_in <= 0.0 || self.reserve_out <= 0.0 {
            return 0.0;
        }

        (self.reserve_in * self.reserve_out).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_reserves_mid_price() {
        let snap = LiquiditySnapshot::new(100.0, 200.0);
        let price = snap.mid_price().expect("should have price");
        assert!((price - 2.0).abs() < 1e-10);
    }

    #[test]
    fn valid_reserves_depth_metric() {
        let snap = LiquiditySnapshot::new(100.0, 400.0);
        let depth = snap.depth_metric();
        // sqrt(100 * 400) = sqrt(40000) = 200
        assert!((depth - 200.0).abs() < 1e-10);
    }

    #[test]
    fn zero_reserve_in_returns_none() {
        let snap = LiquiditySnapshot::new(0.0, 200.0);
        assert!(snap.mid_price().is_none());
        assert_eq!(snap.depth_metric(), 0.0);
    }

    #[test]
    fn zero_reserve_out_returns_none() {
        let snap = LiquiditySnapshot::new(100.0, 0.0);
        assert!(snap.mid_price().is_none());
        assert_eq!(snap.depth_metric(), 0.0);
    }

    #[test]
    fn negative_reserve_in_returns_none() {
        let snap = LiquiditySnapshot::new(-50.0, 200.0);
        assert!(snap.mid_price().is_none());
        assert_eq!(snap.depth_metric(), 0.0);
    }

    #[test]
    fn negative_reserve_out_returns_none() {
        let snap = LiquiditySnapshot::new(100.0, -10.0);
        assert!(snap.mid_price().is_none());
        assert_eq!(snap.depth_metric(), 0.0);
    }
}

