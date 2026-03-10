/// Constant-product (x·y = k) automated market maker for research simulation.
///
/// Models the price-impact mechanics of a Uniswap V2–style pool so execution
/// strategies experience realistic, non-linear slippage.
pub struct ConstantProductAMM {
    pub reserve_x: f64,
    pub reserve_y: f64,
}

impl ConstantProductAMM {
    /// Create a new pool with the given reserves.
    pub fn new(reserve_x: f64, reserve_y: f64) -> Self {
        Self { reserve_x, reserve_y }
    }

    /// Marginal (spot) price: `reserve_y / reserve_x`.
    pub fn price(&self) -> f64 {
        self.reserve_y / self.reserve_x
    }

    /// Sell `dx` units of token X into the pool.
    ///
    /// Updates reserves in-place and returns the **execution price** (`dy / dx`).
    ///
    /// Returns `0.0` if `dx` is non-positive.
    pub fn sell_x(&mut self, dx: f64) -> f64 {
        if dx <= 0.0 {
            return 0.0;
        }

        let k = self.reserve_x * self.reserve_y;
        let new_x = self.reserve_x + dx;
        let new_y = k / new_x;
        let dy = self.reserve_y - new_y;

        self.reserve_x = new_x;
        self.reserve_y = new_y;

        dy / dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariant_preserved_after_trade() {
        let mut amm = ConstantProductAMM::new(1000.0, 100_000.0);
        let k_before = amm.reserve_x * amm.reserve_y;

        amm.sell_x(50.0);

        let k_after = amm.reserve_x * amm.reserve_y;
        assert!(
            (k_before - k_after).abs() < 1e-6,
            "invariant broken: k_before={k_before}, k_after={k_after}",
        );
    }

    #[test]
    fn price_impact_increases_with_trade_size() {
        // Larger trades should cause more price impact (worse execution price).
        let mut amm_small = ConstantProductAMM::new(1000.0, 100_000.0);
        let mut amm_large = ConstantProductAMM::new(1000.0, 100_000.0);

        let spot = amm_small.price();
        let exec_small = amm_small.sell_x(10.0);
        let exec_large = amm_large.sell_x(100.0);

        // Both execution prices should be below spot (selling X depresses Y/X).
        assert!(exec_small < spot, "exec_small {exec_small} should be < spot {spot}");
        assert!(exec_large < spot, "exec_large {exec_large} should be < spot {spot}");

        // Larger trade → worse execution price (lower dy/dx).
        assert!(
            exec_large < exec_small,
            "larger trade should get worse price: large={exec_large}, small={exec_small}",
        );
    }

    #[test]
    fn execution_price_decreases_for_larger_dx() {
        let reserves = (500.0, 50_000.0);
        let sizes = [1.0, 5.0, 25.0, 100.0];
        let mut prev_exec = f64::MAX;

        for &dx in &sizes {
            let mut amm = ConstantProductAMM::new(reserves.0, reserves.1);
            let exec = amm.sell_x(dx);
            assert!(
                exec < prev_exec,
                "exec price should decrease: dx={dx}, exec={exec}, prev={prev_exec}",
            );
            prev_exec = exec;
        }
    }

    #[test]
    fn zero_trade_returns_zero() {
        let mut amm = ConstantProductAMM::new(1000.0, 100_000.0);
        assert_eq!(amm.sell_x(0.0), 0.0);
        assert_eq!(amm.sell_x(-5.0), 0.0);
    }
}
