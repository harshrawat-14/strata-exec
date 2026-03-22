/// Empirical Square-Root Market Impact Model for research simulation.
///
/// Models price-impact according to the widely observed square-root law:
/// Impact = Y * \sigma * \sqrt{Q / V}
///
/// Where:
/// - Y: Impact coefficient (e.g. 0.5)
/// - \sigma: Annualized volatility
/// - Q: Trade quantity
/// - V: Average daily volume
pub struct SquareRootImpact {
    pub daily_volume: f64,
    pub impact_coefficient: f64,
}

impl SquareRootImpact {
    /// Create a new impact model with the given parameters.
    pub fn new(daily_volume: f64, impact_coefficient: f64) -> Self {
        Self {
            daily_volume,
            impact_coefficient,
        }
    }

    /// Compute the fractional price impact of a trade.
    ///
    /// Returns the fractional impact (e.g. 0.0015 for 15bps).
    /// The execution price will be `mid_price * (1.0 - impact)` for a sell,
    /// or `mid_price * (1.0 + impact)` for a buy.
    pub fn compute_impact(&self, quantity: f64, volatility: f64) -> f64 {
        if quantity <= 0.0 || self.daily_volume <= 0.0 {
            return 0.0;
        }

        let q_over_v = quantity / self.daily_volume;
        self.impact_coefficient * volatility * q_over_v.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impact_increases_with_square_root_of_trade_size() {
        let impact_model = SquareRootImpact::new(10_000_000.0, 0.5);
        let vol = 0.20; // 20% annualized volatility

        let impact_10k = impact_model.compute_impact(10_000.0, vol);
        let impact_40k = impact_model.compute_impact(40_000.0, vol);

        // sqrt(40k/10m) = 2 * sqrt(10k/10m)
        // So impact should effectively double.
        assert!(
            (impact_40k - (2.0 * impact_10k)).abs() < 1e-6,
            "impact nonlinear scaling failed: 10k={}, 40k={}",
            impact_10k,
            impact_40k
        );
    }

    #[test]
    fn execution_impact_is_zero_for_zero_trade() {
        let impact_model = SquareRootImpact::new(5_000_000.0, 0.5);
        assert_eq!(impact_model.compute_impact(0.0, 0.20), 0.0);
        assert_eq!(impact_model.compute_impact(-100.0, 0.20), 0.0);
    }
}
