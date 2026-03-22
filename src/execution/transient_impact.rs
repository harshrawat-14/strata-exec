/// Represents a single historical trade for transient impact tracking.
#[derive(Debug, Clone)]
pub struct TradeEvent {
    /// The simulation timestamp when this trade occurred.
    pub timestamp: f64,
    /// The instantaneous (base) fractional impact of this trade.
    pub base_impact: f64,
}

/// Tracks the lingering (transient) price impact of past trades.
///
/// Models the temporary deviation of the mid-price using an exponential decay function:
/// I(t) = Σ base_impact_i * exp(-ρ * (t - t_i))
///
/// Where `ρ` is the liquidity recovery rate.
pub struct TransientImpactTracker {
    history: Vec<TradeEvent>,
    /// The rate at which market impact decays over time (e.g. 50.0).
    rho: f64,
}

impl TransientImpactTracker {
    pub fn new(rho: f64) -> Self {
        Self {
            history: Vec::new(),
            rho,
        }
    }

    /// Record a new trade's impact and prune old trades whose impact has decayed to zero.
    ///
    /// `now` must be monotonic (>= all previous trade timestamps).
    pub fn record_trade(&mut self, now: f64, base_impact: f64) {
        if base_impact <= 0.0 {
            return;
        }

        self.history.push(TradeEvent {
            timestamp: now,
            base_impact,
        });

        // Prune the history to prevent unbounded growth in long Monte Carlo simulations.
        // If exp(-ρ * dt) < 1e-8, the impact is effectively zero.
        // dt > -ln(1e-8) / ρ
        let threshold_dt = -(1e-8f64).ln() / self.rho;
        self.history
            .retain(|evt| (now - evt.timestamp) < threshold_dt);
    }

    /// Compute the total decayed transient impact from all past trades at time `now`.
    pub fn current_impact(&self, now: f64) -> f64 {
        let mut total_impact = 0.0;
        for evt in &self.history {
            // Ensure we don't calculate future impact if events arrive slightly out of expected order.
            let dt = (now - evt.timestamp).max(0.0);
            total_impact += evt.base_impact * (-self.rho * dt).exp();
        }
        total_impact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_trade_decays_exponentially() {
        let mut tracker = TransientImpactTracker::new(10.0); // rho = 10
        tracker.record_trade(0.0, 0.01); // 100 bps impact at t=0

        // At t=0, impact should be instantly exactly the base.
        assert!((tracker.current_impact(0.0) - 0.01).abs() < 1e-9);

        // At t=0.1, dt=0.1. exp(-10 * 0.1) = exp(-1) = ~0.3678...
        let expected = 0.01 * (-1.0f64).exp();
        let actual = tracker.current_impact(0.1);
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn multiple_trades_accumulate() {
        let mut tracker = TransientImpactTracker::new(5.0);
        tracker.record_trade(1.0, 0.01);
        tracker.record_trade(2.0, 0.02);

        // Calculate impact exactly at t=2.0
        // Trade 1 decayed for 1 second: 0.01 * exp(-5 * 1.0)
        // Trade 2 occurred instantly: 0.02 * exp(0)
        let expected = 0.01 * (-5.0f64).exp() + 0.02;
        let actual = tracker.current_impact(2.0);

        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn tracker_prunes_old_trades() {
        // High decay rate so impact vanishes fast
        let mut tracker = TransientImpactTracker::new(100.0);

        // Record trade at t=0
        tracker.record_trade(0.0, 0.01);
        assert_eq!(tracker.history.len(), 1);

        // Wait a long time (relative to rho=100) and record a new trade
        // The old trade's impact will be far below the 1e-8 threshold.
        //
        // T = 5.0 -> dt = 5.0 -> exp(-500) = 0.0
        tracker.record_trade(5.0, 0.01);

        // The old trade should have been removed from the history array.
        assert_eq!(tracker.history.len(), 1, "History array was not pruned!");
        assert_eq!(tracker.history[0].timestamp, 5.0);
    }
}
