use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

// ---------------------------------------------------------------------------
// Simulator trait
// ---------------------------------------------------------------------------

/// Unified interface for price simulators (GBM, GARCH, etc.).
///
/// Every simulator must be able to advance by one time-step and report
/// its current volatility so that `AdaptiveOptimal` can react to
/// changing market conditions.
pub trait PriceSimulator {
    /// Advance by one time-step and return the new price.
    fn step(&mut self) -> f64;

    /// Current annualised volatility estimate.
    fn volatility(&self) -> f64;
}

/// Deterministic Geometric Brownian Motion price simulator.
///
/// Produces a reproducible price path when given a fixed seed, enabling
/// apples-to-apples comparison of execution strategies.
pub struct GbmSimulator {
    current_price: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    rng: StdRng,
}

impl GbmSimulator {
    /// Create a new GBM simulator.
    ///
    /// * `initial_price` — starting asset price.
    /// * `mu`            — annualised drift.
    /// * `sigma`         — annualised volatility.
    /// * `dt`            — time-step size (fraction of a year).
    /// * `seed`          — fixed RNG seed for reproducibility.
    pub fn new(initial_price: f64, mu: f64, sigma: f64, dt: f64, seed: u64) -> Self {
        Self {
            current_price: initial_price,
            mu,
            sigma,
            dt,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Advance the price by one time-step and return the new price.
    ///
    /// Formula: `S_next = S * exp((μ − 0.5·σ²)·dt + σ·√dt·Z)`
    /// where `Z ~ N(0, 1)`.
    pub fn next_price(&mut self) -> f64 {
        let z: f64 = self.rng.sample(StandardNormal);
        // Z ~ N(0,1) from StandardNormal distribution.

        let drift = (self.mu - 0.5 * self.sigma * self.sigma) * self.dt;
        let diffusion = self.sigma * self.dt.sqrt() * z;
        self.current_price *= (drift + diffusion).exp();
        self.current_price
    }

    /// Current price (read-only accessor).
    pub fn current_price(&self) -> f64 {
        self.current_price
    }
}

impl PriceSimulator for GbmSimulator {
    fn step(&mut self) -> f64 {
        self.next_price()
    }

    fn volatility(&self) -> f64 {
        self.sigma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let mut a = GbmSimulator::new(100.0, 0.05, 0.2, 1.0 / 252.0, 42);
        let mut b = GbmSimulator::new(100.0, 0.05, 0.2, 1.0 / 252.0, 42);

        for _ in 0..100 {
            let pa = a.next_price();
            let pb = b.next_price();
            assert!(
                (pa - pb).abs() < 1e-15,
                "paths should be identical: {pa} vs {pb}",
            );
        }
    }

    #[test]
    fn price_stays_positive() {
        let mut sim = GbmSimulator::new(100.0, -0.10, 0.5, 1.0 / 252.0, 99);
        for _ in 0..1000 {
            let p = sim.next_price();
            assert!(p > 0.0, "GBM price must stay positive, got {p}");
        }
    }
}
