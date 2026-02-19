use std::time::Duration;

/// Controls the cadence and timing of execution rounds.
pub struct Scheduler {
    interval: Duration,
}

impl Scheduler {
    pub fn new(interval: Duration) -> Self {
        Self { interval }
    }

    /// Returns the configured interval between rounds.
    pub fn interval(&self) -> Duration {
        self.interval
    }
}
