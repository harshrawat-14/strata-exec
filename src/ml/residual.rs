/// Computes residual signal after subtracting model predictions.
pub struct ResidualModel {
    alpha: f64,
}

impl ResidualModel {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Returns the learning rate (alpha) of the model.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}
