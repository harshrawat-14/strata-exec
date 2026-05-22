/// Volatility regime relative to a reference level observed at run start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VolRegime {
    /// sigma < 0.6 × sigma_ref
    Low,
    /// 0.6 × sigma_ref <= sigma <= 1.4 × sigma_ref
    Normal,
    /// sigma > 1.4 × sigma_ref
    High,
}

/// Liquidity regime relative to the first LOB snapshot's half-spread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiqRegime {
    /// lob_half_spread < 0.7 × spread_ref
    Deep,
    /// 0.7 × spread_ref <= lob_half_spread <= 1.3 × spread_ref
    Normal,
    /// lob_half_spread > 1.3 × spread_ref
    Thin,
}

/// Joint vol × liquidity regime with 9 possible states.
#[derive(Debug, Clone)]
pub struct MarketRegime {
    pub vol: VolRegime,
    pub liq: LiqRegime,
}

impl MarketRegime {
    /// Classify the current market state into a joint regime.
    ///
    /// * `sigma`      — current annualised volatility estimate.
    /// * `sigma_ref`  — baseline volatility at run start.
    /// * `half_spread` — current LOB half-spread in price units.
    /// * `spread_ref` — half-spread from the first LOB snapshot.
    pub fn classify(sigma: f64, sigma_ref: f64, half_spread: f64, spread_ref: f64) -> Self {
        let vol = if sigma < 0.6 * sigma_ref {
            VolRegime::Low
        } else if sigma > 1.4 * sigma_ref {
            VolRegime::High
        } else {
            VolRegime::Normal
        };

        let liq = if half_spread < 0.7 * spread_ref {
            LiqRegime::Deep
        } else if half_spread > 1.3 * spread_ref {
            LiqRegime::Thin
        } else {
            LiqRegime::Normal
        };

        Self { vol, liq }
    }

    /// Human-readable label for this regime (9 distinct strings).
    pub fn label(&self) -> &'static str {
        match (&self.vol, &self.liq) {
            (VolRegime::Low,    LiqRegime::Deep)   => "low-vol/deep",
            (VolRegime::Low,    LiqRegime::Normal) => "low-vol/normal",
            (VolRegime::Low,    LiqRegime::Thin)   => "low-vol/thin",
            (VolRegime::Normal, LiqRegime::Deep)   => "normal/deep",
            (VolRegime::Normal, LiqRegime::Normal) => "normal/normal",
            (VolRegime::Normal, LiqRegime::Thin)   => "normal/thin",
            (VolRegime::High,   LiqRegime::Deep)   => "high-vol/deep",
            (VolRegime::High,   LiqRegime::Normal) => "high-vol/normal",
            (VolRegime::High,   LiqRegime::Thin)   => "high-vol/thin",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_vol_classified_above_threshold() {
        // sigma = 1.5 × sigma_ref → High
        let r = MarketRegime::classify(0.30, 0.20, 1.0, 1.0);
        assert_eq!(r.vol, VolRegime::High);
    }

    #[test]
    fn low_vol_classified_below_threshold() {
        // sigma = 0.5 × sigma_ref → Low
        let r = MarketRegime::classify(0.10, 0.20, 1.0, 1.0);
        assert_eq!(r.vol, VolRegime::Low);
    }

    #[test]
    fn thin_liq_classified_above_threshold() {
        // half_spread = 1.4 × spread_ref → Thin
        let r = MarketRegime::classify(0.20, 0.20, 1.4, 1.0);
        assert_eq!(r.liq, LiqRegime::Thin);
    }

    #[test]
    fn deep_liq_classified_below_threshold() {
        // half_spread = 0.6 × spread_ref → Deep
        let r = MarketRegime::classify(0.20, 0.20, 0.6, 1.0);
        assert_eq!(r.liq, LiqRegime::Deep);
    }

    #[test]
    fn normal_regime_at_reference_values() {
        // sigma == sigma_ref, half_spread == spread_ref → both Normal
        let r = MarketRegime::classify(0.20, 0.20, 1.0, 1.0);
        assert_eq!(r.vol, VolRegime::Normal);
        assert_eq!(r.liq, LiqRegime::Normal);
        assert_eq!(r.label(), "normal/normal");
    }

    #[test]
    fn all_nine_combinations_produce_distinct_labels() {
        let sigmas = [
            (0.10_f64, 0.20_f64), // Low
            (0.20, 0.20),          // Normal
            (0.30, 0.20),          // High
        ];
        let spreads = [
            (0.6_f64, 1.0_f64), // Deep
            (1.0, 1.0),          // Normal
            (1.4, 1.0),          // Thin (> 1.3)
        ];

        let mut labels = std::collections::HashSet::new();
        for (sigma, sigma_ref) in &sigmas {
            for (hs, sr) in &spreads {
                let r = MarketRegime::classify(*sigma, *sigma_ref, *hs, *sr);
                labels.insert(r.label());
            }
        }
        assert_eq!(labels.len(), 9, "expected 9 distinct labels, got {labels:?}");
    }
}
