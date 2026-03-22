/// Multi-constraint slippage risk manager.
///
/// Enforces four invariants on every execution chunk:
///
/// 1. **Global cap** — cumulative slippage never exceeds `max_slippage_budget`.
/// 2. **Running-average** — average slippage rate stays within `max_slippage_pct`.
/// 3. **Proportional allocation** — each chunk gets a fair share of the budget.
/// 4. **Emergency reserve** — a fraction of the budget held back for overruns.
pub struct RiskManager {
    total_notional: f64,
    max_slippage_pct: f64,
    max_slippage_budget: f64,
    executed_notional: f64,
    used_slippage: f64,
    emergency_reserve: f64,
    emergency_used: f64,
}

impl RiskManager {
    pub fn new(total_notional: f64, max_slippage_pct: f64, reserve_frac: f64) -> Self {
        let max_slippage_budget = total_notional * max_slippage_pct;
        let emergency_reserve = reserve_frac * max_slippage_budget;

        Self {
            total_notional,
            max_slippage_pct,
            max_slippage_budget,
            executed_notional: 0.0,
            used_slippage: 0.0,
            emergency_reserve,
            emergency_used: 0.0,
        }
    }

    /// Absolute remaining slippage budget (never negative).
    pub fn remaining_budget(&self) -> f64 {
        (self.max_slippage_budget - self.used_slippage).max(0.0)
    }

    /// Proportional share of the total budget for a chunk of the given size.
    pub fn per_chunk_allocation(&self, chunk_size: f64) -> f64 {
        if self.total_notional <= 0.0 {
            return 0.0;
        }
        (chunk_size / self.total_notional) * self.max_slippage_budget
    }

    /// Running-average headroom: how much slippage can be added while
    /// keeping the average rate at or below `max_slippage_pct`.
    pub fn cap_running(&self, chunk_size: f64) -> f64 {
        let cap =
            self.max_slippage_pct * (self.executed_notional + chunk_size) - self.used_slippage;
        cap.max(0.0)
    }

    /// The tightest constraint across all four limits for the next chunk.
    ///
    /// The per-chunk ceiling is `proportional + remaining_emergency`, which
    /// allows chunks to overshoot their proportional share up to the
    /// emergency reserve.  That ceiling then competes with the running-
    /// average cap and the absolute remaining budget.
    pub fn allowed_for_next(&self, chunk_size: f64) -> f64 {
        let remaining_emergency = (self.emergency_reserve - self.emergency_used).max(0.0);
        let per_chunk_ceiling = self.per_chunk_allocation(chunk_size) + remaining_emergency;

        per_chunk_ceiling
            .min(self.cap_running(chunk_size))
            .min(self.remaining_budget())
    }

    /// Attempt to book a chunk execution with the predicted slippage.
    ///
    /// On success the internal accumulators are updated atomically.
    /// On failure the state is unchanged.
    pub fn request_execute_chunk(
        &mut self,
        chunk_size: f64,
        predicted_slippage: f64,
    ) -> Result<(), String> {
        let allowed = self.allowed_for_next(chunk_size);

        if predicted_slippage > allowed {
            return Err(format!(
                "slippage {predicted_slippage:.6} exceeds allowed budget {allowed:.6}"
            ));
        }

        self.executed_notional += chunk_size;
        self.used_slippage += predicted_slippage;

        // If the chunk exceeded its proportional share, draw from emergency.
        let proportional = self.per_chunk_allocation(chunk_size);
        if predicted_slippage > proportional {
            let draw = predicted_slippage - proportional;
            self.emergency_used += draw;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a manager with 1000 notional, 1% slippage, 20% reserve.
    fn default_rm() -> RiskManager {
        RiskManager::new(1000.0, 0.01, 0.2)
        // budget = 10.0, emergency_reserve = 2.0
    }

    #[test]
    fn global_cap_never_exceeded() {
        let mut rm = default_rm();
        // Exhaust the full budget in one go.
        assert!(rm.request_execute_chunk(1000.0, 10.0).is_ok());
        assert_eq!(rm.remaining_budget(), 0.0);
        // Any further slippage must be denied.
        assert!(rm.request_execute_chunk(1.0, 0.001).is_err());
    }

    #[test]
    fn running_average_enforced() {
        let mut rm = default_rm();
        // Execute half the notional using exactly its proportional share.
        assert!(rm.request_execute_chunk(500.0, 5.0).is_ok());
        // Next chunk of 500 — running cap = 0.01*(500+500) - 5.0 = 5.0
        // Trying 5.01 should fail.
        assert!(rm.request_execute_chunk(500.0, 5.01).is_err());
        // Exactly 5.0 should pass.
        assert!(rm.request_execute_chunk(500.0, 5.0).is_ok());
    }

    #[test]
    fn proportional_allocation_behavior() {
        let rm = default_rm();
        // 100 out of 1000 → 10% of budget → 1.0
        let alloc = rm.per_chunk_allocation(100.0);
        assert!((alloc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn emergency_draw_behavior() {
        let mut rm = default_rm();
        // budget = 10.0, emergency_reserve = 2.0

        // First chunk: 500 notional at 4.0 slippage (under proportional 5.0).
        // Builds running-average headroom for the next chunk.
        assert!(rm.request_execute_chunk(500.0, 4.0).is_ok());

        // Second chunk: 100 notional.
        //   proportional   = (100/1000)*10 = 1.0
        //   cap_running    = 0.01*(500+100) - 4.0 = 2.0
        //   remaining      = 10.0 - 4.0 = 6.0
        //   per_chunk_ceil = 1.0 + 2.0 = 3.0
        //   allowed        = min(3.0, 2.0, 6.0) = 2.0
        // Use slippage 1.5 → proportional overshoot = 0.5 drawn from emergency.
        assert!(rm.request_execute_chunk(100.0, 1.5).is_ok());
        assert!((rm.emergency_used - 0.5).abs() < 1e-10);

        let remaining_emergency = rm.emergency_reserve - rm.emergency_used;
        assert!((remaining_emergency - 1.5).abs() < 1e-10);
    }

    #[test]
    fn denial_when_budget_exhausted() {
        let mut rm = default_rm();
        // Drain budget across many small chunks.
        for _ in 0..10 {
            assert!(rm.request_execute_chunk(100.0, 1.0).is_ok());
        }
        assert_eq!(rm.remaining_budget(), 0.0);
        assert!(rm.request_execute_chunk(100.0, 0.1).is_err());
    }

    #[test]
    fn scenario_600_divided_into_50_chunks() {
        // Total notional 600, 1% slippage, 20% emergency reserve.
        // Budget = 6.0, emergency = 1.2
        let mut rm = RiskManager::new(600.0, 0.01, 0.2);
        let chunk = 600.0 / 12.0; // 50.0 per chunk
        let per_chunk_slip = 0.01 * chunk; // 0.5 per chunk — exactly proportional

        for _ in 0..12 {
            assert!(rm.request_execute_chunk(chunk, per_chunk_slip).is_ok());
        }

        assert!((rm.executed_notional - 600.0).abs() < 1e-10);
        assert!((rm.used_slippage - 6.0).abs() < 1e-10);
        assert_eq!(rm.remaining_budget(), 0.0);
        // No emergency drawn — each chunk was exactly proportional.
        assert!((rm.emergency_used - 0.0).abs() < 1e-10);
    }
}
