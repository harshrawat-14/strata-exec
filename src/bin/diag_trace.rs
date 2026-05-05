/// Diagnostic: per-step trace of TWAP vs AC on a 20-step GARCH path.
///
/// Prints: Step | AC_chunk | TWAP_chunk | AC_price_adj | TWAP_price_adj |
///         AC_eff_price | TWAP_eff_price | AC_total_traded | TWAP_total_traded
///
/// Run with: cargo run --bin diag_trace
use strata_exec::analytics::metrics::ExecutionMetrics;
use strata_exec::engine::risk::RiskManager;
use strata_exec::engine::scheduler::{ExecutionMode, Scheduler};
use strata_exec::execution::impact::SquareRootImpact;
use strata_exec::execution::transient_impact::TransientImpactTracker;
use strata_exec::market::garch::GarchSimulator;
use strata_exec::market::gbm::PriceSimulator;
use strata_exec::market::liquidity::LiquiditySnapshot;

fn main() {
    let max_steps = 20;
    let total_notional = 100_000.0_f64;
    let arrival_price = 100.0_f64;
    let sigma_ref = 0.02_f64;
    let eta = 0.001_f64;
    let lambda = 1e-4_f64;
    let gamma = 0.005_f64;
    let dt = 1.0 / max_steps as f64;

    // --- Build simulator ---
    let mut sim = GarchSimulator::new(
        arrival_price,
        0.05,
        0.000002,
        0.08,
        0.90,
        sigma_ref * sigma_ref,
        dt,
        42,
    );

    // --- Build TWAP scheduler ---
    let twap_chunk = total_notional / max_steps as f64;
    let mut twap_sched = Scheduler::new(
        RiskManager::new(total_notional, 0.99, 0.0),
        total_notional,
        twap_chunk,
        sigma_ref,
        500.0,
        ExecutionMode::Twap,
        eta,
        lambda,
        Some(max_steps),
    );
    let mut twap_metrics = ExecutionMetrics::new(arrival_price, total_notional);
    let mut twap_impact_model = SquareRootImpact::new(5_000_000.0, 0.5);
    let mut twap_transient = TransientImpactTracker::new(50.0);
    let mut twap_price_adj = 0.0_f64;
    let mut twap_total_traded = 0.0_f64;

    // --- Build AC (Optimal) scheduler ---
    let ac_chunk = total_notional / max_steps as f64;
    let mut ac_sched = Scheduler::new(
        RiskManager::new(total_notional, 0.99, 0.0),
        total_notional,
        ac_chunk,
        sigma_ref,
        500.0,
        ExecutionMode::Optimal,
        eta,
        lambda,
        Some(max_steps),
    );
    let mut ac_metrics = ExecutionMetrics::new(arrival_price, total_notional);
    let mut ac_impact_model = SquareRootImpact::new(5_000_000.0, 0.5);
    let mut ac_transient = TransientImpactTracker::new(50.0);
    let mut ac_price_adj = 0.0_f64;
    let mut ac_total_traded = 0.0_f64;

    let total_notional_ref = total_notional;

    println!(
        "{:>4} | {:>12} | {:>12} | {:>13} | {:>14} | {:>15} | {:>16} | {:>16} | {:>17}",
        "Step",
        "AC_chunk",
        "TWAP_chunk",
        "AC_price_adj",
        "TWAP_price_adj",
        "AC_eff_price",
        "TWAP_eff_price",
        "AC_total_traded",
        "TWAP_total_traded",
    );
    println!("{}", "-".repeat(140));

    // Snapshot totals at steps 5, 10, 15, 20.
    let mut snapshots: Vec<(usize, f64, f64)> = Vec::new();

    for step in 1..=max_steps {
        let price = sim.step();
        let vol = sim.volatility();
        let liquidity = LiquiditySnapshot::new(500.0, 500.0);
        let real_time = step as f64 * dt;

        // --- TWAP step ---
        let twap_before = twap_sched.remaining_notional();
        twap_sched.on_block(vol, &liquidity).unwrap();
        let twap_after = twap_sched.remaining_notional();
        let twap_traded = (twap_before - twap_after).max(0.0);
        let twap_eff = price * (1.0 - twap_price_adj);

        if twap_traded > 0.0 {
            let base_imp = twap_impact_model.compute_impact(twap_traded, vol);
            let trans_imp = twap_transient.current_impact(real_time);
            let exec_price = twap_eff * (1.0 - base_imp - trans_imp);
            let slippage = exec_price - twap_eff;
            let twap_delta_perm = gamma * (twap_traded / total_notional_ref);
            twap_price_adj += twap_delta_perm;
            twap_transient.record_trade(real_time, base_imp);
            // Half-spread proxy: 0.5 × σ × √dt × price.
            // TODO Phase 2 Track B — replace with real LOB spread.
            let half_spread = 0.5 * vol * dt.sqrt() * twap_eff;
            twap_metrics.record_trade(twap_traded, twap_eff, slippage, trans_imp, twap_delta_perm, price, half_spread);
            twap_metrics.record_risk_step(twap_before, vol, dt);
            twap_total_traded += twap_traded;
        }

        // --- AC step ---
        let ac_before = ac_sched.remaining_notional();
        ac_sched.on_block(vol, &liquidity).unwrap();
        let ac_after = ac_sched.remaining_notional();
        let ac_traded = (ac_before - ac_after).max(0.0);
        let ac_eff = price * (1.0 - ac_price_adj);

        if ac_traded > 0.0 {
            let base_imp = ac_impact_model.compute_impact(ac_traded, vol);
            let trans_imp = ac_transient.current_impact(real_time);
            let exec_price = ac_eff * (1.0 - base_imp - trans_imp);
            let slippage = exec_price - ac_eff;
            let ac_delta_perm = gamma * (ac_traded / total_notional_ref);
            ac_price_adj += ac_delta_perm;
            ac_transient.record_trade(real_time, base_imp);
            // Half-spread proxy: 0.5 × σ × √dt × price.
            // TODO Phase 2 Track B — replace with real LOB spread.
            let half_spread = 0.5 * vol * dt.sqrt() * ac_eff;
            ac_metrics.record_trade(ac_traded, ac_eff, slippage, trans_imp, ac_delta_perm, price, half_spread);
            ac_metrics.record_risk_step(ac_before, vol, dt);
            ac_total_traded += ac_traded;
        }

        println!(
            "{:>4} | {:>12.2} | {:>12.2} | {:>13.6} | {:>14.6} | {:>15.6} | {:>16.6} | {:>16.2} | {:>17.2}",
            step,
            ac_traded,
            twap_traded,
            ac_price_adj,
            twap_price_adj,
            ac_eff,
            twap_eff,
            ac_total_traded,
            twap_total_traded,
        );

        if [5, 10, 15, 20].contains(&step) {
            snapshots.push((step, ac_total_traded, twap_total_traded));
        }
    }

    println!();
    println!("=== Quantity traded snapshots ===");
    println!("{:>4} | {:>16} | {:>17}", "Step", "AC_total_traded", "TWAP_total_traded");
    for (step, ac, twap) in &snapshots {
        println!("{:>4} | {:>16.2} | {:>17.2}", step, ac, twap);
    }

    println!();
    println!("=== Summary ===");
    println!(
        "AC final IS:   {:.6}%",
        ac_metrics.shortfall_percent().unwrap_or(f64::NAN)
    );
    println!(
        "TWAP final IS: {:.6}%",
        twap_metrics.shortfall_percent().unwrap_or(f64::NAN)
    );
    println!("AC   final price_adj: {:.6}", ac_price_adj);
    println!("TWAP final price_adj: {:.6}", twap_price_adj);
}
