/// Integration tests: calibration pipeline end-to-end.
///
/// Verifies that fitting an impact coefficient from synthetic trade data actually
/// changes simulation outputs compared to running with the default coefficient.
///
/// No real historical file is required — each test writes a temp CSV and cleans up.
use strata_exec::analytics::metrics::ExecutionMetrics;
use strata_exec::calibration::impact_fit::CalibrationEngine;
use strata_exec::engine::risk::RiskManager;
use strata_exec::engine::scheduler::{ExecutionMode, Scheduler};
use strata_exec::execution::impact::SquareRootImpact;
use strata_exec::execution::transient_impact::TransientImpactTracker;
use strata_exec::market::gbm::GbmSimulator;
use strata_exec::research::multi_runner::{MultiStrategyRunner, StrategyInstance};

use std::io::Write;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Write a synthetic CSV of historical trades and return the file path.
///
/// Each trade is constructed so the calibrated Y equals `target_y` exactly:
///   impact_fraction = target_y * sigma * sqrt(Q/V)
fn write_synthetic_trades_csv(target_y: f64, daily_volume: f64, sigma: f64) -> String {
    let path = std::env::temp_dir()
        .join(format!("strata_test_trades_{}.csv", std::process::id()))
        .to_string_lossy()
        .to_string();

    let mut file = std::fs::File::create(&path).expect("create temp CSV");
    writeln!(file, "timestamp_sec,price_before,price_after,trade_size")
        .expect("write CSV header");

    // Generate 5 synthetic trades at different sizes, all implying Y = target_y.
    // impact_fraction = target_y * sigma * sqrt(Q/V)
    // price_after = price_before * (1 + impact_fraction) for a buy, or exact maths for calcs.
    let price_before = 100.0_f64;
    let trade_sizes = [10_000.0, 25_000.0, 50_000.0, 100_000.0, 200_000.0];
    for (i, &q) in trade_sizes.iter().enumerate() {
        let impact_frac = target_y * sigma * (q / daily_volume).sqrt();
        let price_after = price_before * (1.0 + impact_frac);
        writeln!(
            file,
            "{},{:.6},{:.6},{:.6}",
            i + 1,
            price_before,
            price_after,
            q,
        )
        .expect("write CSV row");
    }

    path
}

fn run_twap(impact_coeff: f64, steps: usize) -> f64 {
    let total_notional = 100_000.0;
    let arrival_price = 100.0;
    let sigma_ref = 0.02;
    let daily_volume = 5_000_000.0;
    let base_chunk = total_notional / steps as f64;

    let strat = StrategyInstance {
        name: "TWAP".to_string(),
        scheduler: Scheduler::new(
            RiskManager::new(total_notional, 0.99, 0.0),
            total_notional,
            base_chunk,
            sigma_ref,
            500.0,
            ExecutionMode::Twap,
            0.001,
            1e-4,
            Some(steps),
        ),
        metrics: ExecutionMetrics::new(arrival_price, total_notional),
        impact_model: SquareRootImpact::new(daily_volume, impact_coeff),
        transient_tracker: TransientImpactTracker::new(50.0),
        gamma: 0.0,
        completed: false,
    };

    let dt = 1.0 / steps as f64;
    let sim = Box::new(GbmSimulator::new(arrival_price, 0.05, sigma_ref, dt, 42));
    let mut runner = MultiStrategyRunner::new(vec![strat], sim, steps);
    runner.run().expect("runner must succeed");

    runner.strategies()[0]
        .metrics
        .shortfall_percent()
        .expect("must have shortfall_percent")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// WHAT: Calibrating from synthetic trade data and running with the fitted
///       impact coefficient produces a different IS than running with the default.
/// WHY: The calibration pipeline must actually change outputs — if fitted and
///      default coefficients produce identical IS, calibration is a no-op and
///      the pipeline is broken.
#[test]
fn calibrated_gamma_changes_shortfall() {
    let daily_volume = 5_000_000.0;
    let sigma = 0.02;
    let target_y = 2.0; // deliberately far from the default of 0.5
    let default_y = 0.5; // matches SquareRootImpact default in research_sim

    // Write synthetic trades and calibrate.
    let csv_path = write_synthetic_trades_csv(target_y, daily_volume, sigma);
    let engine = CalibrationEngine::new(daily_volume, sigma);
    let trades = CalibrationEngine::load_csv(&csv_path)
        .expect("must load synthetic CSV");
    let fitted_y = engine.calibrate(&trades).expect("calibration must return a coefficient");

    // Clean up temp file.
    let _ = std::fs::remove_file(&csv_path);

    // The calibrated Y must be close to the target we encoded.
    assert!(
        (fitted_y - target_y).abs() < 0.01,
        "calibrated Y={fitted_y:.4} must be close to target Y={target_y}",
    );

    // Fitted coefficient must differ from the default by enough to matter.
    assert!(
        (fitted_y - default_y).abs() > 0.1,
        "fitted Y={fitted_y:.4} must differ from default Y={default_y} by >0.1 for a meaningful test",
    );

    // Run simulation with both coefficients on the same path.
    let steps = 500;
    let is_default = run_twap(default_y, steps);
    let is_fitted = run_twap(fitted_y, steps);

    // The shortfalls must differ — calibration must change outputs.
    assert!(
        (is_default - is_fitted).abs() > 1e-6,
        "IS with default Y={is_default:.6}% and fitted Y={is_fitted:.6}% are identical; \
         calibration pipeline is not affecting simulation outputs",
    );
}
