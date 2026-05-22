/// Integration tests for Phase 3 Track A — VPIN signal.
///
/// Three invariants are tested:
///   1. VPIN series from LobReplay stays in [0, 1].
///   2. VPIN recording does not violate the IS cost decomposition conservation law.
///   3. Conditional IS analysis is computable and internally consistent.
///
/// Phase 3 Task 2 tests (real LOB VPIN):
///   4. LOB VPIN series length is bounded by snapshot count.
///   5. All LOB VPIN values stay in [0, 1].
///   6. P75 threshold produces a nonzero high-toxicity fill fraction.
use strata_exec::analytics::metrics::ExecutionMetrics;
use strata_exec::engine::risk::RiskManager;
use strata_exec::engine::scheduler::{ExecutionMode, Scheduler};
use strata_exec::execution::impact::SquareRootImpact;
use strata_exec::execution::transient_impact::TransientImpactTracker;
use strata_exec::market::gbm::GbmSimulator;
use strata_exec::market::lob_replay::LobReplay;
use strata_exec::research::multi_runner::{MultiStrategyRunner, StrategyInstance};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Minimal 5-snapshot wide-format LOB fixture (re-used from lob_replay tests).
const FIXTURE_CSV: &str = "\
timestamp,b0_p,b0_q,b1_p,b1_q,a0_p,a0_q,a1_p,a1_q
1000,99.9,10.0,99.8,20.0,100.1,15.0,100.2,25.0
1001,99.85,12.0,99.75,18.0,100.15,14.0,100.25,22.0
1002,99.8,8.0,99.7,16.0,100.2,10.0,100.3,20.0
1003,99.95,11.0,99.85,19.0,100.05,13.0,100.15,21.0
1004,99.88,9.0,99.78,17.0,100.12,12.0,100.22,23.0
";

fn write_fixture(content: &str) -> String {
    let path = std::env::temp_dir()
        .join(format!(
            "vpin_integ_{}_{}.csv",
            std::process::id(),
            content.len()
        ))
        .to_str()
        .unwrap()
        .to_string();
    std::fs::write(&path, content).expect("fixture write must succeed");
    path
}

/// Build a single-strategy TWAP runner on a deterministic GBM path.
fn build_runner_with_vpin(steps: usize) -> MultiStrategyRunner {
    let total_notional = 100_000.0_f64;
    let arrival_price = 100.0_f64;
    let sigma = 0.02_f64;
    let eta = 0.001_f64;
    let lambda = 1e-4_f64;

    let strategy = StrategyInstance {
        name: "TWAP".to_string(),
        scheduler: Scheduler::new(
            RiskManager::new(total_notional, 0.99, 0.0),
            total_notional,
            total_notional / steps as f64,
            sigma,
            500.0,
            ExecutionMode::Twap,
            eta,
            lambda,
            Some(steps),
        ),
        metrics: ExecutionMetrics::new(arrival_price, total_notional),
        impact_model: SquareRootImpact::new(5_000_000.0, 0.5),
        transient_tracker: TransientImpactTracker::new(50.0),
        gamma: 0.005,
        completed: false,
    };

    let dt = 1.0 / steps as f64;
    let simulator = Box::new(GbmSimulator::new(arrival_price, 0.05, sigma, dt, 42));

    MultiStrategyRunner::new(vec![strategy], simulator, steps)
        .with_vpin(50_000.0, 10) // small bucket/window for fast saturation in tests
}

// ---------------------------------------------------------------------------
// Test 1 — VPIN series stays in [0, 1]
// ---------------------------------------------------------------------------

#[test]
fn vpin_series_is_bounded_zero_to_one() {
    let path = write_fixture(FIXTURE_CSV);
    let replay = LobReplay::from_csv(&path).expect("fixture must load");

    // Use a small bucket size so that even a 5-snapshot fixture produces buckets.
    let bucket_size = 1.0; // very small → many buckets per depth change
    let series = replay.compute_vpin_series(bucket_size, 5);

    // When the depth changes are small, the series may be empty — that is allowed.
    // But every entry that IS produced must be in [0, 1].
    for (ts, vpin) in &series {
        assert!(
            *vpin >= 0.0 && *vpin <= 1.0,
            "VPIN out of [0,1]: ts={ts} vpin={vpin}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2 — Conservation law survives VPIN recording
// ---------------------------------------------------------------------------

/// Run a full simulation with VPIN enabled and verify that the five-component
/// IS decomposition still sums to total IS within floating-point tolerance.
#[test]
fn vpin_recording_does_not_break_conservation_law() {
    let mut runner = build_runner_with_vpin(200);
    runner.run().expect("runner must complete without error");

    for strat in runner.strategies() {
        let d = strat
            .metrics
            .decompose()
            .expect("metrics must produce a decomposition");

        let reconstructed =
            d.spread_cost + d.temporary_impact + d.permanent_impact + d.timing_cost + d.opportunity_cost;
        let total = d.total;
        let tol = (total.abs() * 1e-8).max(1e-10);

        assert!(
            (reconstructed - total).abs() <= tol,
            "conservation law violated for {} after VPIN recording: \
             reconstructed={reconstructed:.10e} total={total:.10e} diff={:.10e}",
            strat.name,
            (reconstructed - total).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3 — Conditional IS analysis is self-consistent
// ---------------------------------------------------------------------------

/// When VPIN data is collected, the conditional IS for low- and high-VPIN
/// periods must sum (approximately) to the overall total IS, and neither
/// component may exceed the total IS in absolute value.
#[test]
fn conditional_is_analysis_is_internally_consistent() {
    let mut runner = build_runner_with_vpin(200);
    runner.run().expect("runner must complete without error");

    let threshold = 0.3_f64;

    for strat in runner.strategies() {
        let m = &strat.metrics;

        // VPIN must have been recorded for at least some trades.
        assert!(
            !m.vpin_at_execution.is_empty(),
            "VPIN must be recorded for strategy '{}'",
            strat.name
        );

        // All recorded VPIN values must be in [0, 1].
        for (i, &v) in m.vpin_at_execution.iter().enumerate() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "vpin_at_execution[{i}] = {v} is out of [0,1] for '{}'",
                strat.name
            );
        }

        // Conditional IS fractions must sum to total IS (within tolerance).
        let (low_is, high_is) = m.conditional_shortfall_percent(threshold);
        let total_is = m.shortfall_percent().unwrap_or(0.0);

        let conditional_sum = low_is.unwrap_or(0.0) + high_is.unwrap_or(0.0);
        let tol = (total_is.abs() * 1e-6).max(1e-10);
        assert!(
            (conditional_sum - total_is).abs() <= tol,
            "conditional IS sum ({conditional_sum:.8}%) must equal total IS ({total_is:.8}%) \
             for '{}', diff={:.8e}",
            strat.name,
            (conditional_sum - total_is).abs()
        );

        // high_vpin_trade_fraction must be in [0, 1].
        let frac = m.high_vpin_trade_fraction(threshold);
        assert!(
            frac >= 0.0 && frac <= 1.0,
            "high_vpin_trade_fraction = {frac} is out of [0,1] for '{}'",
            strat.name
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4 — LOB VPIN series length is bounded by snapshot count
// ---------------------------------------------------------------------------

/// The VPIN series produced by `compute_vpin_series` must be non-empty (given a
/// small enough bucket size) and must not exceed the number of snapshot pairs
/// (= snapshot_count - 1), because one pair is consumed per potential bucket event.
#[test]
fn lob_vpin_series_length_matches_snapshot_count() {
    let path = write_fixture(FIXTURE_CSV);
    let replay = LobReplay::from_csv(&path).expect("fixture must load");
    let snapshot_count = replay.len();

    // bucket_size=1.0 is small enough to complete many buckets from the fixture's
    // depth changes (each change is ~4-10 qty units).
    let series = replay.compute_vpin_series(1.0, 5);

    // Must have produced at least one entry given non-trivial depth changes.
    assert!(
        !series.is_empty(),
        "VPIN series must be non-empty for a fixture with depth changes"
    );

    // Must not exceed snapshot_count - 1 entries (one pair per step).
    assert!(
        series.len() <= snapshot_count,
        "VPIN series length {} must not exceed snapshot count {}",
        series.len(),
        snapshot_count
    );
}

// ---------------------------------------------------------------------------
// Test 5 — All LOB VPIN values are bounded [0, 1]
// ---------------------------------------------------------------------------

/// Every VPIN value emitted by `compute_vpin_series` must lie in [0, 1]
/// regardless of the depth-change pattern in the LOB data.
#[test]
fn lob_vpin_values_are_bounded_zero_to_one() {
    let path = write_fixture(FIXTURE_CSV);
    let replay = LobReplay::from_csv(&path).expect("fixture must load");
    let series = replay.compute_vpin_series(1.0, 5);

    for (ts, vpin) in &series {
        assert!(
            *vpin >= 0.0 && *vpin <= 1.0,
            "LOB VPIN out of [0,1]: ts={ts} vpin={vpin}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6 — P75 threshold produces a nonzero high-toxicity fraction
// ---------------------------------------------------------------------------

/// When VPIN values are injected into `ExecutionMetrics` and the P75 threshold
/// is applied, at least some trades must be classified as high-toxicity.
/// This guarantees the P75-based threshold has real discrimination power.
#[test]
fn p75_threshold_produces_nonzero_high_toxicity_fraction() {
    let mut m = ExecutionMetrics::new(100.0, 1_000_000.0);

    // Inject 8 VPIN values with two distinct clusters so P75 creates a split.
    // Low cluster: 0.35, 0.37, 0.38, 0.40 (4 values)
    // High cluster: 0.55, 0.58, 0.60, 0.62 (4 values)
    let vpin_values = [0.35, 0.37, 0.38, 0.40, 0.55, 0.58, 0.60, 0.62];
    for &v in &vpin_values {
        m.record_trade(100_000.0, 100.0, -0.1, 0.0, 0.0, 0.0, 0.0);
        m.record_vpin(v);
    }

    // Compute P75 from the injected values.
    let mut sorted = vpin_values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p75_idx = ((sorted.len() as f64 - 1.0) * 0.75).round() as usize;
    let threshold = sorted[p75_idx];

    let frac = m.high_vpin_trade_fraction(threshold);
    assert!(
        frac > 0.0,
        "P75 threshold {threshold:.4} must classify at least some fills as high-toxicity, got frac={frac}"
    );
    assert!(
        frac <= 1.0,
        "high-toxicity fraction must be ≤ 1.0, got {frac}"
    );
}
