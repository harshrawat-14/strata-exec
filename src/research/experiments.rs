use std::fmt::Display;
use std::fs;
use std::io::Write;

use crate::analytics::distribution::DistributionStats;
use crate::analytics::metrics::ExecutionMetrics;
use crate::engine::risk::RiskManager;
use crate::engine::scheduler::{ExecutionMode, Scheduler};
use crate::events::event::{Event as DebugEvent, EventSender};
use crate::execution::impact::SquareRootImpact;
use crate::execution::transient_impact::TransientImpactTracker;
use crate::market::garch::GarchSimulator;
use crate::market::gbm::{GbmSimulator, PriceSimulator};
use crate::research::multi_runner::{MultiStrategyRunner, StrategyInstance};

/// Configuration for a single execution setup
#[derive(Clone, Debug)]
pub struct SweepConfig {
    pub total_notional: f64,
    pub base_chunk: f64,
    pub sigma_ref: f64,
    pub liquidity_ref: f64,
    pub eta: f64,
    pub lambda: f64,
    /// Permanent impact coefficient γ (dimensionless fraction).
    /// Each step contributes γ·(traded/total_notional) to the cumulative
    /// fractional price depression applied to all future steps.
    /// Default: 0.005 — selling the full inventory causes ~0.5% permanent
    /// price impact, consistent with AC literature for 20% ADV executions.
    pub gamma: f64,
    pub max_slippage_pct: f64,
    pub reserve_frac: f64,
    pub arrival_price: f64,
    pub daily_volume: f64,
    pub impact_coefficient: f64,
    pub transient_rho: f64,
    pub horizon_days: f64,
    /// When `true`, each path uses a GARCH(1,1) simulator instead of GBM.
    pub use_garch: bool,
    /// Option B: force-complete residual inventory at horizon end.
    pub force_complete: bool,
    /// Number of simulation steps per path (= runner max_steps).
    /// AC horizon is set to this value so all strategies share the same horizon.
    pub max_steps: usize,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            total_notional: 1_000_000.0,
            base_chunk: 10_000.0, // Default 100 slices
            sigma_ref: 0.02,
            liquidity_ref: 500.0,
            eta: 0.001,
            lambda: 1e-4,
            gamma: 0.005,
            max_slippage_pct: 0.5,
            reserve_frac: 0.0,
            arrival_price: 100.0,
            daily_volume: 5_000_000.0,
            impact_coefficient: 0.5,
            transient_rho: 50.0,
            horizon_days: 1.0,
            use_garch: false,
            force_complete: false,
            max_steps: 500,
        }
    }
}

const HORIZON_GRID_DAYS: [f64; 4] = [0.25, 0.5, 1.0, 2.0];
const TRADE_SLICE_GRID: [usize; 4] = [25, 50, 100, 200];
const VOLATILITY_GRID: [f64; 4] = [0.10, 0.20, 0.30, 0.40];
const IMPACT_GRID: [f64; 4] = [0.10, 0.50, 1.00, 2.00];

const STRATEGIES: [&str; 4] = ["TWAP", "Heuristic", "Optimal", "AdaptiveOptimal"];

#[derive(Debug)]
struct StrategySweepStats {
    strategy: String,
    shortfall: DistributionStats,
    ac_objective: DistributionStats,
}

/// Helper to build strategy instances for a given configuration
fn build_sweep_strategies(p: &SweepConfig, max_steps: usize) -> Vec<StrategyInstance> {
    let modes = [
        (STRATEGIES[0], ExecutionMode::Twap),
        (STRATEGIES[1], ExecutionMode::Heuristic),
        (STRATEGIES[2], ExecutionMode::Optimal),
        (STRATEGIES[3], ExecutionMode::AdaptiveOptimal),
    ];

    modes
        .iter()
        .map(|(name, mode)| StrategyInstance {
            name: name.to_string(),
            scheduler: Scheduler::new(
                RiskManager::new(p.total_notional, p.max_slippage_pct, p.reserve_frac),
                p.total_notional,
                p.base_chunk,
                p.sigma_ref,
                p.liquidity_ref,
                *mode,
                p.eta,
                p.lambda,
                Some(max_steps), // Option A: AC horizon matches runner step count
            ),
            metrics: ExecutionMetrics::new(p.arrival_price, p.total_notional),
            impact_model: SquareRootImpact::new(p.daily_volume, p.impact_coefficient),
            transient_tracker: TransientImpactTracker::new(p.transient_rho),
            gamma: p.gamma,
            completed: false,
        })
        .collect()
}

/// Helper to run a Monte Carlo block for one parameter configuration.
fn evaluate_config(
    params: &SweepConfig,
    num_paths: usize,
    progress_done: usize,
    total_paths: usize,
    event_tx: Option<&EventSender>,
) -> Result<Vec<StrategySweepStats>, String> {
    let mut shortfall_accumulators: Vec<Vec<f64>> = vec![vec![]; STRATEGIES.len()];
    let mut ac_accumulators: Vec<Vec<f64>> = vec![vec![]; STRATEGIES.len()];
    let base_seed = 42_u64;

    for path_id in 0..num_paths {
        if let Some(tx) = event_tx {
            let _ = tx.send(DebugEvent::Progress {
                completed: progress_done + path_id + 1,
                total: total_paths,
            });
        }

        let seed = base_seed + path_id as u64;
        // Derive max_steps from horizon_days (500 steps/day).
        // This ensures AC's horizon always matches the simulation step count.
        let steps_per_day = 500usize;
        let max_steps = ((params.horizon_days * steps_per_day as f64).round() as usize).max(1);
        let dt = params.horizon_days / max_steps as f64;
        let simulator: Box<dyn PriceSimulator> = if params.use_garch {
            // GARCH(1,1): sigma2_init = sigma_ref² so that GARCH starts at the
            // same annualised volatility as the GBM path (sigma_ref).
            // The two simulators are then apples-to-apples at t=0; GARCH drifts
            // away from sigma_ref in subsequent steps due to volatility clustering.
            let sigma2_init = params.sigma_ref * params.sigma_ref;
            Box::new(GarchSimulator::new(
                params.arrival_price,
                0.05,      // mu
                0.000002,  // omega
                0.08,      // alpha
                0.90,      // beta
                sigma2_init,
                dt,
                seed,
            ))
        } else {
            Box::new(GbmSimulator::new(
                params.arrival_price,
                0.05,
                params.sigma_ref, // same sigma_ref as GARCH sigma2_init = sigma_ref²
                dt,
                seed,
            ))
        };

        let strategies = build_sweep_strategies(params, max_steps);
        let mut runner = MultiStrategyRunner::new(strategies, simulator, max_steps);
        runner.force_complete = params.force_complete;

        if let Err(e) = runner.run() {
            return Err(format!("Sweep evaluation failed for path {path_id}: {e}"));
        }

        for (i, strat) in runner.strategies().iter().enumerate() {
            let sf_pct = strat.metrics.shortfall_percent().unwrap_or(0.0);
            let ac_pct = strat.metrics.ac_objective_percent(params.lambda).unwrap_or(0.0);

            shortfall_accumulators[i].push(sf_pct);
            ac_accumulators[i].push(ac_pct);
        }
    }

    let mut results = Vec::with_capacity(STRATEGIES.len());
    for i in 0..STRATEGIES.len() {
        let shortfall = DistributionStats::from_samples(&shortfall_accumulators[i])
            .ok_or_else(|| format!("No shortfall samples for strategy {}", STRATEGIES[i]))?;
        let ac_objective = DistributionStats::from_samples(&ac_accumulators[i])
            .ok_or_else(|| format!("No AC objective samples for strategy {}", STRATEGIES[i]))?;

        results.push(StrategySweepStats {
            strategy: STRATEGIES[i].to_string(),
            shortfall,
            ac_objective,
        });
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Main Executor
// ---------------------------------------------------------------------------

pub fn run_all_sweeps(num_paths: usize, event_tx: Option<&EventSender>) -> Result<(), String> {
    let results_dir = "results";
    if fs::metadata(results_dir).is_err() {
        fs::create_dir(results_dir).map_err(|e| e.to_string())?;
    }

    let total_configs = HORIZON_GRID_DAYS.len()
        + TRADE_SLICE_GRID.len()
        + VOLATILITY_GRID.len()
        + IMPACT_GRID.len();
    let total_paths = total_configs * num_paths;
    let mut progress_done = 0usize;

    // 1. Execution Horizon Sweep
    run_horizon_sweep(num_paths, total_paths, &mut progress_done, event_tx)?;

    // 2. Trade Slice Sweep
    run_trade_count_sweep(num_paths, total_paths, &mut progress_done, event_tx)?;

    // 3. Volatility Sweep
    run_volatility_sweep(num_paths, total_paths, &mut progress_done, event_tx)?;

    // 4. Impact Coefficient Sweep
    run_impact_sweep(num_paths, total_paths, &mut progress_done, event_tx)?;

    Ok(())
}

fn write_sweep_header(file: &mut fs::File, parameter_name: &str) -> Result<(), String> {
    writeln!(
        file,
        "ParameterName,ParameterValue,Strategy,NumPaths,MeanImplementationShortfall_Pct,ImplementationShortfallVariance_Pct2,CVaR95ImplementationShortfall_Pct,MeanACObjective_Pct,ACObjectiveVariance_Pct2,CVaR95ACObjective_Pct"
    )
    .map_err(|e| format!("Failed to write header for {parameter_name}: {e}"))
}

fn write_sweep_rows<T: Display>(
    file: &mut fs::File,
    parameter_name: &str,
    parameter_value: T,
    num_paths: usize,
    strategy_stats: &[StrategySweepStats],
) -> Result<(), String> {
    for stats in strategy_stats {
        let sf_var = stats.shortfall.std_dev * stats.shortfall.std_dev;
        let ac_var = stats.ac_objective.std_dev * stats.ac_objective.std_dev;
        writeln!(
            file,
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            parameter_name,
            parameter_value,
            stats.strategy,
            num_paths,
            stats.shortfall.mean,
            sf_var,
            stats.shortfall.cvar_95,
            stats.ac_objective.mean,
            ac_var,
            stats.ac_objective.cvar_95,
        )
        .map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn run_horizon_sweep(
    num_paths: usize,
    total_paths: usize,
    progress_done: &mut usize,
    event_tx: Option<&EventSender>,
) -> Result<(), String> {
    let path = "results/sweep_horizon.csv";
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    write_sweep_header(&mut file, "ExecutionHorizonDays")?;

    for horizon_days in HORIZON_GRID_DAYS {
        let mut params = SweepConfig::default();
        params.horizon_days = horizon_days;

        let stats = evaluate_config(&params, num_paths, *progress_done, total_paths, event_tx)?;
        *progress_done += num_paths;
        write_sweep_rows(
            &mut file,
            "ExecutionHorizonDays",
            format!("{horizon_days:.4}"),
            num_paths,
            &stats,
        )?;
    }

    println!("Exported {}", path);
    Ok(())
}

fn run_trade_count_sweep(
    num_paths: usize,
    total_paths: usize,
    progress_done: &mut usize,
    event_tx: Option<&EventSender>,
) -> Result<(), String> {
    let path = "results/sweep_trade_chunks.csv";
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    write_sweep_header(&mut file, "TradeSlices")?;

    for slices in TRADE_SLICE_GRID {
        let mut params = SweepConfig::default();
        params.base_chunk = params.total_notional / slices as f64;

        let stats = evaluate_config(&params, num_paths, *progress_done, total_paths, event_tx)?;
        *progress_done += num_paths;
        write_sweep_rows(&mut file, "TradeSlices", slices, num_paths, &stats)?;
    }

    println!("Exported {}", path);
    Ok(())
}

fn run_volatility_sweep(
    num_paths: usize,
    total_paths: usize,
    progress_done: &mut usize,
    event_tx: Option<&EventSender>,
) -> Result<(), String> {
    let path = "results/sweep_volatility.csv";
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    write_sweep_header(&mut file, "Volatility")?;

    for vol in VOLATILITY_GRID {
        let mut params = SweepConfig::default();
        params.sigma_ref = vol;

        let stats = evaluate_config(&params, num_paths, *progress_done, total_paths, event_tx)?;
        *progress_done += num_paths;
        write_sweep_rows(
            &mut file,
            "Volatility",
            format!("{vol:.4}"),
            num_paths,
            &stats,
        )?;
    }

    println!("Exported {}", path);
    Ok(())
}

fn run_impact_sweep(
    num_paths: usize,
    total_paths: usize,
    progress_done: &mut usize,
    event_tx: Option<&EventSender>,
) -> Result<(), String> {
    let path = "results/sweep_impact.csv";
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    write_sweep_header(&mut file, "ImpactCoefficient")?;

    for impact in IMPACT_GRID {
        let mut params = SweepConfig::default();
        params.impact_coefficient = impact;

        let stats = evaluate_config(&params, num_paths, *progress_done, total_paths, event_tx)?;
        *progress_done += num_paths;
        write_sweep_rows(
            &mut file,
            "ImpactCoefficient",
            format!("{impact:.4}"),
            num_paths,
            &stats,
        )?;
    }

    println!("Exported {}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_config_defaults_are_valid() {
        let cfg = SweepConfig::default();
        assert!(cfg.total_notional > 0.0, "total_notional must be positive");
        assert!(cfg.base_chunk > 0.0, "base_chunk must be positive");
        assert!(cfg.sigma_ref > 0.0, "sigma_ref must be positive");
        assert!(cfg.arrival_price > 0.0, "arrival_price must be positive");
        assert!(cfg.daily_volume > 0.0, "daily_volume must be positive");
        assert!(cfg.impact_coefficient > 0.0, "impact_coefficient must be positive");
        assert!(cfg.horizon_days > 0.0, "horizon_days must be positive");
        // Verify default produces ~100 slices
        let slices = (cfg.total_notional / cfg.base_chunk).round() as usize;
        assert_eq!(slices, 100, "default should produce ~100 slices");
    }

    #[test]
    fn sweep_grids_are_non_empty() {
        assert!(!HORIZON_GRID_DAYS.is_empty());
        assert!(!TRADE_SLICE_GRID.is_empty());
        assert!(!VOLATILITY_GRID.is_empty());
        assert!(!IMPACT_GRID.is_empty());
        // All elements must be strictly positive
        for &h in &HORIZON_GRID_DAYS {
            assert!(h > 0.0, "horizon must be positive, got {h}");
        }
        for &s in &TRADE_SLICE_GRID {
            assert!(s > 0, "trade slices must be positive, got {s}");
        }
        for &v in &VOLATILITY_GRID {
            assert!(v > 0.0, "volatility must be positive, got {v}");
        }
        for &i in &IMPACT_GRID {
            assert!(i > 0.0, "impact must be positive, got {i}");
        }
    }

    #[test]
    fn evaluate_config_returns_all_strategies() {
        let params = SweepConfig::default();
        let stats = evaluate_config(&params, 2, 0, 2, None)
            .expect("evaluate_config should succeed");
        assert_eq!(
            stats.len(),
            STRATEGIES.len(),
            "should return one StrategySweepStats per strategy"
        );
        for s in &stats {
            assert!(!s.strategy.is_empty(), "strategy name must not be empty");
            // mean shortfall should be finite for a well-configured run
            assert!(
                s.shortfall.mean.is_finite(),
                "shortfall mean should be finite for {}",
                s.strategy
            );
            assert!(
                s.ac_objective.mean.is_finite(),
                "ac_objective mean should be finite for {}",
                s.strategy
            );
        }
    }

    #[test]
    fn evaluate_config_deterministic_across_runs() {
        let params = SweepConfig::default();
        let stats_a = evaluate_config(&params, 3, 0, 3, None)
            .expect("run A should succeed");
        let stats_b = evaluate_config(&params, 3, 0, 3, None)
            .expect("run B should succeed");

        for (a, b) in stats_a.iter().zip(stats_b.iter()) {
            assert_eq!(a.strategy, b.strategy);
            assert!(
                (a.shortfall.mean - b.shortfall.mean).abs() < 1e-12,
                "deterministic shortfall mean mismatch for {}: {} vs {}",
                a.strategy, a.shortfall.mean, b.shortfall.mean
            );
            assert!(
                (a.ac_objective.mean - b.ac_objective.mean).abs() < 1e-12,
                "deterministic AC objective mean mismatch for {}: {} vs {}",
                a.strategy, a.ac_objective.mean, b.ac_objective.mean
            );
        }
    }

    #[test]
    fn evaluate_config_varies_with_horizon() {
        let mut params_short = SweepConfig::default();
        params_short.horizon_days = 0.25;
        let mut params_long = SweepConfig::default();
        params_long.horizon_days = 2.0;

        let stats_short = evaluate_config(&params_short, 3, 0, 3, None)
            .expect("short horizon should succeed");
        let stats_long = evaluate_config(&params_long, 3, 0, 3, None)
            .expect("long horizon should succeed");

        // At minimum, running with different horizons should produce different results
        // for the Optimal strategy (which is horizon-aware)
        let opt_short = stats_short.iter().find(|s| s.strategy == "Optimal").unwrap();
        let opt_long = stats_long.iter().find(|s| s.strategy == "Optimal").unwrap();
        assert!(
            (opt_short.shortfall.mean - opt_long.shortfall.mean).abs() > 1e-15,
            "different horizons should produce different shortfall results"
        );
    }

    #[test]
    fn evaluate_config_varies_with_volatility() {
        let mut params_low = SweepConfig::default();
        params_low.sigma_ref = 0.01;
        let mut params_high = SweepConfig::default();
        params_high.sigma_ref = 0.40;

        let stats_low = evaluate_config(&params_low, 3, 0, 3, None)
            .expect("low vol should succeed");
        let stats_high = evaluate_config(&params_high, 3, 0, 3, None)
            .expect("high vol should succeed");

        // Higher volatility should generally produce higher shortfall variance
        let heur_low = stats_low.iter().find(|s| s.strategy == "Heuristic").unwrap();
        let heur_high = stats_high.iter().find(|s| s.strategy == "Heuristic").unwrap();
        assert!(
            heur_high.shortfall.std_dev > heur_low.shortfall.std_dev,
            "higher volatility should produce higher shortfall std_dev: {} vs {}",
            heur_high.shortfall.std_dev, heur_low.shortfall.std_dev
        );
    }

    #[test]
    fn build_sweep_strategies_creates_all_modes() {
        let params = SweepConfig::default();
        let strategies = build_sweep_strategies(&params, params.max_steps);
        assert_eq!(strategies.len(), STRATEGIES.len());
        for (i, strat) in strategies.iter().enumerate() {
            assert_eq!(strat.name, STRATEGIES[i]);
            assert!(!strat.completed);
        }
    }

    #[test]
    fn csv_header_contains_all_metric_columns() {
        // Verify the header written by write_sweep_header has all required columns
        let expected_cols = [
            "ParameterName",
            "ParameterValue",
            "Strategy",
            "NumPaths",
            "MeanImplementationShortfall_Pct",
            "ImplementationShortfallVariance_Pct2",
            "CVaR95ImplementationShortfall_Pct",
            "MeanACObjective_Pct",
            "ACObjectiveVariance_Pct2",
            "CVaR95ACObjective_Pct",
        ];

        let dir = std::env::temp_dir().join("strata_test_csv_header");
        let _ = fs::create_dir(&dir);
        let path = dir.join("test_header.csv");
        let mut file = fs::File::create(&path).expect("create temp file");
        write_sweep_header(&mut file, "TestParam").expect("write header");
        drop(file);

        let content = fs::read_to_string(&path).expect("read temp file");
        for col in &expected_cols {
            assert!(
                content.contains(col),
                "CSV header missing column '{col}': {content}"
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
