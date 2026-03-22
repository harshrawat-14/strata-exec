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
use crate::market::gbm::GbmSimulator;
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
    pub max_slippage_pct: f64,
    pub reserve_frac: f64,
    pub arrival_price: f64,
    pub daily_volume: f64,
    pub impact_coefficient: f64,
    pub transient_rho: f64,
    pub horizon_days: f64,
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
            max_slippage_pct: 0.5,
            reserve_frac: 0.0,
            arrival_price: 100.0,
            daily_volume: 5_000_000.0,
            impact_coefficient: 0.5,
            transient_rho: 50.0,
            horizon_days: 1.0,
        }
    }
}

const HORIZON_GRID_DAYS: [f64; 4] = [0.25, 0.5, 1.0, 2.0];
const TRADE_SLICE_GRID: [usize; 4] = [25, 50, 100, 200];
const VOLATILITY_GRID: [f64; 4] = [0.10, 0.20, 0.30, 0.40];
const IMPACT_GRID: [f64; 4] = [0.10, 0.50, 1.00, 2.00];

const STRATEGIES: [&str; 3] = ["Heuristic", "Optimal", "AdaptiveOptimal"];

#[derive(Debug)]
struct StrategySweepStats {
    strategy: String,
    shortfall: DistributionStats,
    ac_objective: DistributionStats,
}

/// Helper to build strategy instances for a given configuration
fn build_sweep_strategies(p: &SweepConfig) -> Vec<StrategyInstance> {
    let modes = [
        (STRATEGIES[0], ExecutionMode::Heuristic),
        (STRATEGIES[1], ExecutionMode::Optimal),
        (STRATEGIES[2], ExecutionMode::AdaptiveOptimal),
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
            ),
            metrics: ExecutionMetrics::new(p.arrival_price, p.total_notional),
            impact_model: SquareRootImpact::new(p.daily_volume, p.impact_coefficient),
            transient_tracker: TransientImpactTracker::new(p.transient_rho),
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
        let steps_per_day = 500usize;
        let max_steps = ((params.horizon_days * steps_per_day as f64).round() as usize).max(1);
        let dt = params.horizon_days / max_steps as f64;
        let simulator = Box::new(GbmSimulator::new(
            params.arrival_price,
            0.05,
            params.sigma_ref,
            dt,
            seed,
        ));

        let strategies = build_sweep_strategies(params);
        let mut runner = MultiStrategyRunner::new(strategies, simulator, max_steps);

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
