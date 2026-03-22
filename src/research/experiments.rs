use std::fs;
use std::io::Write;

use crate::analytics::distribution::DistributionStats;
use crate::events::event::{Event as DebugEvent, EventSender};
use crate::market::gbm::GbmSimulator;
use crate::research::multi_runner::MultiStrategyRunner;

use crate::analytics::metrics::ExecutionMetrics;
use crate::engine::risk::RiskManager;
use crate::engine::scheduler::{ExecutionMode, Scheduler};
use crate::execution::impact::SquareRootImpact;
use crate::execution::transient_impact::TransientImpactTracker;
use crate::research::multi_runner::StrategyInstance;

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

/// Helper to build strategy instances for a given configuration
fn build_sweep_strategies(p: &SweepConfig) -> Vec<StrategyInstance> {
    let modes = [
        ("Heuristic", ExecutionMode::Heuristic),
        ("Optimal", ExecutionMode::Optimal),
        ("AdaptiveOptimal", ExecutionMode::AdaptiveOptimal),
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

/// Helper to run a Monte Carlo block for a specific parameter `SweepConfig`
fn evaluate_config(
    idx: usize,
    total_configs: usize,
    params: &SweepConfig,
    num_paths: usize,
    event_tx: Option<&EventSender>,
) -> Vec<(String, DistributionStats, DistributionStats, DistributionStats)> {
    let mut shortfall_accumulators: Vec<Vec<f64>> = vec![vec![]; 3]; // For 3 strategies
    let mut mean_cost_accumulators: Vec<Vec<f64>> = vec![vec![]; 3];
    let mut cvar_accumulators: Vec<Vec<f64>> = vec![vec![]; 3];
    let strategy_names = ["Heuristic", "Optimal", "AdaptiveOptimal"];

    let base_seed = 42;

    for path_id in 0..num_paths {
        if let Some(tx) = event_tx {
            let _ = tx.send(DebugEvent::Progress {
                completed: (idx * num_paths) + path_id,
                total: total_configs * num_paths,
            });
        }

        let seed = base_seed + path_id as u64;
        let steps_per_day = 500;
        let max_steps = (params.horizon_days * steps_per_day as f64) as usize;
        let dt = params.horizon_days / max_steps as f64;
        let simulator = Box::new(GbmSimulator::new(params.arrival_price, 0.05, 0.20, dt, seed));
        
        let strategies = build_sweep_strategies(params);
        let mut runner = MultiStrategyRunner::new(strategies, simulator, max_steps);
        
        // We silence the internal simulation CSV exports and event spams for grid sweeps
        // to prevent disk IO bottlenecking.

        if let Err(e) = runner.run() {
            eprintln!("ERROR during sweep eval: {e}");
            continue;
        }

        for (i, strat) in runner.strategies().iter().enumerate() {
            let sf_pct = strat.metrics.shortfall_percent().unwrap_or(0.0);
            let mean_pct = strat.metrics.mean_cost_percent().unwrap_or(0.0);
            let ac_pct = strat.metrics.ac_objective_percent(params.lambda).unwrap_or(0.0);
            
            shortfall_accumulators[i].push(sf_pct);
            mean_cost_accumulators[i].push(mean_pct);
            cvar_accumulators[i].push(ac_pct);
        }
    }

    let mut results = Vec::new();
    for i in 0..3 {
        let sf = DistributionStats::from_samples(&shortfall_accumulators[i]).unwrap();
        let mc = DistributionStats::from_samples(&mean_cost_accumulators[i]).unwrap();
        let ac = DistributionStats::from_samples(&cvar_accumulators[i]).unwrap();
        results.push((strategy_names[i].to_string(), sf, mc, ac));
    }
    
    results
}

// ---------------------------------------------------------------------------
// Main Executor
// ---------------------------------------------------------------------------

pub fn run_all_sweeps(num_paths: usize, event_tx: Option<&EventSender>) -> Result<(), String> {
    let results_dir = "results";
    if fs::metadata(results_dir).is_err() {
        fs::create_dir(results_dir).map_err(|e| e.to_string())?;
    }

    // 1. Trade Count Sweep
    run_trade_count_sweep(num_paths, event_tx)?;

    // 2. Volatility Sweep
    run_volatility_sweep(num_paths, event_tx)?;

    // 3. Impact Coefficient Sweep
    run_impact_sweep(num_paths, event_tx)?;

    Ok(())
}

fn run_trade_count_sweep(num_paths: usize, event_tx: Option<&EventSender>) -> Result<(), String> {
    let slices = [50.0, 100.0, 200.0, 400.0];
    let path = "results/sweep_trade_chunks.csv";
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    
    writeln!(file, "TradeSlices,Strategy,MeanShortfall_Pct,Shortfall_Variance,CVaR_95_Pct")
        .map_err(|e| e.to_string())?;

    for (idx, &target_slices) in slices.iter().enumerate() {
        let mut params = SweepConfig::default();
        params.base_chunk = params.total_notional / target_slices;

        let evals = evaluate_config(idx, slices.len(), &params, num_paths, event_tx);
        
        for (strategy, sf_stats, _) in evals {
            // Write parameter, Strategy name, Mean, Variance (std_dev^2), and CVaR95
            let var = sf_stats.std_dev * sf_stats.std_dev;
            writeln!(file, "{},{},{:.6},{:.6},{:.6}", target_slices, strategy, sf_stats.mean, var, sf_stats.cvar_95)
                .map_err(|e| e.to_string())?;
        }
    }

    println!("Exported {}", path);
    Ok(())
}

fn run_volatility_sweep(num_paths: usize, event_tx: Option<&EventSender>) -> Result<(), String> {
    let volatilities = [0.01, 0.02, 0.05, 0.10];
    let path = "results/sweep_volatility.csv";
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    
    writeln!(file, "Volatility,Strategy,MeanShortfall_Pct,Shortfall_Variance,CVaR_95_Pct")
        .map_err(|e| e.to_string())?;

    for (idx, &vol) in volatilities.iter().enumerate() {
        let mut params = SweepConfig::default();
        params.sigma_ref = vol;

        let evals = evaluate_config(idx, volatilities.len(), &params, num_paths, event_tx);
        
        for (strategy, sf_stats, _) in evals {
            let var = sf_stats.std_dev * sf_stats.std_dev;
            writeln!(file, "{:.2},{},{:.6},{:.6},{:.6}", vol, strategy, sf_stats.mean, var, sf_stats.cvar_95)
                .map_err(|e| e.to_string())?;
        }
    }

    println!("Exported {}", path);
    Ok(())
}

fn run_impact_sweep(num_paths: usize, event_tx: Option<&EventSender>) -> Result<(), String> {
    let impacts = [0.1, 0.5, 1.0, 2.0];
    let path = "results/sweep_impact.csv";
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    
    writeln!(file, "ImpactCoefficient,Strategy,MeanShortfall_Pct,Shortfall_Variance,CVaR_95_Pct")
        .map_err(|e| e.to_string())?;

    for (idx, &impact) in impacts.iter().enumerate() {
        let mut params = SweepConfig::default();
        params.impact_coefficient = impact;

        let evals = evaluate_config(idx, impacts.len(), &params, num_paths, event_tx);
        
        for (strategy, sf_stats, _) in evals {
            let var = sf_stats.std_dev * sf_stats.std_dev;
            writeln!(file, "{:.2},{},{:.6},{:.6},{:.6}", impact, strategy, sf_stats.mean, var, sf_stats.cvar_95)
                .map_err(|e| e.to_string())?;
        }
    }

    println!("Exported {}", path);
    Ok(())
}
