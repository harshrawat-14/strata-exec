use strata_exec::analytics::distribution::DistributionStats;
use strata_exec::analytics::metrics::ExecutionMetrics;
use strata_exec::engine::risk::RiskManager;
use strata_exec::engine::scheduler::{ExecutionMode, Scheduler};
use strata_exec::events::event::{Event as DebugEvent, EventSender};
use strata_exec::market::garch::GarchSimulator;
use strata_exec::market::gbm::{GbmSimulator, PriceSimulator};
use strata_exec::observability::logger::init_debug_channel;
use strata_exec::research::multi_runner::{MultiStrategyRunner, StrategyInstance};

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

fn parse_paths_arg() -> usize {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--paths" {
            if let Some(val) = args.get(i + 1) {
                return val.parse::<usize>().unwrap_or(1).max(1);
            }
        }
        i += 1;
    }
    1
}

fn parse_model_arg() -> String {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" {
            if let Some(val) = args.get(i + 1) {
                let v = val.to_lowercase();
                if v == "garch" {
                    return v;
                }
            }
        }
        i += 1;
    }
    "gbm".to_string()
}

fn parse_debug_paths_flag() -> usize {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--debug-paths" {
            if let Some(val) = args.get(i + 1) {
                return val.parse::<usize>().unwrap_or(1);
            }
            return 1;
        }
        i += 1;
    }
    usize::MAX // If not specified, print all if --debug is present
}

fn parse_debug_flag() -> bool {
    std::env::args().any(|a| a == "--debug" || a == "--debug-paths")
}

fn parse_progress_flag() -> bool {
    std::env::args().any(|a| a == "--progress")
}

fn parse_experiments_flag() -> bool {
    std::env::args().any(|a| a == "--experiments")
}

fn parse_calibrate_flag() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--calibrate" {
            if let Some(val) = args.get(i + 1) {
                return Some(val.to_string());
            }
        }
        i += 1;
    }
    None
}

/// Build the simulator based on the `--model` argument.
fn build_simulator(
    model: &str,
    initial_price: f64,
    seed: u64,
    steps: usize,
) -> Box<dyn PriceSimulator> {
    let dt = 1.0 / steps as f64; // T = 1 year assumed for scaling

    match model {
        "garch" => Box::new(GarchSimulator::new(
            initial_price,
            0.05,     // mu
            0.000002, // omega
            0.08,     // alpha
            0.90,     // beta
            0.04,     // sigma2_init
            dt,
            seed,
        )),
        _ => Box::new(GbmSimulator::new(initial_price, 0.05, 0.20, dt, seed)),
    }
}

// ---------------------------------------------------------------------------
// Strategy factory
// ---------------------------------------------------------------------------

struct SimParams {
    total_notional: f64,
    base_chunk: f64,
    sigma_ref: f64,
    liquidity_ref: f64,
    eta: f64,
    lambda: f64,
    max_slippage_pct: f64,
    reserve_frac: f64,
    arrival_price: f64,
    daily_volume: f64,
    impact_coefficient: f64,
    transient_rho: f64,
}

fn build_strategies(p: &SimParams) -> Vec<StrategyInstance> {
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
            impact_model: strata_exec::execution::impact::SquareRootImpact::new(
                p.daily_volume,
                p.impact_coefficient,
            ),
            transient_tracker:
                strata_exec::execution::transient_impact::TransientImpactTracker::new(
                    p.transient_rho,
                ),
            completed: false,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Summary printing
// ---------------------------------------------------------------------------

fn print_single_run_summary(strategies: &[StrategyInstance], lambda: f64) {
    println!("\n{:=<70}", "");
    println!("  STRATEGY COMPARISON SUMMARY");
    println!("{:=<70}\n", "");

    for strat in strategies {
        println!("── {} ──", strat.name);
        println!("  Completed       : {}", strat.completed);

        // ── Diagnostics ──
        println!("\n  ===== Diagnostics =====");
        println!("  Arrival price   : {:.4}", strat.metrics.arrival_price());
        if let Some(avg) = strat.metrics.avg_exec_price() {
            println!("  Avg exec price  : {avg:.4}");
        }
        println!("  Trade count     : {}", strat.metrics.trade_count());
        println!("  Total executed  : {:.4}", strat.metrics.total_executed());

        // ── Cost breakdown ──
        println!("\n  ===== Cost Breakdown =====");
        println!(
            "  Shortfall ($)   : {:.6}",
            strat.metrics.implementation_shortfall()
        );
        if let Some(mean) = strat.metrics.mean_cost() {
            println!("  Mean cost ($)   : {mean:.6}");
        }
        if let Some(var) = strat.metrics.cost_variance() {
            let total_var = var * strat.metrics.trade_count() as f64;
            println!("  Total variance  : {total_var:.6}");
        }
        if let Some(ac) = strat.metrics.ac_objective(lambda) {
            println!("  AC objective ($): {ac:.6}");
        }

        // ── Percentage metrics (normalised by arrival_price × inventory) ──
        println!("\n  ===== Percentage of Notional =====");
        if let Some(pct) = strat.metrics.shortfall_percent() {
            println!("  Shortfall %     : {pct:.4}%");
        }
        if let Some(pct) = strat.metrics.mean_cost_percent() {
            println!("  Mean cost %     : {pct:.4}%");
        }
        if let Some(pct) = strat.metrics.risk_penalty_percent() {
            println!("  Risk Penalty %² : {pct:.10}");
        }
        if let Some(pct) = strat.metrics.ac_objective_percent(lambda) {
            println!("  AC objective %  : {pct:.4}%");
        }
        println!();
    }

    println!("CSV written to results/comparison.csv");
}

// ---------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------

/// Per-strategy accumulators for Monte Carlo statistics.
struct MonteCarloAccumulator {
    name: String,
    shortfall_pcts: Vec<f64>,
    ac_obj_pcts: Vec<f64>,
}

impl MonteCarloAccumulator {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            shortfall_pcts: Vec::new(),
            ac_obj_pcts: Vec::new(),
        }
    }
}

fn run_monte_carlo(
    p: &SimParams,
    num_paths: usize,
    base_seed: u64,
    model: &str,
    event_tx: Option<&EventSender>,
    debug_paths: usize,
) {
    let strategy_names = ["Heuristic", "Optimal", "AdaptiveOptimal"];
    let mut accumulators: Vec<MonteCarloAccumulator> = strategy_names
        .iter()
        .map(|n| MonteCarloAccumulator::new(n))
        .collect();

    let mut total_avg_vol = 0.0;

    for path_idx in 0..num_paths {
        let path_tx = if path_idx < debug_paths {
            event_tx
        } else {
            None
        };

        if let Some(_tx) = path_tx {
            // Nothing to send here
        }

        let seed = base_seed + path_idx as u64;
        let strategies = build_strategies(p);
        let max_steps = 500;
        let simulator = build_simulator(model, p.arrival_price, seed, max_steps);
        let mut runner = MultiStrategyRunner::new(strategies, simulator, max_steps);
        if let Some(tx) = path_tx {
            runner = runner.with_event_sender(tx.clone());
        }

        if let Err(e) = runner.run() {
            eprintln!("ERROR on path {path_idx}: {e}");
            continue;
        }

        total_avg_vol += runner.avg_volatility();

        for (i, strat) in runner.strategies().iter().enumerate() {
            let shortfall_pct = strat.metrics.shortfall_percent().unwrap_or(0.0);
            let ac_pct = strat.metrics.ac_objective_percent(p.lambda).unwrap_or(0.0);
            accumulators[i].shortfall_pcts.push(shortfall_pct);
            accumulators[i].ac_obj_pcts.push(ac_pct);
        }

        if let Some(_tx) = path_tx {
            // Nothing to send here
        }
    }

    let overall_avg_vol = if num_paths > 0 {
        total_avg_vol / num_paths as f64
    } else {
        0.0
    };

    // ── Print Monte Carlo summary ──────────────────────────────────
    println!("\n{:=<70}", "");
    println!("  MONTE CARLO SUMMARY  ({num_paths} paths)");
    println!("{:=<70}\n", "");

    for acc in &accumulators {
        let sf = DistributionStats::from_samples(&acc.shortfall_pcts);
        let ac = DistributionStats::from_samples(&acc.ac_obj_pcts);

        println!("── {} ──", acc.name);

        if let Some(s) = &sf {
            println!("  Shortfall Distribution:");
            println!("    Mean       : {:.4}%", s.mean);
            println!("    Std Dev    : {:.4}%", s.std_dev);
            println!("    Median     : {:.4}%", s.median);
            println!("    VaR 95     : {:.4}%", s.var_95);
            println!("    CVaR 95    : {:.4}%", s.cvar_95);
            println!("    VaR 99     : {:.4}%", s.var_99);
            println!("    CVaR 99    : {:.4}%", s.cvar_99);
            println!("    Max        : {:.4}%", s.max);
            println!("    Skewness   : {:.4}", s.skewness);
            println!("    Kurtosis   : {:.4}", s.kurtosis);
        }

        if let Some(a) = &ac {
            println!("  AC Objective Distribution:");
            println!("    Mean       : {:.4}%", a.mean);
            println!("    Std Dev    : {:.4}%", a.std_dev);
            println!("    Median     : {:.4}%", a.median);
            println!("    VaR 95     : {:.4}%", a.var_95);
            println!("    CVaR 95    : {:.4}%", a.cvar_95);
            println!("    VaR 99     : {:.4}%", a.var_99);
            println!("    CVaR 99    : {:.4}%", a.cvar_99);
            println!("    Max        : {:.4}%", a.max);
            println!("    Skewness   : {:.4}", a.skewness);
            println!("    Kurtosis   : {:.4}", a.kurtosis);
        }
        println!();
    }

    println!("  Avg Volatility: {:.4}%", overall_avg_vol * 100.0);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let num_paths = parse_paths_arg();
    let model = parse_model_arg();
    let debug = parse_debug_flag();
    let debug_paths = parse_debug_paths_flag();
    let progress = parse_progress_flag();
    let experiments = parse_experiments_flag();
    let calibrate_path = parse_calibrate_flag();

    // ── Observability setup ────────────────────────────────────────
    let debug_state = if debug || progress {
        let (tx, handle) = init_debug_channel();
        Some((tx, handle))
    } else {
        None
    };
    let event_tx = debug_state.as_ref().map(|(tx, _)| tx);

    let mut params = SimParams {
        total_notional: 1_000_000.0,
        base_chunk: 10_000.0,
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
    };

    if let Some(path) = calibrate_path {
        println!("Calibration Mode Enabled -> Analyzing: {}", path);
        match strata_exec::calibration::impact_fit::CalibrationEngine::load_csv(&path) {
            Ok(trades) => {
                let engine = strata_exec::calibration::impact_fit::CalibrationEngine::new(
                    params.daily_volume,
                    params.sigma_ref,
                );
                if let Some(y_opt) = engine.calibrate(&trades) {
                    println!("  Calibrated Y-Coefficient: {:.4}", y_opt);
                    params.impact_coefficient = y_opt;
                } else {
                    println!("  Warning: Calibration yielded no valid trades, defaulting Y=0.5");
                }
            }
            Err(e) => {
                eprintln!("  Error loading calibration CSV: {e}");
                std::process::exit(1);
            }
        }
    }

    let _base_seed: u64 = 42;

    println!("Model: {model}");

    if let Some(_tx) = event_tx {
        // Nothing for now, could emit SimulationStart if variant existed
    }

    if experiments {
        // ── Experiment Parameter Sweeps ───────────────────────────────
        if let Err(e) = strata_exec::research::experiments::run_all_sweeps(num_paths, event_tx) {
            eprintln!("Experiment Sweeps Failed: {e}");
            std::process::exit(1);
        }
        
    } else {
        // ── Standard Multi-Strategy Simulator ─────────────────────────
        let mut overall_avg_vol = 0.0;
        let base_seed: u64 = 42;

    for path_id in 0..num_paths {
        if let Some(tx) = event_tx {
            let _ = tx.send(DebugEvent::PathStart { path_id });
        }

        let strategies = build_strategies(&params);
        let max_steps = 500;
        let simulator = build_simulator(
            &model,
            params.arrival_price,
            base_seed + path_id as u64,
            max_steps,
        );
        let mut runner = MultiStrategyRunner::new(strategies, simulator, max_steps);

        // Only pipe events into runner if debug mode is explicitly set to true or path matches
        if debug && path_id < debug_paths {
            if let Some(tx) = event_tx {
                runner = runner.with_event_sender(tx.clone());
            }
        }

        if let Err(e) = runner.run() {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(DebugEvent::PathEnd { path_id });
            if progress {
                let _ = tx.send(DebugEvent::Progress {
                    completed: path_id + 1,
                    total: num_paths,
                });
            }
        }

        overall_avg_vol += runner.avg_volatility();

        if path_id == 0 && !progress {
            // Only print summary on the first run to not clutter console
            println!("  Avg Volatility: {:.4}%", runner.avg_volatility() * 100.0);
            print_single_run_summary(runner.strategies(), params.lambda);
        }
    }
    } // End of else block

    // ── Observability teardown ──────────────────────────────────────
    if let Some(tx) = event_tx {
        let _ = tx.send(DebugEvent::SimulationEnd);
    }
    if let Some((tx, handle)) = debug_state {
        drop(tx); // close the channel
        let _ = handle.join(); // flush remaining events
    }
}
