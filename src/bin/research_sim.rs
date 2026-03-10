use strata_exec::core::metrics::ExecutionMetrics;
use strata_exec::core::risk::RiskManager;
use strata_exec::core::scheduler::{ExecutionMode, Scheduler};
use strata_exec::observability::events::{DebugEvent, EventSender};
use strata_exec::observability::logger::init_debug_channel;
use strata_exec::research::amm::ConstantProductAMM;
use strata_exec::research::garch::GarchSimulator;
use strata_exec::research::multi_runner::{MultiStrategyRunner, StrategyInstance};
use strata_exec::research::simulator::{GbmSimulator, PriceSimulator};
use strata_exec::research::stats::DistributionStats;

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
            0.05,       // mu
            0.000002,   // omega
            0.08,       // alpha
            0.90,       // beta
            0.04,       // sigma2_init
            dt,
            seed,
        )),
        _ => Box::new(GbmSimulator::new(
            initial_price,
            0.05,
            0.20,
            dt,
            seed,
        )),
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
}

fn build_strategies(p: &SimParams) -> Vec<StrategyInstance> {
    // AMM reserves must be >> total_notional to avoid unrealistic slippage.
    // reserve_x = 100_000 makes a 100-unit trade only 0.1% of pool depth.
    let amm_reserve_x = 100_000.0;
    let amm_reserve_y = amm_reserve_x * p.arrival_price;

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
            amm: ConstantProductAMM::new(amm_reserve_x, amm_reserve_y),
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
        println!(
            "  Total executed  : {:.4}",
            strat.metrics.total_executed()
        );

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

        if let Some(tx) = path_tx {
            let _ = tx.send(DebugEvent::PathStart { path_id: path_idx });
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
            let ac_pct = strat
                .metrics
                .ac_objective_percent(p.lambda)
                .unwrap_or(0.0);
            accumulators[i].shortfall_pcts.push(shortfall_pct);
            accumulators[i].ac_obj_pcts.push(ac_pct);
        }

        if let Some(tx) = path_tx {
            let _ = tx.send(DebugEvent::PathEnd { path_id: path_idx });
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

    // ── Observability setup ────────────────────────────────────────
    let debug_state = if debug {
        let (tx, handle) = init_debug_channel();
        Some((tx, handle))
    } else {
        None
    };
    let event_tx = debug_state.as_ref().map(|(tx, _)| tx);

    let params = SimParams {
        total_notional: 1000.0,
        base_chunk: 100.0,
        sigma_ref: 0.02,
        liquidity_ref: 500.0,
        eta: 0.1,
        lambda: 0.5,
        max_slippage_pct: 0.5,
        reserve_frac: 0.0,
        arrival_price: 100.0,
    };

    let base_seed: u64 = 42;

    println!("Model: {model}");

    if let Some(tx) = event_tx {
        let _ = tx.send(DebugEvent::SimulationStart {
            model: model.clone(),
            paths: num_paths,
        });
    }

    if num_paths > 1 {
        // ── Monte Carlo mode ───────────────────────────────────────
        run_monte_carlo(&params, num_paths, base_seed, &model, event_tx, debug_paths);
    } else {
        // ── Single-path mode (backward compatible) ─────────────────
        if let Some(tx) = event_tx {
            let _ = tx.send(DebugEvent::PathStart { path_id: 0 });
        }

        let strategies = build_strategies(&params);
        let max_steps = 500;
        let simulator = build_simulator(&model, params.arrival_price, base_seed, max_steps);
        let mut runner = MultiStrategyRunner::new(strategies, simulator, max_steps);
        if let Some(tx) = event_tx {
            runner = runner.with_event_sender(tx.clone());
        }

        if let Err(e) = runner.run() {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(DebugEvent::PathEnd { path_id: 0 });
        }

        println!("  Avg Volatility: {:.4}%", runner.avg_volatility() * 100.0);
        print_single_run_summary(runner.strategies(), params.lambda);
    }

    // ── Observability teardown ──────────────────────────────────────
    if let Some(tx) = event_tx {
        let _ = tx.send(DebugEvent::SimulationEnd);
    }
    if let Some((tx, handle)) = debug_state {
        drop(tx); // close the channel
        let _ = handle.join(); // flush remaining events
    }
}
