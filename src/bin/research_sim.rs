use strata_exec::analytics::metrics::ExecutionMetrics;
use strata_exec::engine::risk::RiskManager;
use strata_exec::engine::scheduler::{ExecutionMode, Scheduler};
use strata_exec::events::event::Event as DebugEvent;
use strata_exec::market::garch::GarchSimulator;
use strata_exec::market::gbm::{GbmSimulator, PriceSimulator};
use strata_exec::market::agg_trades::AggTradesDay;
use strata_exec::market::lob_replay::LobReplay;
use strata_exec::market::vpin::calibrate_bucket_size;
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

fn parse_lob_flag() -> bool {
    std::env::args().any(|a| a == "--lob")
}

fn parse_include_regime_ac_flag() -> bool {
    std::env::args().any(|a| a == "--include-regime-ac")
}

fn parse_bucket_size_arg() -> f64 {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--bucket-size" {
            if let Some(val) = args.get(i + 1) {
                return val.parse::<f64>().unwrap_or(0.0).max(0.0);
            }
        }
        i += 1;
    }
    0.0 // 0.0 means auto-calibrate
}

fn parse_lob_data_arg() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--lob-data" {
            if let Some(val) = args.get(i + 1) {
                return Some(val.to_string());
            }
        }
        i += 1;
    }
    None
}

fn parse_agg_trades_arg() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--agg-trades" {
            if let Some(val) = args.get(i + 1) {
                return Some(val.to_string());
            }
        }
        i += 1;
    }
    None
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
///
/// Both GBM and GARCH are initialised with the same annualised volatility
/// `sigma` so that cross-model comparisons start from the same regime.
/// GARCH drifts away from `sigma` in subsequent steps (clustering); GBM stays.
fn build_simulator(
    model: &str,
    initial_price: f64,
    sigma: f64,
    seed: u64,
    steps: usize,
) -> Box<dyn PriceSimulator> {
    let dt = 1.0 / steps as f64; // T = 1 year assumed for scaling

    match model {
        "garch" => Box::new(GarchSimulator::new(
            initial_price,
            0.05,            // mu
            0.000002,        // omega
            0.08,            // alpha
            0.90,            // beta
            sigma * sigma,   // sigma2_init = sigma_ref² → same starting vol as GBM
            dt,
            seed,
        )),
        _ => Box::new(GbmSimulator::new(initial_price, 0.05, sigma, dt, seed)),
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
    /// Permanent impact coefficient γ (dimensionless fraction).
    /// Selling full inventory causes gamma * 100% permanent price depression.
    gamma: f64,
    max_slippage_pct: f64,
    reserve_frac: f64,
    arrival_price: f64,
    daily_volume: f64,
    impact_coefficient: f64,
    transient_rho: f64,
    max_steps: usize,
}

fn build_strategies(p: &SimParams, include_regime_ac: bool) -> Vec<StrategyInstance> {
    let mut modes: Vec<(&str, ExecutionMode)> = vec![
        ("TWAP", ExecutionMode::Twap),
        ("Heuristic", ExecutionMode::Heuristic),
        ("Optimal", ExecutionMode::Optimal),
        ("AdaptiveOptimal", ExecutionMode::AdaptiveOptimal),
    ];
    if include_regime_ac {
        modes.push(("RegimeAC", ExecutionMode::RegimeAc));
    }

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
                Some(p.max_steps),
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
            gamma: p.gamma,
            completed: false,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Summary printing
// ---------------------------------------------------------------------------

fn print_single_run_summary(strategies: &[StrategyInstance], lambda: f64, fill_mode: &str) {
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

        // ── Cost decomposition ──
        print_cost_decomposition(&strat.metrics, fill_mode);

        // ── VPIN analysis (Phase 3 Track A) ──
        print_vpin_analysis(&strat.metrics);

        // ── Regime analysis (Phase 3 Track B — RegimeAC only) ──
        print_regime_analysis(strat);

        println!();
    }

    println!("CSV written to results/comparison.csv");
}

fn spread_label(fill_mode: &str) -> &'static str {
    if fill_mode == "formula-based" {
        "Spread* (proxy)"
    } else {
        "Spread  (real LOB)"
    }
}

/// Print the five-component IS decomposition table for one strategy.
/// Print the VPIN analysis block for one strategy (Phase 3 Track A).
/// Only printed when VPIN data was collected (non-empty vpin_at_execution).
fn vpin_percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_vpin_analysis(metrics: &strata_exec::analytics::metrics::ExecutionMetrics) {
    let vpin_data = &metrics.vpin_at_execution;
    if vpin_data.is_empty() {
        return;
    }

    let avg_vpin = vpin_data.iter().sum::<f64>() / vpin_data.len() as f64;

    // Compute distribution statistics and set threshold = P75.
    let mut sorted = vpin_data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let v_min    = sorted.first().copied().unwrap_or(0.0);
    let v_p25    = vpin_percentile(&sorted, 0.25);
    let v_median = vpin_percentile(&sorted, 0.50);
    let v_p75    = vpin_percentile(&sorted, 0.75);
    let v_p90    = vpin_percentile(&sorted, 0.90);
    let v_max    = sorted.last().copied().unwrap_or(0.0);

    println!("\n  ===== VPIN Distribution =====");
    println!("  Min    : {v_min:.4}");
    println!("  P25    : {v_p25:.4}");
    println!("  Median : {v_median:.4}");
    println!("  P75    : {v_p75:.4}");
    println!("  P90    : {v_p90:.4}");
    println!("  Max    : {v_max:.4}");

    // Use P75 as the threshold so ~25% of fills are always classified as high-toxicity.
    let threshold = v_p75;
    let high_frac = metrics.high_vpin_trade_fraction(threshold);
    let (low_is, high_is) = metrics.conditional_shortfall_percent(threshold);

    println!("\n  ===== VPIN Analysis (P75 threshold {threshold:.4}) =====");
    println!("  Avg VPIN           : {avg_vpin:.4}");
    println!("  High-toxicity fills: {:.1}%", high_frac * 100.0);
    match (low_is, high_is) {
        (Some(l), Some(h)) => {
            let premium = h - l;
            println!("  IS (low  VPIN)     : {l:+.4}%");
            println!("  IS (high VPIN)     : {h:+.4}%");
            println!("  Toxicity premium   : {premium:+.4}%");
        }
        (Some(l), None) => println!("  IS (low  VPIN)     : {l:+.4}%  (no high-VPIN fills)"),
        (None, Some(h)) => println!("  IS (high VPIN)     : {h:+.4}%  (no low-VPIN fills)"),
        (None, None) => println!("  (insufficient VPIN data for conditional IS)"),
    }
}

fn print_regime_analysis(strat: &strata_exec::research::multi_runner::StrategyInstance) {
    if strat.name != "RegimeAC" {
        return;
    }

    let dist = strat.metrics.regime_distribution();
    if dist.is_empty() {
        return;
    }

    println!("\n  ===== Regime Distribution =====");
    for (label, count, frac) in &dist {
        println!("  {:<22}: {:>5} fills ({:.2}%)", label, count, frac * 100.0);
    }

    let by_regime = strat.metrics.is_by_regime();
    if !by_regime.is_empty() {
        println!("\n  ===== IS By Regime =====");
        println!("  {:<22}  {:>10}  {:>6}", "Regime", "Mean IS%", "Fills");
        println!("  {}", "-".repeat(44));
        for (label, is_pct, count) in &by_regime {
            println!("  {:<22}  {:>+10.4}%  {:>6}", label, is_pct, count);
        }
    }

    // Print calibration anchors from the underlying RegimeAcStrategy.
    if let Some(ref rac) = strat.scheduler.regime_ac {
        println!(
            "\n  RegimeAC warm-up: {} steps, sigma_ref set to {:.4}% per step",
            rac.warmup_steps,
            rac.sigma_ref * 100.0,
        );
        println!("  Spread ref (classification anchor): {:.6}", rac.spread_ref);
    }
}

fn print_cost_decomposition(
    metrics: &strata_exec::analytics::metrics::ExecutionMetrics,
    fill_mode: &str,
) {
    if let Some(d) = metrics.decompose() {
        let total_abs = d.total.abs().max(1e-12);
        println!("\n  ===== IS Cost Decomposition =====");
        if fill_mode == "formula-based" {
            println!("  (* spread uses proxy: 0.5×σ×√dt×price)");
        }
        println!(
            "  {:<24} {:>12}  {:>7}",
            "Component", "Cost ($)", "Share %"
        );
        println!("  {}", "-".repeat(48));

        let rows: &[(&str, f64)] = &[
            (spread_label(fill_mode), d.spread_cost),
            ("Temporary impact",      d.temporary_impact),
            ("Permanent impact",      d.permanent_impact),
            ("Timing cost",           d.timing_cost),
            ("Opportunity cost",      d.opportunity_cost),
        ];
        for (label, val) in rows {
            let share = val / total_abs * 100.0;
            println!("  {:<24} {:>12.4}  {:>+7.2}%", label, val, share);
        }
        println!("  {}", "-".repeat(48));
        println!(
            "  {:<24} {:>12.4}  ({:.4}% of notional)",
            "Total IS", d.total, d.total_as_pct
        );
    }
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
    let use_lob = parse_lob_flag();
    let include_regime_ac = parse_include_regime_ac_flag();
    let lob_data_path = parse_lob_data_arg();
    let agg_trades_path = parse_agg_trades_arg();
    let bucket_size_arg = parse_bucket_size_arg();
    let calibrate_path = parse_calibrate_flag();

    // ── Observability setup ────────────────────────────────────────
    let debug_state = if debug || progress {
        let (tx, handle) = init_debug_channel();
        Some((tx, handle))
    } else {
        None
    };
    let event_tx = debug_state.as_ref().map(|(tx, _)| tx);

    let max_steps = 500;
    let mut params = SimParams {
        total_notional: 1_000_000.0,
        base_chunk: 10_000.0,
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
        max_steps,
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

    // Validate and report the historical LOB file early (before the path loop).
    // When --lob-data is provided, align max_steps to the snapshot count so that
    // every simulation step maps 1:1 to a LOB snapshot. This ensures the full
    // VPIN series is traversed (not just the first 500/2880 snapshots).
    // spread_ref for RegimeAC regime classification — anchored to first LOB snapshot.
    let mut spread_ref_for_display: Option<f64> = None;
    if let Some(ref p) = lob_data_path {
        match LobReplay::from_csv(p) {
            Ok(mut r) => {
                let n = r.len();
                println!("LOB data : {n} snapshots loaded from {p}");
                params.max_steps = n;   // align simulation horizon to data coverage
                // Re-scale base_chunk so strategies trade one slice per step across the
                // full day, giving VPIN enough trades to warm up its bucket window.
                params.base_chunk = params.total_notional / n as f64;

                // Capture first snapshot spread for RegimeAC spread_ref display.
                if let Some(first_snap) = r.next_snapshot() {
                    if let Some(hs) = first_snap.half_spread() {
                        spread_ref_for_display = Some(hs);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error loading --lob-data CSV '{p}': {e}");
                std::process::exit(1);
            }
        }
    }

    // Load aggTrades file if provided. Printed stats come from AggTradesDay::from_csv.
    let agg_trades_day: Option<AggTradesDay> = if let Some(ref p) = agg_trades_path {
        match AggTradesDay::from_csv(p) {
            Ok(day) => Some(day),
            Err(e) => {
                eprintln!("Error loading --agg-trades CSV '{p}': {e}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // Determine effective VPIN bucket size.
    // Priority: real aggTrades volume > explicit --bucket-size > placeholder fallback.
    const TARGET_BUCKETS: usize = 50;
    let (effective_bucket_size, bucket_label) = if let Some(ref agg) = agg_trades_day {
        let s = agg.calibrated_bucket_size(TARGET_BUCKETS);
        let label = format!("{s:.1} BTC (auto-calibrated, {TARGET_BUCKETS} buckets/day)");
        (s, label)
    } else if bucket_size_arg > 0.0 {
        let s = bucket_size_arg;
        let label = format!("{s:.1} BTC (explicit --bucket-size)");
        (s, label)
    } else {
        let s = calibrate_bucket_size(50_000.0, TARGET_BUCKETS);
        let label = format!(
            "{s:.1} BTC (placeholder — pass --agg-trades for real calibration)"
        );
        (s, label)
    };

    // VPIN source label.
    let vpin_source = if agg_trades_day.is_some() && lob_data_path.is_some() {
        format!("real aggTrades ({:.1} BTC/bucket, {TARGET_BUCKETS} buckets)", effective_bucket_size)
    } else {
        "synthetic (depth-change proxy)".to_string()
    };

    let fill_label = if lob_data_path.is_some() {
        "historical LOB (Binance replay)"
    } else if use_lob {
        "synthetic LOB depth-consuming"
    } else {
        "formula-based"
    };
    println!("Model: {model}");
    println!("Fill:  {fill_label}");
    println!("VPIN bucket: {bucket_label}");
    println!("VPIN source: {vpin_source}");
    if let Some(sr) = spread_ref_for_display {
        println!("Spread ref: ${sr:.4} per unit (first snapshot, RegimeAC anchor)");
    }
    if include_regime_ac {
        println!();
        println!("⚠  RegimeAC: greedy receding-horizon formulation.");
        println!("    IS not comparable to global Optimal schedule.");
        println!("    Root cause: myopic step-by-step diverges from");
        println!("    globally optimal sinh schedule over 2880 steps.");
        println!("    See Phase 3 findings for full analysis.");
        println!();
    }

    if let Some(_tx) = event_tx {
        // Nothing for now, could emit SimulationStart if variant existed
    }

    if experiments {
        // ── Experiment Parameter Sweeps ───────────────────────────────
        println!("\n{:=<70}", "");
        println!("  EXPERIMENT PARAMETER SWEEPS  ({num_paths} paths per config)");
        println!("{:=<70}\n", "");

        let start = std::time::Instant::now();
        if let Err(e) = strata_exec::research::experiments::run_all_sweeps(num_paths, event_tx) {
            eprintln!("Experiment Sweeps Failed: {e}");
            std::process::exit(1);
        }
        let elapsed = start.elapsed();

        println!("\n{:=<70}", "");
        println!("  SWEEP COMPLETE");
        println!("{:=<70}", "");
        println!("  Elapsed    : {:.2}s", elapsed.as_secs_f64());
        println!("  Outputs    :");
        println!("    • results/sweep_horizon.csv");
        println!("    • results/sweep_trade_chunks.csv");
        println!("    • results/sweep_volatility.csv");
        println!("    • results/sweep_impact.csv");
        println!();
        
    } else {
        // ── Standard Multi-Strategy Simulator ─────────────────────────
        let mut overall_avg_vol = 0.0_f64;
        let base_seed: u64 = 42;

    for path_id in 0..num_paths {
        if let Some(tx) = event_tx {
            let _ = tx.send(DebugEvent::PathStart { path_id });
        }

        let strategies = build_strategies(&params, include_regime_ac);
        let simulator = build_simulator(
            &model,
            params.arrival_price,
            params.sigma_ref,
            base_seed + path_id as u64,
            params.max_steps,
        );
        let mut runner = if let Some(ref path) = lob_data_path {
            // Reload the replay for each path so every path starts from snapshot 0.
            match LobReplay::from_csv(path) {
                Ok(replay) => {
                    let mut r = MultiStrategyRunner::new(strategies, simulator, params.max_steps)
                        .with_historical_lob(replay)
                        .with_vpin(effective_bucket_size, TARGET_BUCKETS);
                    // Wire real aggTrades on the first path; subsequent paths reuse the
                    // pre-computed real_vpin_series (rebuilt from scratch each path anyway).
                    if let Some(ref agg) = agg_trades_day {
                        // Clone is cheap relative to file I/O; trades vec is reused read-only.
                        r = r.with_agg_trades(AggTradesDay {
                            trades: agg.trades.iter().map(|t| strata_exec::market::agg_trades::AggTrade {
                                timestamp_ms: t.timestamp_ms,
                                price: t.price,
                                quantity_btc: t.quantity_btc,
                                is_sell_aggressor: t.is_sell_aggressor,
                            }).collect(),
                            date: agg.date.clone(),
                            skipped_rows: agg.skipped_rows,
                        });
                    }
                    r
                }
                Err(e) => {
                    eprintln!("ERROR reloading --lob-data for path {path_id}: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            MultiStrategyRunner::new(strategies, simulator, params.max_steps)
                .with_lob(use_lob)
                .with_vpin(effective_bucket_size, TARGET_BUCKETS)
        };

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
            print_single_run_summary(runner.strategies(), params.lambda, fill_label);
        }
    }

        if num_paths > 1 {
            let avg_vol = overall_avg_vol / num_paths as f64;
            println!("\n  Overall Avg Volatility ({num_paths} paths): {:.4}%", avg_vol * 100.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_label_is_proxy_in_formula_mode_and_real_in_lob_mode() {
        assert_eq!(spread_label("formula-based"), "Spread* (proxy)");
        assert_eq!(spread_label("synthetic LOB depth-consuming"), "Spread  (real LOB)");
        assert_eq!(spread_label("historical LOB (Binance replay)"), "Spread  (real LOB)");
    }
}
