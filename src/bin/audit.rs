/// AUDIT BINARY — read-only instrumentation, no production code changed.
///
/// Runs three investigations and prints a verdict table.
use strata_exec::execution::impact::SquareRootImpact;
use strata_exec::execution::transient_impact::TransientImpactTracker;
use strata_exec::market::garch::GarchSimulator;
use strata_exec::market::gbm::{GbmSimulator, PriceSimulator};
use strata_exec::market::liquidity::LiquiditySnapshot;
use strata_exec::strategies::heuristic::HeuristicStrategy;
use strata_exec::strategies::optimal::AlmgrenChriss;
use strata_exec::strategies::optimal::ExecutionModel;

// ──────────────────────────────────────────────────────────────────────────────
// INVESTIGATION 1 — Heuristic Strategy Correctness
// ──────────────────────────────────────────────────────────────────────────────

fn investigation_1() {
    println!("\n{:=>80}", "");
    println!("INVESTIGATION 1: Heuristic Strategy Correctness");
    println!("{:=>80}\n", "");

    // Parameters mirroring research_sim defaults
    let total_notional = 1_000_000.0_f64;
    let base_chunk = 10_000.0_f64;
    let sigma_ref = 0.02_f64;
    let liquidity_ref = 500.0_f64;

    println!("=== 1.3 Hardcoded reference values ===");
    println!("  sigma_ref     = {sigma_ref}  (set in research_sim SimParams / SweepConfig::default)");
    println!("  liquidity_ref = {liquidity_ref}  (same source)");
    println!("  base_chunk    = {base_chunk}");
    println!("  These values are FIXED at construction. They are NEVER updated during a run.");
    println!("  HeuristicStrategy has no setter; sigma_ref/liquidity_ref are pub fields");
    println!("  but MultiStrategyRunner and Scheduler never write them after creation.\n");

    println!("=== 1.1 / 1.2  Per-step trace (20 GARCH steps, seed=42) ===");
    println!("  Formula:");
    println!("    liq              = sqrt(reserve_in * reserve_out)");
    println!("    liquidity_factor = clamp(liq / liquidity_ref, 0.25, 2.0)");
    println!("    volatility_factor= clamp(sigma_ref / max(vol, 1e-8), 0.25, 2.0)");
    println!("    adaptive_chunk   = base_chunk * liquidity_factor * volatility_factor");
    println!("    final_chunk      = min(adaptive_chunk, remaining)\n");

    let dt = 1.0 / 500.0_f64; // same as research_sim 500 steps
    let mut garch = GarchSimulator::new(
        100.0, 0.05, 0.000002, 0.08, 0.90,
        sigma_ref * sigma_ref, // sigma2_init = sigma_ref^2
        dt,
        42,
    );

    let heuristic = HeuristicStrategy::new(base_chunk, sigma_ref, liquidity_ref, total_notional);
    let mut remaining = total_notional;

    // Fixed reserves (mean value from MultiStrategyRunner OU process at t=0)
    let reserve_mean = 500.0_f64;

    let header = format!(
        "{:>4}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}  {:>14}",
        "Step", "sigma(vol)", "liq(depth)", "liq_factor", "vol_factor", "chunk", "remaining"
    );
    println!("{header}");
    println!("{:->width$}", "", width = header.len());

    let mut clamp_count = 0usize;

    // Build AC schedule for side-by-side comparison (inv. 1.5)
    let ac_sigma = sigma_ref; // use sigma_ref because vol is ~0.02 at start
    let ac = AlmgrenChriss::new(total_notional, 0.001, 1e-4, ac_sigma, Some(20))
        .expect("AC construction failed");
    let ac_schedule = ac.schedule();

    let mut heur_chunks: Vec<f64> = Vec::new();

    for step in 0..20 {
        let _price = garch.step();
        let vol = garch.volatility(); // returns sqrt(sigma2) = sigma

        // Use reserve_mean for both reserves (static liquidity for clarity)
        let liq_snap = LiquiditySnapshot::new(reserve_mean, reserve_mean);
        let liq_depth = liq_snap.depth_metric(); // sqrt(500*500) = 500

        let liq_factor_raw = liq_depth / liquidity_ref;
        let vol_factor_raw = sigma_ref / vol.max(1e-8);

        let liq_factor = liq_factor_raw.clamp(0.25, 2.0);
        let vol_factor = vol_factor_raw.clamp(0.25, 2.0);

        let adaptive = base_chunk * liq_factor * vol_factor;
        let chunk = adaptive.min(remaining);

        // Count clamp activations
        if liq_factor_raw < 0.25 || liq_factor_raw > 2.0 || vol_factor_raw < 0.25 || vol_factor_raw > 2.0 {
            clamp_count += 1;
        }

        println!(
            "{:>4}  {:>10.6}  {:>12.4}  {:>14.6}  {:>14.6}  {:>14.4}  {:>14.2}",
            step + 1,
            vol,
            liq_depth,
            liq_factor,
            vol_factor,
            chunk,
            remaining,
        );

        heur_chunks.push(chunk);
        remaining -= chunk;
        if remaining < 0.0 {
            remaining = 0.0;
        }
    }

    println!();
    println!("=== 1.4 Clamp activation count (out of 20 steps) ===");
    println!("  liq_factor_raw is ALWAYS exactly 1.0 (liq=500, ref=500) → NEVER clamped.");
    println!("  vol_factor_raw = sigma_ref / sigma_t = 0.02 / sigma_t");
    println!("    When sigma_t < 0.01  → factor > 2.0 → upper clamp at 2.0");
    println!("    When sigma_t > 0.04  → factor < 0.25 (wait: 0.02/0.04=0.5, no; need sigma_t > 0.08 for lower clamp)");
    println!("    Lower clamp (0.25) fires when sigma_t > 0.08");
    println!("  Steps where a clamp activated: {clamp_count} / 20");
    println!("  (Liquidity factor cannot clamp with default params — depth == ref exactly)");

    println!();
    println!("=== 1.5 Side-by-side: Heuristic vs Almgren-Chriss (AC) ===");
    println!("  AC params: eta=0.001, lambda=1e-4, sigma=0.02, horizon=20");
    println!("  Note: AC is front-loaded; Heuristic adapts to vol each step.\n");
    println!(
        "{:>4}  {:>14}  {:>14}  {:>10}",
        "Step", "Heuristic_chunk", "AC_chunk", "Heur>AC?"
    );
    for i in 0..20 {
        let h = heur_chunks.get(i).copied().unwrap_or(0.0);
        let a = ac_schedule.get(i).copied().unwrap_or(0.0);
        println!(
            "{:>4}  {:>14.4}  {:>14.4}  {:>10}",
            i + 1,
            h,
            a,
            if h > a { "YES(more)" } else { "no(less)" }
        );
    }

    println!();
    println!("=== 1.6 Economic Verdict ===");
    println!("  The heuristic formula is economically directionally correct:");
    println!("  - High vol  → vol_factor DECREASES chunk (trades LESS when risky). ✓");
    println!("  - Low liq   → liq_factor DECREASES chunk (trades LESS when illiquid). ✓");
    println!("  HOWEVER there are two structural problems:");
    println!("  [A] sigma_ref and liquidity_ref are NEVER updated during execution.");
    println!("      They represent baseline conditions at the START of the run, not");
    println!("      the current regime. In a GARCH path with vol clustering, sigma_ref");
    println!("      quickly becomes stale — the ratio sigma_ref/sigma_t is applied to");
    println!("      a baseline that was calibrated for 2% vol but the market may be at 8%.");
    println!("  [B] The clamp [0.25, 2.0] means the heuristic can ONLY vary chunk size");
    println!("      by 4x total (2.0/0.25). In extreme vol regimes the clamp acts as a");
    println!("      hard floor/ceiling, disconnecting the signal from the formula entirely.");
    println!("  [C] vol_factor uses sigma_ref/sigma_t (inverse relationship). This means");
    println!("      in HIGH volatility the heuristic trades LESS (cautious). AC with");
    println!("      higher lambda also front-loads in high vol. The heuristic therefore");
    println!("      trades in the OPPOSITE direction to AC under high vol: AC speeds up,");
    println!("      Heuristic slows down. This is the primary reason heuristic tends to");
    println!("      underperform AC in volatile GARCH regimes — it is economically sensible");
    println!("      but implements a risk-aversion strategy, not a cost-minimization one.");
}

// ──────────────────────────────────────────────────────────────────────────────
// INVESTIGATION 2 — Almgren-Chriss Implementation Completeness
// ──────────────────────────────────────────────────────────────────────────────

fn investigation_2() {
    println!("\n{:=>80}", "");
    println!("INVESTIGATION 2: Almgren-Chriss Implementation Completeness");
    println!("{:=>80}\n", "");

    println!("=== 2.1 Parameter struct ===");
    println!("  AlmgrenChriss {{ total_inventory, eta, lambda, sigma, horizon }}");
    println!("  SINGLE impact parameter: eta (η) only.");
    println!("  There is NO gamma (γ) permanent impact coefficient.");
    println!("  This is a ONE-PARAMETER impact model, not the full two-parameter AC model.");
    println!();
    println!("  The docstring says: 'temporary market-impact cost (eta) and timing risk (lambda * sigma²)'");
    println!("  Permanent impact is ABSENT from the struct and from all formulas.");

    println!();
    println!("=== 2.2 κ formula ===");
    println!("  κ = √(λ·σ²/η)");
    println!("  Uses η ALONE (no γ). This is correct for the temporary-impact-only AC variant.");
    println!("  In the full AC model, κ = √(λ·σ²/η) also — permanent impact γ enters the");
    println!("  cost expression but NOT κ itself. So the κ formula is correct for η-only.");

    println!();
    println!("=== 2.3 Permanent impact on remaining inventory ===");
    println!("  ABSENT. There is no gamma parameter, no price path update, and no mechanism");
    println!("  by which a completed trade shifts the 'arrival price' for future trades.");
    println!("  In a proper AC model, permanent impact γ·q_k depresses the mid-price for");
    println!("  ALL subsequent steps. Here, every trade's execution_price is:");
    println!("    exec_price = market_price * (1 - total_impact_fraction)");
    println!("  where market_price comes from the GBM/GARCH simulator — it is INDEPENDENT");
    println!("  of what was previously traded. Permanent impact is not modelled at all.");

    println!();
    println!("=== 2.4 Temporary vs permanent impact separation ===");
    println!("  The execution pipeline in MultiStrategyRunner uses:");
    println!("    base_impact (SquareRootImpact) + transient_impact (TransientImpactTracker)");
    println!("  SquareRootImpact: Y·σ·√(Q/V) — this is TEMPORARY impact per-trade.");
    println!("  TransientImpactTracker: exponential decay of past base impacts.");
    println!("  Neither represents permanent impact in the AC sense.");
    println!("  The two-level decomposition is: instantaneous + lingering transient.");
    println!("  There is NO third term for permanent (non-decaying) price shift.");

    println!();
    println!("=== 2.5 Single-trade execution sequence ===");
    println!("  Trace for step k (from MultiStrategyRunner::run):");
    println!("    1. price_before  = simulator.step()  ← pure GBM/GARCH, no AC price model");
    println!("    2. base_impact   = SquareRootImpact::compute_impact(traded, volatility)");
    println!("                     = Y * σ * sqrt(Q/V)   [dimensionless fraction]");
    println!("    3. transient_impact = TransientImpactTracker::current_impact(real_time)");
    println!("                     = Σ past_base_impact_i * exp(-ρ*(t - t_i))");
    println!("    4. total_impact  = base_impact + transient_impact");
    println!("    5. exec_price    = price * (1.0 - total_impact)");
    println!("    6. slippage_cost = exec_price - price  [negative for a sell]");
    println!("    7. transient_tracker.record_trade(real_time, base_impact)");
    println!("       ← base_impact is stored, decays for FUTURE steps");
    println!("    8. Next step: simulator.step() gives a FRESH price from GBM/GARCH.");
    println!("       No permanent impact propagates into the price model itself.");

    // Demonstrate with concrete numbers
    println!();
    println!("  Concrete example (params from research_sim defaults):");
    let daily_volume = 5_000_000.0_f64;
    let impact_coeff = 0.5_f64;
    let rho = 50.0_f64;
    let vol = 0.02_f64;
    let price = 100.0_f64;
    let traded = 10_000.0_f64;

    let impact_model = SquareRootImpact::new(daily_volume, impact_coeff);
    let mut tracker = TransientImpactTracker::new(rho);

    let dt = 1.0 / 500.0_f64;

    let base_imp = impact_model.compute_impact(traded, vol);
    let trans_imp_before = tracker.current_impact(0.0 * dt);
    let total_imp = base_imp + trans_imp_before;
    let exec = price * (1.0 - total_imp);
    println!("    traded         = {traded:.0}");
    println!("    vol            = {vol}");
    println!("    base_impact    = {base_imp:.6}  ({:.4} bps)", base_imp * 10000.0);
    println!("    transient(t=0) = {trans_imp_before:.6}  (no prior trades)");
    println!("    total_impact   = {total_imp:.6}");
    println!("    exec_price     = {exec:.6}  (from {price})");
    tracker.record_trade(0.0 * dt, base_imp);

    // Step 2: what persists into next step
    let trans_next = tracker.current_impact(1.0 * dt);
    println!("    transient at t+1dt ({dt:.5}) = {trans_next:.8}");
    println!("    → price at t+1 from simulator is INDEPENDENT of this transient");
    println!("      (simulator has no knowledge of trades; permanent impact = 0)");

    println!();
    println!("=== 2.6 Verdict: single-parameter approximation ===");
    println!("  This is a SINGLE-PARAMETER APPROXIMATION of Almgren-Chriss.");
    println!("  - η (temporary) : present, used correctly in κ and schedule");
    println!("  - γ (permanent) : ABSENT entirely");
    println!("  The schedule shape (sinh-based, front-loaded) is correct.");
    println!("  But the cost accounting systematically understates true execution cost");
    println!("  because each trade's permanent price impact is not carried forward.");
    println!("  In a real market, selling 1% of daily volume should depress all future");
    println!("  prices slightly. Here it does not. The model is AC in name and schedule");
    println!("  shape only — cost measurement is not full two-parameter AC.");
}

// ──────────────────────────────────────────────────────────────────────────────
// INVESTIGATION 3 — Price Simulation Alignment
// ──────────────────────────────────────────────────────────────────────────────

fn investigation_3() {
    println!("\n{:=>80}", "");
    println!("INVESTIGATION 3: Price Simulation Alignment");
    println!("{:=>80}\n", "");

    let dt = 1.0 / 500.0_f64; // research_sim default: 500 steps, T=1yr
    let seed = 42_u64;
    let initial_price = 100.0_f64;
    let sigma2_init = 0.02_f64 * 0.02_f64; // = 0.0004
    let sigma_ref = 0.02_f64;

    println!("=== 3.1 What does simulator.volatility() return? ===");
    println!("  GbmSimulator::volatility() → self.sigma");
    println!("    self.sigma = sigma_ref = {sigma_ref}  (the annualised σ, NOT σ²)");
    println!("    For GBM this is constant; returns the same value every step.");
    println!();
    println!("  GarchSimulator::volatility() → self.sigma2.sqrt()");
    println!("    self.sigma2 = current conditional VARIANCE (σ²)");
    println!("    Returns √σ² = σ (annualised standard deviation, same scale as GBM)");
    println!("    Initial: sigma2_init = {sigma2_init}  →  vol = {:.4}", sigma2_init.sqrt());
    println!("  VERDICT: both simulators return σ (std dev), NOT σ². Scale is consistent.");

    println!();
    println!("=== 3.2 Volatility passed to strategies ===");
    println!("  MultiStrategyRunner::run() calls:");
    println!("    let volatility = self.simulator.volatility();");
    println!("    strat.scheduler.on_block(volatility, &liquidity)");
    println!("  This is the raw return from volatility() — i.e. σ (annualised std dev).");
    println!("  No rescaling occurs between simulator output and strategy input.");

    println!();
    println!("=== 3.3 GARCH path trace: σ², σ, actual return (20 steps) ===");
    println!("  dt = {dt:.6}  (1/500 year ≈ 0.5 trading days)");
    println!("  Expected |return| ≈ σ·√dt = {:.6}  ({:.4} bps per step at init vol)",
        sigma_ref * dt.sqrt(), sigma_ref * dt.sqrt() * 10000.0);
    println!();
    println!(
        "{:>4}  {:>12}  {:>12}  {:>14}  {:>12}",
        "Step", "sigma2", "sigma(vol)", "price_return", "1sigma_band"
    );

    let mut garch = GarchSimulator::new(
        initial_price, 0.05, 0.000002, 0.08, 0.90,
        sigma2_init, dt, seed,
    );

    let mut prev_price = initial_price;
    let mut return_outside_1sigma = 0usize;

    for step in 0..20 {
        let new_price = garch.step();
        let sigma2 = garch.sigma2;
        let sigma = sigma2.sqrt();
        let ret = (new_price - prev_price) / prev_price;
        let one_sigma = sigma * dt.sqrt();
        let outside = ret.abs() > one_sigma;
        if outside {
            return_outside_1sigma += 1;
        }
        println!(
            "{:>4}  {:>12.8}  {:>12.8}  {:>14.8}  {:>12.8}  {}",
            step + 1,
            sigma2,
            sigma,
            ret,
            one_sigma,
            if outside { "<-- OUTSIDE 1σ" } else { "" }
        );
        prev_price = new_price;
    }
    println!("  Steps outside 1σ band: {return_outside_1sigma}/20 (expect ~6-7 at 1σ = 68% CI)");

    println!();
    println!("=== 3.4 dt value and time interpretation ===");
    println!("  dt = 1.0 / max_steps = 1.0 / 500 = {dt:.6}");
    println!("  T = 1 year  →  one step ≈ 1/500 year ≈ {:.2} trading days", 252.0 / 500.0);
    println!("  or ≈ {:.1} hours (252-day year, 6.5hr session)", 252.0 * 6.5 / 500.0);
    println!("  Approximately 30-minute intraday bars.");

    println!();
    println!("=== 3.5 Transient impact tracker: decay rate and half-life ===");
    let rho = 50.0_f64;
    let steps_to_1pct = -(0.01f64).ln() / rho / dt; // steps for exp(-rho*t) < 0.01
    let steps_to_half = (0.5f64).ln().abs() / rho / dt;
    println!("  rho = {rho}");
    println!("  dt  = {dt:.6}  (per step)");
    println!("  real_time per step = step_index * dt  (0 to 1.0 over 500 steps)");
    println!();
    println!("  Decay: exp(-ρ·Δt) per real-time unit.");
    println!("  Between consecutive steps: Δ_real_time = dt = {dt:.6}");
    println!("    decay per step = exp(-{rho} * {dt:.6}) = {:.6}", (-rho * dt).exp());
    println!("  Half-life in steps: ln(2)/ρ/dt = {steps_to_half:.2} steps");
    println!("  Decay to 1% in steps: ln(100)/ρ/dt = {steps_to_1pct:.2} steps");
    println!();
    println!("  Interpretation:");
    println!("  At rho=50 and dt=0.002, a trade's impact halves every {:.1} steps.", steps_to_half);
    println!("  With ~30min bars, half-life = {:.1} bars = {:.1} hours.", steps_to_half, steps_to_half * 0.5);
    println!("  This means impact from a trade lasts roughly {:.1}h — MUCH TOO LONG", steps_to_half * 0.5);
    println!("  for typical equity microstructure where transient impact decays in minutes.");
    println!("  However, if the model represents a LESS liquid market (e.g. crypto AMM),");
    println!("  multi-hour decay can be defensible.");
    println!();
    println!("  CRITICAL ISSUE: In MultiStrategyRunner the real_time is computed as:");
    println!("    let real_time = current_time * dt;  // current_time = step (integer)");
    println!("    tracker.record_trade(real_time, base_impact)");
    println!("    tracker.current_impact(real_time)");
    println!("  So real_time goes from 0.0 to max_steps*dt = 1.0. This is correct.");
    println!("  BUT the decay exponent is -rho * (now - event.timestamp)");
    println!("  where both now and timestamp are in [0, 1]. So the effective decay");
    println!("  is on the NORMALISED time axis, not calendar time. rho=50 means the");
    println!("  signal decays with a characteristic time of 1/50 = 0.02 normalised units");
    println!("  = 0.02 * 500 steps = 10 steps = ~5 hours. Persistent, but bounded.");

    println!();
    println!("=== 3.6 GBM vs GARCH: first 10 prices with identical seeds ===");
    println!("  NOTE: GBM and GARCH consume RNG draws differently (GARCH has extra");
    println!("  sigma2 update logic but still draws ONE z per step). However, both use");
    println!("  the SAME StdRng seed, so z[0] is identical in both.");
    println!("  GBM: fixed sigma=0.02; GARCH: sigma evolves each step.");
    println!("  Prices will DIVERGE even with the same seed because GBM uses sigma=0.20");
    println!("  (research_sim default) while GARCH starts at sigma=0.02.");
    println!("  They would only match with identical sigma AND identical sequence.\n");

    let mut gbm = GbmSimulator::new(initial_price, 0.05, 0.20, dt, seed);
    let mut garch2 = GarchSimulator::new(
        initial_price, 0.05, 0.000002, 0.08, 0.90,
        sigma2_init, dt, seed,
    );

    println!(
        "{:>4}  {:>12}  {:>12}  {:>14}  {:>14}",
        "Step", "GBM_price", "GARCH_price", "GBM_vol", "GARCH_vol"
    );
    for step in 0..10 {
        let gbm_p = gbm.step();
        let garch_p = garch2.step();
        println!(
            "{:>4}  {:>12.6}  {:>12.6}  {:>14.8}  {:>14.8}",
            step + 1,
            gbm_p,
            garch_p,
            gbm.volatility(),
            garch2.volatility(),
        );
    }
    println!("  GBM sigma=0.20 (annualised), GARCH initial sigma=0.02.");
    println!("  The 10x difference in initial vol means GBM moves ~10x more per step,");
    println!("  so paths diverge immediately. GARCH clustering has a secondary effect");
    println!("  but the initial vol mismatch is the dominant difference.");

    println!();
    println!("=== 3.7 Timescale alignment verdict ===");
    println!("  Price model   dt = 1/500 year ≈ 0.5 trading day per step  ✓");
    println!("  Volatility    signal returned as annualised σ (NOT σ²)     ✓");
    println!("  Impact model  SquareRootImpact uses annualised vol         ✓");
    println!("  Transient     rho=50 on normalised [0,1] time axis");
    println!("    → 1/rho = 0.02 normalised = 10 steps ≈ 5 trading hours  ✓ (plausible)");
    println!("  GBM sigma vs GARCH sigma: GBM uses 0.20, GARCH starts 0.02");
    println!("    → NOT directly comparable without re-calibration         ✗");
    println!("  Permanent impact (price model ← trade): ABSENT              ✗✗");
}

// ──────────────────────────────────────────────────────────────────────────────
// VERDICT TABLE
// ──────────────────────────────────────────────────────────────────────────────

fn verdict_table() {
    println!("\n{:=>80}", "");
    println!("FINAL VERDICT TABLE");
    println!("{:=>80}\n", "");

    println!(
        "{:<28}  {:<16}  {}",
        "Component", "Correct?", "Severity if wrong"
    );
    println!("{:-<78}", "");
    println!(
        "{:<28}  {:<16}  {}",
        "Heuristic formula",
        "PARTIAL",
        "MEDIUM — direction correct; stale refs & 4x clamp limit responsiveness"
    );
    println!(
        "{:<28}  {:<16}  {}",
        "AC permanent impact",
        "MISSING",
        "HIGH — γ is absent; cost is systematically understated"
    );
    println!(
        "{:<28}  {:<16}  {}",
        "AC temporary impact",
        "PARTIAL",
        "MEDIUM — η drives schedule shape correctly; cost via sqrt model (not linear η)"
    );
    println!(
        "{:<28}  {:<16}  {}",
        "simulator.volatility()",
        "CORRECT",
        "LOW — returns σ (std dev) consistently in both GBM and GARCH"
    );
    println!(
        "{:<28}  {:<16}  {}",
        "Timescale alignment",
        "PARTIAL",
        "MEDIUM — dt / impact timeline coherent; GBM 0.20 vs GARCH 0.02 initial vol mismatch"
    );
    println!(
        "{:<28}  {:<16}  {}",
        "Transient decay rate",
        "PLAUSIBLE",
        "LOW-MEDIUM — rho=50 on normalised axis → ~5hr half-life; defensible for AMM"
    );

    println!();
    println!("MUST-FIX BEFORE PHASE 2 RESULTS CAN BE TRUSTED:");
    println!();
    println!("  [1] ADD PERMANENT IMPACT (γ)");
    println!("      Every sell of size q_k should depress the simulator's reference price");
    println!("      by γ·q_k permanently. Without it, IS and AC-objective are systematically");
    println!("      too low, and strategies look better than they are in a real market.");
    println!("      Fix: add a gamma field to AlmgrenChriss and track a price adjustment");
    println!("      accumulator in MultiStrategyRunner that reduces effective_price each step.");
    println!();
    println!("  [2] ALIGN GBM AND GARCH INITIAL VOLATILITY");
    println!("      GbmSimulator uses sigma=0.20 (hardcoded in research_sim::build_simulator).");
    println!("      GarchSimulator starts at sigma2_init = sigma_ref² = 0.0004 → σ=0.02.");
    println!("      This 10x vol mismatch means GBM paths move ~10x more per step.");
    println!("      Cross-model comparison (GBM vs GARCH runs) is not apples-to-apples.");
    println!("      Fix: either pass sigma_ref to GBM or set GARCH sigma2_init = 0.04 (σ=0.20).");
    println!();
    println!("  [3] UPDATE sigma_ref IN HeuristicStrategy DURING EXECUTION");
    println!("      Currently sigma_ref is frozen at construction time. In a GARCH regime");
    println!("      where vol clusters at 2x baseline, the heuristic will systematically");
    println!("      trade at half the intended rate (vol_factor pinned at clamp floor).");
    println!("      Fix: pass a rolling vol estimate and update sigma_ref periodically,");
    println!("      or use a different normalisation (e.g. ratio to EWM vol).");
    println!();
    println!("  [4] DOCUMENT SINGLE-PARAMETER AC CHOICE");
    println!("      The code uses 'Almgren-Chriss' in comments and variable names but");
    println!("      implements only the temporary-impact variant. This is a valid model");
    println!("      (Almgren 2003 / reduced-form), but the omission of γ should be");
    println!("      explicitly noted so Phase 2 analysis doesn't incorrectly claim");
    println!("      full two-parameter AC validation.");
}

fn main() {
    investigation_1();
    investigation_2();
    investigation_3();
    verdict_table();
}
