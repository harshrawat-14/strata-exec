use std::io::Write;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::analytics::metrics::ExecutionMetrics;
use crate::engine::scheduler::Scheduler;
use crate::events::event::{Event as DebugEvent, EventSender};
use crate::execution::impact::SquareRootImpact;
use crate::execution::transient_impact::TransientImpactTracker;
use crate::market::gbm::PriceSimulator;
use crate::market::liquidity::LiquiditySnapshot;
use crate::market::lob_generator::{generate_lob, LobGeneratorParams};
use crate::market::lob_replay::LobReplay;
use crate::market::order_book::OrderBook;

// ---------------------------------------------------------------------------
// Fill mode
// ---------------------------------------------------------------------------

/// Selects which fill mechanism the runner uses for every trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    /// Original formula-based fill: exec = permanent-adjusted mid × (1 − transient − base_impact).
    Formula,
    /// Synthetic LOB fill via `generate_lob()` (Phase 2 Track B baseline).
    SyntheticLob,
    /// Real historical L2 replay from Binance depth snapshots.
    HistoricalLob,
}

// ---------------------------------------------------------------------------
// Strategy wrapper
// ---------------------------------------------------------------------------

/// One strategy instance tracked by the runner.
pub struct StrategyInstance {
    pub name: String,
    pub scheduler: Scheduler,
    pub metrics: ExecutionMetrics,
    pub impact_model: SquareRootImpact,
    pub transient_tracker: TransientImpactTracker,
    /// Permanent impact coefficient γ (dimensionless fraction).
    /// Each trade contributes γ·(traded/total_notional) to the cumulative
    /// fractional price depression.  Applied as:
    ///   effective_price = simulator_price * (1 - Σ γ·(q_k / X₀))
    pub gamma: f64,
    pub completed: bool,
}

// ---------------------------------------------------------------------------
// Multi-strategy runner
// ---------------------------------------------------------------------------

/// Runs multiple execution strategies against the same simulated price path
/// and exports a comparative CSV.
pub struct MultiStrategyRunner {
    strategies: Vec<StrategyInstance>,
    simulator: Box<dyn PriceSimulator>,
    max_steps: usize,
    vol_sum: f64,
    vol_count: usize,
    event_tx: Option<EventSender>,
    /// Option B: if true, any strategy with remaining inventory > 0 at the final
    /// step receives a forced liquidation at the last observed price (zero
    /// additional impact). The cost is recorded separately in
    /// `metrics.forced_liquidation_cost` for post-hoc analysis.
    pub force_complete: bool,
    /// Fill mechanism selector. Defaults to `FillMode::Formula`.
    pub fill_mode: FillMode,
    /// Historical LOB replay used when `fill_mode == FillMode::HistoricalLob`.
    /// When the replay is exhausted, falls back to `generate_lob()`.
    pub lob_replay: Option<LobReplay>,
}

/// Per-step snapshot for one strategy (used during CSV export).
struct StepRecord {
    remaining: f64,
    cost: f64,
}

impl MultiStrategyRunner {
    pub fn new(
        strategies: Vec<StrategyInstance>,
        simulator: Box<dyn PriceSimulator>,
        max_steps: usize,
    ) -> Self {
        Self {
            strategies,
            simulator,
            max_steps,
            vol_sum: 0.0,
            vol_count: 0,
            event_tx: None,
            force_complete: false,
            fill_mode: FillMode::Formula,
            lob_replay: None,
        }
    }

    /// Attach a debug event sender (builder pattern).
    pub fn with_event_sender(mut self, tx: EventSender) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// Select fill mode via the legacy bool flag (backward-compatible).
    /// `true` → `FillMode::SyntheticLob`; `false` → `FillMode::Formula`.
    pub fn with_lob(mut self, enabled: bool) -> Self {
        self.fill_mode = if enabled { FillMode::SyntheticLob } else { FillMode::Formula };
        self
    }

    /// Use historical L2 replay as the fill mechanism.
    /// Sets `FillMode::HistoricalLob` and stores the replay.
    pub fn with_historical_lob(mut self, replay: LobReplay) -> Self {
        self.fill_mode = FillMode::HistoricalLob;
        self.lob_replay = Some(replay);
        self
    }

    /// Helper: send an event if the debug channel is active.
    fn emit(&self, event: DebugEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(event);
        }
    }

    /// Drive every strategy through the simulated price path.
    ///
    /// After the loop completes (or all strategies finish early), a CSV
    /// is written to `results/comparison.csv`.
    pub fn run(&mut self) -> Result<(), String> {
        // ── Dynamic AMM reserve process ────────────────────────────────────────
        // Reserves follow a discrete mean-reverting (OU-style) process:
        //   r_new = r + theta * (r_mean - r) + sigma_r * r * noise
        // theta = mean-reversion speed; sigma_r = per-step reserve volatility.
        // Fixed seed for reproducibility — independent of the price simulator seed.
        let reserve_mean = 500.0_f64;
        let reserve_theta = 0.05_f64;  // pulls 5% of the gap back toward mean each step
        let reserve_sigma = 0.02_f64;  // 2% per-step multiplicative noise on each reserve
        let mut rng: StdRng = StdRng::seed_from_u64(1234);
        let mut reserve_in = reserve_mean;
        let mut reserve_out = reserve_mean;

        // Extract event sender so we can emit without borrowing all of `self`.
        let event_tx = self.event_tx.clone();
        let emit = |event: DebugEvent| {
            if let Some(tx) = &event_tx {
                let _ = tx.send(event);
            }
        };

        // Accumulator: one vec per strategy, each vec holds per-step records.
        let n = self.strategies.len();
        let mut history: Vec<Vec<StepRecord>> = (0..n).map(|_| Vec::new()).collect();
        let mut prices: Vec<f64> = Vec::new();

        // Per-strategy permanent price adjustment (γ accumulator).
        // Accumulates as a dimensionless fraction: gamma * (traded / total_notional).
        // Applied as: effective_price = simulator_price * (1 - price_adjustment[i]).
        // Using fraction form keeps γ on the same scale as the transient/base impact
        // fractions, and prevents collapse for large total_notional values.
        let total_notional_ref: f64 = self
            .strategies
            .first()
            .map(|s| s.scheduler.remaining_notional())
            .unwrap_or(1.0)
            .max(1.0);
        let mut price_adjustment: Vec<f64> = vec![0.0; n];
        let fill_mode = self.fill_mode; // Copy — extracted to avoid borrow conflicts below.

        for step in 0..self.max_steps {
            let current_time = step as f64; // T=1 scaling assuming 1.0 total time below. More accurately, `step / max_steps` but `step` is fine for relative decay scaling as long as rho matches.
                                            // Using continuous step counting for the transient tracker timeline.

            // Advance mean-reverting reserve process before each step.
            let z_in: f64 = rng.sample(StandardNormal);
            let z_out: f64 = rng.sample(StandardNormal);
            reserve_in += reserve_theta * (reserve_mean - reserve_in)
                + reserve_sigma * reserve_in * z_in;
            reserve_out += reserve_theta * (reserve_mean - reserve_out)
                + reserve_sigma * reserve_out * z_out;
            // Clamp to prevent degenerate near-zero reserves.
            reserve_in = reserve_in.max(1.0);
            reserve_out = reserve_out.max(1.0);
            let liquidity = LiquiditySnapshot::new(reserve_in, reserve_out);

            let price = self.simulator.step();
            let volatility = self.simulator.volatility();
            self.vol_sum += volatility;
            self.vol_count += 1;
            prices.push(price);

            emit(DebugEvent::PriceUpdate { price, volatility });

            // For HistoricalLob: advance the replay cursor once per time step so
            // all strategies in this step share the same L2 snapshot. The snapshot
            // is cloned per-strategy because fill_market_sell is destructive.
            let step_snapshot: Option<OrderBook> = if fill_mode == FillMode::HistoricalLob {
                self.lob_replay.as_mut().and_then(|r| r.next_snapshot())
            } else {
                None
            };

            for (i, strat) in self.strategies.iter_mut().enumerate() {
                if strat.completed {
                    // Carry forward last-known values.
                    let last = history[i].last().map_or(
                        StepRecord {
                            remaining: strat.scheduler.remaining_notional(),
                            cost: strat.metrics.implementation_shortfall(),
                        },
                        |r| StepRecord {
                            remaining: r.remaining,
                            cost: r.cost,
                        },
                    );
                    history[i].push(last);
                    continue;
                }

                // Effective mid-price seen by this strategy: simulator price multiplied
                // by (1 - accumulated permanent fraction).  The permanent fraction grows
                // with each sell and never decays, depressing all future reference prices.
                let effective_price = price * (1.0 - price_adjustment[i]);

                let remaining_before = strat.scheduler.remaining_notional();
                strat
                    .scheduler
                    .on_block(volatility, &liquidity)
                    .map_err(|e| format!("{}: {e}", strat.name))?;
                let remaining_after = strat.scheduler.remaining_notional();
                let traded = remaining_before - remaining_after;

                // T = 1 year scaling
                let dt = 1.0 / self.max_steps as f64;
                strat
                    .metrics
                    .record_risk_step(remaining_before, volatility, dt);

                if traded > 0.0 {
                    // Log the strategy decision as a SubmitOrder event.
                    emit(DebugEvent::SubmitOrder {
                        order: crate::events::order::Order::market_sell(
                            step as u64,
                            strat.name.clone(),
                            traded,
                        ),
                    });

                    // Transient impact tracker: always maintained regardless of fill path,
                    // so the formula path continues to work exactly as before.
                    let base_impact = strat.impact_model.compute_impact(traded, volatility);
                    let real_time = current_time * dt;
                    let transient_impact = strat.transient_tracker.current_impact(real_time);
                    strat.transient_tracker.record_trade(real_time, base_impact);

                    // Fill mechanism: select based on fill_mode.
                    //
                    // Formula (default):
                    //   adjusted_mid = permanent + transient depression.
                    //   exec_price   = adjusted_mid (no depth walk).
                    //   half_spread  = 0.5 × σ × √dt proxy.
                    //   temporary_fraction = transient_impact.
                    //
                    // SyntheticLob:
                    //   adjusted_mid = permanent-only depression (no transient mid-shift).
                    //   Fill walks generate_lob() bid ladder.
                    //   half_spread  = real LOB half-spread.
                    //   temporary_fraction = slippage_from_mid / adjusted_mid.
                    //
                    // HistoricalLob:
                    //   Same accounting as SyntheticLob but the book comes from
                    //   lob_replay.next_snapshot(). Falls back to generate_lob() if
                    //   the replay is exhausted (cursor past end of file).
                    let adjusted_mid_perm = price * (1.0 - price_adjustment[i]);
                    let (exec_price, half_spread, temporary_fraction) = match fill_mode {
                        FillMode::Formula => {
                            let adjusted_mid =
                                price * (1.0 - price_adjustment[i] - transient_impact);
                            let hs = 0.5 * volatility * dt.sqrt() * price;
                            (adjusted_mid, hs, transient_impact)
                        }
                        FillMode::SyntheticLob => {
                            let lob_params = LobGeneratorParams::default();
                            let mut book = generate_lob(adjusted_mid_perm, volatility, &lob_params);
                            let fill = book.fill_market_sell(traded);
                            let ep = if fill.filled_qty > 1e-12 {
                                fill.avg_fill_price
                            } else {
                                adjusted_mid_perm
                            };
                            let hs = fill.real_spread_cost / traded.max(1e-15);
                            let temp_frac = fill.slippage_from_mid.abs() / adjusted_mid_perm.max(1e-15);
                            (ep, hs, temp_frac)
                        }
                        FillMode::HistoricalLob => {
                            let lob_params = LobGeneratorParams::default();
                            // Clone the step snapshot (captured before the strategy loop);
                            // fall back to synthetic if the replay was exhausted this step.
                            let raw_book = step_snapshot
                                .clone()
                                .unwrap_or_else(|| generate_lob(adjusted_mid_perm, volatility, &lob_params));
                            // Normalize from real-data price scale to simulation scale.
                            // After rescaling, book.mid_price() == adjusted_mid_perm and
                            // all fill prices are directly comparable to effective_price.
                            let scale = adjusted_mid_perm
                                / raw_book.mid_price().unwrap_or(adjusted_mid_perm);
                            let mut book = raw_book.rescale(scale);
                            let fill = book.fill_market_sell(traded);
                            let ep = if fill.filled_qty > 1e-12 {
                                fill.avg_fill_price
                            } else {
                                adjusted_mid_perm
                            };
                            let hs = fill.real_spread_cost / traded.max(1e-15);
                            let temp_frac = fill.slippage_from_mid.abs() / adjusted_mid_perm.max(1e-15);
                            (ep, hs, temp_frac)
                        }
                    };

                    let slippage_cost = exec_price - effective_price;

                    // Permanent impact: each sell contributes gamma × (traded / X0).
                    let delta_permanent = strat.gamma * (traded / total_notional_ref);
                    price_adjustment[i] += delta_permanent;

                    strat.metrics.record_trade(
                        traded,
                        effective_price,
                        slippage_cost,
                        temporary_fraction,
                        delta_permanent,
                        price,
                        half_spread,
                    );

                    emit(DebugEvent::OrderFilled {
                        order_id: step as u64,
                        qty: traded,
                        price: exec_price,
                        impact: (-slippage_cost).max(0.0),
                    });
                }

                if strat.scheduler.remaining_notional() <= 1e-6 {
                    strat.completed = true;
                }

                // Track into history
                history[i].push(StepRecord {
                    remaining: strat.scheduler.remaining_notional(),
                    cost: strat.metrics.implementation_shortfall(),
                });
            }

            // Early exit when every strategy has completed.
            if self.strategies.iter().all(|s| s.completed) {
                break;
            }
        }

        // Option B: forced liquidation of any residual inventory at final price.
        if self.force_complete {
            let last_price = prices.last().copied().unwrap_or(1.0);
            for strat in self.strategies.iter_mut() {
                let residual = strat.scheduler.remaining_notional();
                if residual > 1e-8 {
                    strat.metrics.record_forced_liquidation(residual, last_price);
                    strat.completed = true;
                }
            }
        }

        // Export history to CSV if run successfully.
        self.export_csv(&prices, &history)?;

        Ok(())
    }

    /// Borrow the strategy instances for post-run inspection.
    pub fn strategies(&self) -> &[StrategyInstance] {
        &self.strategies
    }

    /// Average volatility observed across all simulation steps.
    pub fn avg_volatility(&self) -> f64 {
        if self.vol_count == 0 {
            0.0
        } else {
            self.vol_sum / self.vol_count as f64
        }
    }

    /// Borrow strategies mutably — used only in tests to avoid CSV I/O side-effects.
    #[cfg(test)]
    pub fn strategies_mut(&mut self) -> &mut Vec<StrategyInstance> {
        &mut self.strategies
    }

    /// Internal helper to write out path results.
    fn export_csv(&self, prices: &[f64], history: &[Vec<StepRecord>]) -> Result<(), String> {
        let results_dir = "results";
        if std::fs::metadata(results_dir).is_err() {
            std::fs::create_dir(results_dir).map_err(|e| e.to_string())?;
        }
        let path = format!("{results_dir}/comparison.csv");
        let mut file = std::fs::File::create(&path).map_err(|e| e.to_string())?;

        let mut header = String::from("Step,Price");
        for strat in &self.strategies {
            let tag = strat.name.to_lowercase().replace(' ', "_");
            header.push_str(&format!(",{tag}_remaining,{tag}_cost"));
        }
        header.push('\n');
        file.write_all(header.as_bytes())
            .map_err(|e| e.to_string())?;

        for (step, price) in prices.iter().enumerate() {
            let mut line = format!("{},{:.6}", step + 1, price);
            for strat_hist in history {
                if let Some(rec) = strat_hist.get(step) {
                    line.push_str(&format!(",{:.6},{:.6}", rec.remaining, rec.cost));
                } else {
                    line.push_str(",,");
                }
            }
            line.push('\n');
            file.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::metrics::ExecutionMetrics;
    use crate::engine::risk::RiskManager;
    use crate::engine::scheduler::{ExecutionMode, Scheduler};
    use crate::execution::impact::SquareRootImpact;
    use crate::execution::transient_impact::TransientImpactTracker;
    use crate::market::gbm::GbmSimulator;

    // ── Helper ───────────────────────────────────────────────────────────────

    /// Build a minimal two-strategy runner (TWAP + Optimal) on a deterministic
    /// GBM path.  gamma and max_steps are parameterised so tests can vary them.
    fn build_runner(gamma: f64, max_steps: usize) -> MultiStrategyRunner {
        let total_notional = 100_000.0; // large enough for meaningful slices
        let arrival_price = 100.0;
        let sigma_ref = 0.02;    // 2% vol — low enough that schedules complete
        let eta = 0.001;
        let lambda = 1e-4;
        let max_slippage_pct = 0.99; // permissive so strategies always trade
        let reserve_frac = 0.0;
        let base_chunk = total_notional / max_steps as f64; // one slice per step

        let make_instance = |name: &str, mode: ExecutionMode| StrategyInstance {
            name: name.to_string(),
            scheduler: Scheduler::new(
                RiskManager::new(total_notional, max_slippage_pct, reserve_frac),
                total_notional,
                base_chunk,
                sigma_ref,
                500.0, // liquidity_ref
                mode,
                eta,
                lambda,
                Some(max_steps), // Option A: AC horizon matches runner step count
            ),
            metrics: ExecutionMetrics::new(arrival_price, total_notional),
            impact_model: SquareRootImpact::new(5_000_000.0, 0.5),
            transient_tracker: TransientImpactTracker::new(50.0),
            gamma,
            completed: false,
        };

        let strategies = vec![
            make_instance("TWAP", ExecutionMode::Twap),
            make_instance("Optimal", ExecutionMode::Optimal),
        ];

        // Deterministic GBM, seed=42.
        let dt = 1.0 / max_steps as f64;
        let simulator = Box::new(GbmSimulator::new(arrival_price, 0.05, sigma_ref, dt, 42));

        MultiStrategyRunner::new(strategies, simulator, max_steps)
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    /// WHAT: With gamma=0.005, TWAP and Optimal accumulate different delta_IS
    ///       (implementation shortfall) — they must differ by more than 0.01%.
    /// WHY: If all strategies show the same delta_IS, the gamma normalisation is
    ///      wrong (the suspected bug). This test directly confirms or refutes it.
    ///      FLAG: if this test FAILS, it confirms the gamma normalisation bug.
    #[test]
    fn permanent_impact_is_not_uniform_across_strategies() {
        let mut runner = build_runner(
            0.005,  // gamma=0.5% total permanent depression over full sell
            500,    // 500 steps → enough for both strategies to complete
        );
        runner.run().expect("runner should complete without error");

        let total_notional = 100_000.0_f64;
        let strats = runner.strategies();

        // Both strategies must have fully executed.
        for strat in strats {
            let pct = strat.metrics.total_executed() / total_notional * 100.0;
            assert!(
                pct >= 99.99,
                "strategy '{}' pct_executed={:.4}% — must be >=99.99% for a valid IS comparison",
                strat.name, pct,
            );
        }

        let twap_is = strats[0]
            .metrics
            .shortfall_percent()
            .expect("TWAP must have shortfall_percent after run");
        let opt_is = strats[1]
            .metrics
            .shortfall_percent()
            .expect("Optimal must have shortfall_percent after run");

        let delta = (twap_is - opt_is).abs();
        assert!(
            delta > 0.0001, // 0.01% threshold
            "TWAP IS={twap_is:.6}% and Optimal IS={opt_is:.6}% are too close (delta={delta:.6}%); \
             gamma normalisation may be wrong",
        );
    }

    /// WHAT: For a sell order, effective_price decreases (or stays equal) at each step.
    /// WHY: Permanent impact on sells must accumulate monotonically — if price bounces
    ///      back up between steps, the price_adjustment accumulator is being reset or
    ///      subtracted, which would produce incorrect IS calculations.
    #[test]
    fn effective_price_decreases_monotonically_during_sell() {
        // Simulate the permanent impact math directly (mirrors MultiStrategyRunner logic)
        // so we don't need filesystem access or a full runner.
        let gamma = 0.05;           // 5% total permanent impact over full inventory
        let total_notional = 1000.0;
        let raw_price = 100.0;
        let trade_per_step = 100.0; // sell 10% per step for 10 steps

        let mut price_adjustment = 0.0_f64;
        let mut prev_effective = raw_price; // before any trade, full price

        for step in 0..10usize {
            price_adjustment += gamma * (trade_per_step / total_notional);
            let effective = raw_price * (1.0 - price_adjustment);

            assert!(
                effective <= prev_effective + 1e-12,
                "step {step}: effective_price {effective:.6} must not exceed previous {prev_effective:.6}",
            );

            prev_effective = effective;
        }
    }
}

/// Formula-based runner for Table A comparison (LOB delta analysis).
///
/// Replicates the pre-LOB execution model:
///   exec_price = effective_price × (1 − base_impact − transient_impact)
/// Used only in tests/benchmarks to produce Table A numbers.
pub struct FormulaRunner {
    pub strategies: Vec<StrategyInstance>,
    simulator: Box<dyn crate::market::gbm::PriceSimulator>,
    max_steps: usize,
}

impl FormulaRunner {
    pub fn new(
        strategies: Vec<StrategyInstance>,
        simulator: Box<dyn crate::market::gbm::PriceSimulator>,
        max_steps: usize,
    ) -> Self {
        Self { strategies, simulator, max_steps }
    }

    pub fn run(&mut self) -> Result<(), String> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use rand_distr::StandardNormal;

        let reserve_mean = 500.0_f64;
        let reserve_theta = 0.05_f64;
        let reserve_sigma = 0.02_f64;
        let mut rng: StdRng = StdRng::seed_from_u64(1234);
        let mut reserve_in = reserve_mean;
        let mut reserve_out = reserve_mean;

        let n = self.strategies.len();
        let total_notional_ref: f64 = self
            .strategies
            .first()
            .map(|s| s.scheduler.remaining_notional())
            .unwrap_or(1.0)
            .max(1.0);
        let mut price_adjustment: Vec<f64> = vec![0.0; n];

        for step in 0..self.max_steps {
            let current_time = step as f64;
            let z_in: f64 = rng.sample(StandardNormal);
            let z_out: f64 = rng.sample(StandardNormal);
            reserve_in += reserve_theta * (reserve_mean - reserve_in)
                + reserve_sigma * reserve_in * z_in;
            reserve_out += reserve_theta * (reserve_mean - reserve_out)
                + reserve_sigma * reserve_out * z_out;
            reserve_in = reserve_in.max(1.0);
            reserve_out = reserve_out.max(1.0);
            let liquidity = crate::market::liquidity::LiquiditySnapshot::new(reserve_in, reserve_out);

            let price = self.simulator.step();
            let volatility = self.simulator.volatility();
            let dt = 1.0 / self.max_steps as f64;

            for (i, strat) in self.strategies.iter_mut().enumerate() {
                if strat.completed { continue; }
                let effective_price = price * (1.0 - price_adjustment[i]);
                let remaining_before = strat.scheduler.remaining_notional();
                strat.scheduler.on_block(volatility, &liquidity).map_err(|e| format!("{e}"))?;
                let traded = remaining_before - strat.scheduler.remaining_notional();
                strat.metrics.record_risk_step(remaining_before, volatility, dt);

                if traded > 0.0 {
                    let base_impact = strat.impact_model.compute_impact(traded, volatility);
                    let real_time = current_time * dt;
                    let transient_impact = strat.transient_tracker.current_impact(real_time);
                    let total_impact = base_impact + transient_impact;
                    strat.transient_tracker.record_trade(real_time, base_impact);

                    // Formula-based fill: exec = effective_price × (1 − impact)
                    let exec_price = effective_price * (1.0 - total_impact);
                    let slippage_cost = exec_price - effective_price;

                    let delta_permanent = strat.gamma * (traded / total_notional_ref);
                    price_adjustment[i] += delta_permanent;

                    let half_spread_proxy = 0.5 * volatility * dt.sqrt() * effective_price;
                    strat.metrics.record_trade(
                        traded, effective_price, slippage_cost,
                        transient_impact, delta_permanent, price, half_spread_proxy,
                    );
                }
                if strat.scheduler.remaining_notional() <= 1e-6 { strat.completed = true; }
            }
            if self.strategies.iter().all(|s| s.completed) { break; }
        }
        Ok(())
    }
}
