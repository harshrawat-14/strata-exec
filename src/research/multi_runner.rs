use std::fs;
use std::io::Write;

use crate::analytics::metrics::ExecutionMetrics;
use crate::engine::scheduler::Scheduler;
use crate::events::event::{Event as DebugEvent, EventSender};
use crate::execution::impact::SquareRootImpact;
use crate::execution::transient_impact::TransientImpactTracker;
use crate::market::gbm::PriceSimulator;
use crate::market::liquidity::LiquiditySnapshot;

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
        }
    }

    /// Attach a debug event sender (builder pattern).
    pub fn with_event_sender(mut self, tx: EventSender) -> Self {
        self.event_tx = Some(tx);
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
        let liquidity = LiquiditySnapshot::new(500.0, 500.0);

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

        for step in 0..self.max_steps {
            let current_time = step as f64; // T=1 scaling assuming 1.0 total time below. More accurately, `step / max_steps` but `step` is fine for relative decay scaling as long as rho matches.
                                            // Using continuous step counting for the transient tracker timeline.
            let price = self.simulator.step();
            let volatility = self.simulator.volatility();
            self.vol_sum += volatility;
            self.vol_count += 1;
            prices.push(price);

            emit(DebugEvent::PriceUpdate { price, volatility });

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

                    // 1) Compute instantaneous base impact
                    let base_impact = strat.impact_model.compute_impact(traded, volatility);

                    // 2) Query lingering impact from previous trades inside this simulation path
                    let real_time = current_time * dt; // Scale actual time mapping (e.g. 0.0 to 1.0)
                    let transient_impact = strat.transient_tracker.current_impact(real_time);

                    // 3) Total impact combines both effects
                    let total_impact_fraction = base_impact + transient_impact;

                    let exec_price = price * (1.0 - total_impact_fraction);
                    // Slippage in price terms: mid_price - exec_price (positive = adverse).
                    let slippage = price - exec_price;

                    // 4) Record this trade into history so its own base impact can decay forward
                    strat.transient_tracker.record_trade(real_time, base_impact);

                    strat.metrics.record_trade(traded, price, slippage.max(0.0));

                    emit(DebugEvent::OrderFilled {
                        order_id: step as u64,
                        qty: traded,
                        price: exec_price,
                        impact: slippage.max(0.0),
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
