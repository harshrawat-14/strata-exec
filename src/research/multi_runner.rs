use std::fs;
use std::io::Write;

use crate::core::metrics::ExecutionMetrics;
use crate::core::scheduler::Scheduler;
use crate::features::liquidity::LiquiditySnapshot;
use crate::observability::events::{DebugEvent, EventSender};
use crate::research::amm::ConstantProductAMM;
use crate::research::simulator::PriceSimulator;

// ---------------------------------------------------------------------------
// Strategy wrapper
// ---------------------------------------------------------------------------

/// One strategy instance tracked by the runner.
pub struct StrategyInstance {
    pub name: String,
    pub scheduler: Scheduler,
    pub metrics: ExecutionMetrics,
    pub amm: ConstantProductAMM,
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
            let price = self.simulator.step();
            let volatility = self.simulator.volatility();
            self.vol_sum += volatility;
            self.vol_count += 1;
            prices.push(price);

            emit(DebugEvent::PriceUpdate { step, price, volatility });

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
                strat.metrics.record_risk_step(remaining_before, volatility, dt);

                if traded > 0.0 {
                    emit(DebugEvent::StrategyDecision {
                        strategy: strat.name.clone(),
                        order_size: traded,
                        remaining: remaining_after,
                    });

                    // Execute through constant-product AMM.
                    let mid_price = strat.amm.price();
                    let exec_price = strat.amm.sell_x(traded);
                    // Slippage = difference between execution and mid price.
                    // exec_price is dy/dx (Y received per X sold).
                    // mid_price is reserve_y/reserve_x.
                    // Slippage in price terms: mid_price - exec_price (positive = adverse).
                    let slippage = mid_price - exec_price;
                    strat
                        .metrics
                        .record_trade(traded, price, slippage.max(0.0));

                    emit(DebugEvent::TradeExecuted {
                        strategy: strat.name.clone(),
                        price: exec_price,
                        quantity: traded,
                        impact: slippage.max(0.0),
                    });
                }

                if strat.scheduler.is_completed() {
                    strat.completed = true;
                }

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

        // ── Export CSV ──────────────────────────────────────────────────
        self.export_csv(&prices, &history)
    }

    /// Write `results/comparison.csv`.
    fn export_csv(
        &self,
        prices: &[f64],
        history: &[Vec<StepRecord>],
    ) -> Result<(), String> {
        fs::create_dir_all("results").map_err(|e| format!("create results dir: {e}"))?;

        let mut file = fs::File::create("results/comparison.csv")
            .map_err(|e| format!("create CSV: {e}"))?;

        // Header — dynamic based on strategy names.
        let mut header = String::from("block,price");
        for strat in &self.strategies {
            let tag = strat.name.to_lowercase().replace(' ', "_");
            header.push_str(&format!(",{tag}_remaining,{tag}_cost"));
        }
        writeln!(file, "{header}").map_err(|e| format!("write header: {e}"))?;

        // Rows.
        for (step, price) in prices.iter().enumerate() {
            let mut row = format!("{},{price:.6}", step + 1);
            for strat_history in history {
                if let Some(rec) = strat_history.get(step) {
                    row.push_str(&format!(",{:.6},{:.6}", rec.remaining, rec.cost));
                } else {
                    row.push_str(",,");
                }
            }
            writeln!(file, "{row}").map_err(|e| format!("write row: {e}"))?;
        }

        Ok(())
    }

    /// Borrow the strategy instances for post-run inspection.
    pub fn strategies(&self) -> &[StrategyInstance] {
        &self.strategies
    }

    /// Average volatility observed across all simulation steps.
    pub fn avg_volatility(&self) -> f64 {
        if self.vol_count == 0 {
            return 0.0;
        }
        self.vol_sum / self.vol_count as f64
    }
}
