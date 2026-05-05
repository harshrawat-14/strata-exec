# Research Module

> Offline execution research tooling — Monte Carlo simulation, multi-strategy comparison, and parameter sweep experiments.

This module is designed to be used via the `research-sim` binary and does **not** modify any core simulator components.

---

## Components

### `multi_runner.rs` — MultiStrategyRunner

Runs multiple execution strategies against **the same** simulated price path, enabling fair comparison.

```
GBM/GARCH Simulator
        │
        ▼
┌───────────────────────────┐
│    MultiStrategyRunner     │
│                            │
│  Step 1: sim.step() → price
│  Step 2: For each strategy:
│     ├─ scheduler.on_block(σ, liq)
│     ├─ impact_model.compute_impact(traded, σ)
│     ├─ transient_tracker.current_impact(t)
│     ├─ total = base + transient
│     ├─ exec_price = price × (1 - total)
│     ├─ transient_tracker.record_trade(t, base)
│     └─ metrics.record_trade(traded, price, slippage)
│  Step 3: Check if all strategies completed
│  Step 4: Export results/comparison.csv
└───────────────────────────┘
```

#### Key Design Decisions

- **Shared price path**: All strategies see identical prices — differences come purely from execution timing
- **Transient impact isolation**: Each strategy has its own `TransientImpactTracker` — one strategy's trades don't affect another's
- **Event channel**: Optional `EventSender` can be attached for debug logging (non-blocking `crossbeam-channel`)

### `experiments.rs` — Parameter Sweep Framework

Orchestrates grid-search Monte Carlo experiments across four parameter dimensions.

#### Architecture

```
run_all_sweeps(num_paths, event_tx)
    │
    ├─ run_horizon_sweep()      → results/sweep_horizon.csv
    │   └─ for horizon in [0.25, 0.5, 1.0, 2.0]:
    │       └─ evaluate_config(params, num_paths, ...)
    │           └─ for path in 0..num_paths:
    │               ├─ GbmSimulator(seed=42+path)
    │               ├─ build_sweep_strategies(params)
    │               ├─ MultiStrategyRunner.run()
    │               └─ collect shortfall%, ac_objective%
    │
    ├─ run_trade_count_sweep()  → results/sweep_trade_chunks.csv
    │   └─ for slices in [25, 50, 100, 200]:
    │       └─ evaluate_config(...)
    │
    ├─ run_volatility_sweep()   → results/sweep_volatility.csv
    │   └─ for σ in [0.10, 0.20, 0.30, 0.40]:
    │       └─ evaluate_config(...)
    │
    └─ run_impact_sweep()       → results/sweep_impact.csv
        └─ for Y in [0.10, 0.50, 1.00, 2.00]:
            └─ evaluate_config(...)
```

#### `SweepConfig`

Configuration for a single experiment point:

| Field | Default | Description |
|-------|---------|-------------|
| `total_notional` | 1,000,000 | Total notional to execute |
| `base_chunk` | 10,000 | Base chunk (→ ~100 slices) |
| `sigma_ref` | 0.02 | Reference σ for strategies |
| `liquidity_ref` | 500.0 | Reference liquidity depth |
| `eta` | 0.001 | Temporary impact coefficient |
| `lambda` | 1e-4 | Risk aversion |
| `max_slippage_pct` | 0.5 | Risk manager cap |
| `reserve_frac` | 0.0 | Emergency reserve |
| `arrival_price` | 100.0 | Benchmark price |
| `daily_volume` | 5,000,000 | For square-root impact |
| `impact_coefficient` | 0.5 | Y in the impact model |
| `transient_rho` | 50.0 | Impact decay rate |
| `horizon_days` | 1.0 | Execution window |

#### Horizon Scaling

When `horizon_days` varies, the simulator adjusts both the number of steps and the time-step size:

```rust
let steps_per_day = 500;
let max_steps = (horizon_days * 500.0).round() as usize;
let dt = horizon_days / max_steps as f64;
```

This correctly scales the Brownian Motion variance (`σ²·dt`) relative to the operational window.

#### CSV Output Schema

Each sweep CSV contains one row per (parameter_value, strategy) pair:

```
ParameterName,ParameterValue,Strategy,NumPaths,
MeanImplementationShortfall_Pct,ImplementationShortfallVariance_Pct2,
CVaR95ImplementationShortfall_Pct,MeanACObjective_Pct,
ACObjectiveVariance_Pct2,CVaR95ACObjective_Pct
```

#### Determinism

- Base seed = 42
- Path `i` uses seed `42 + i`
- Same config → identical results across runs (verified by unit test)

---

## Usage Examples

### Quick comparison (single path)

```bash
cargo run --bin research-sim
```

### Monte Carlo with GARCH

```bash
cargo run --bin research-sim -- --paths 1000 --model garch
```

### Full experiment sweep

```bash
cargo run --bin research-sim -- --experiments --paths 200
```

Produces:
```
results/
├── sweep_horizon.csv       (4 horizons × 3 strategies = 12 rows)
├── sweep_trade_chunks.csv  (4 counts × 3 strategies = 12 rows)
├── sweep_volatility.csv    (4 vols × 3 strategies = 12 rows)
└── sweep_impact.csv        (4 coefficients × 3 strategies = 12 rows)
```

### Calibrate from real data then simulate

```bash
cargo run --bin research-sim -- --calibrate historical_trades.csv --paths 500
```

### Debug specific paths

```bash
# Log events for first 3 paths only (out of 500)
cargo run --bin research-sim -- --paths 500 --debug --debug-paths 3
```

---

## Test Coverage

8 tests in `experiments.rs`:

| Test | What it validates |
|------|-------------------|
| `sweep_config_defaults_are_valid` | All defaults positive, ~100 slices |
| `sweep_grids_are_non_empty` | Grid values strictly positive |
| `evaluate_config_returns_all_strategies` | One result per strategy, finite metrics |
| `evaluate_config_deterministic_across_runs` | Same seed → identical results |
| `evaluate_config_varies_with_horizon` | Different horizons → different outputs |
| `evaluate_config_varies_with_volatility` | Higher σ → higher shortfall std_dev |
| `build_sweep_strategies_creates_all_modes` | 3 strategies, correct names |
| `csv_header_contains_all_metric_columns` | All 10 columns present in header |

Run:
```bash
cargo test research::experiments -- --nocapture
```

---

## Extending the Framework

### Adding a new sweep dimension

1. Add a grid constant:
   ```rust
   const MY_GRID: [f64; 4] = [0.1, 0.2, 0.5, 1.0];
   ```

2. Add a field to `SweepConfig` and its `Default` impl

3. Write `run_my_sweep()` following the pattern of existing sweeps:
   ```rust
   fn run_my_sweep(num_paths, total_paths, progress_done, event_tx) -> Result<(), String> {
       let path = "results/sweep_my_param.csv";
       let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
       write_sweep_header(&mut file, "MyParam")?;

       for value in MY_GRID {
           let mut params = SweepConfig::default();
           params.my_field = value;
           let stats = evaluate_config(&params, num_paths, *progress_done, total_paths, event_tx)?;
           *progress_done += num_paths;
           write_sweep_rows(&mut file, "MyParam", format!("{value:.4}"), num_paths, &stats)?;
       }
       Ok(())
   }
   ```

4. Call it from `run_all_sweeps()`

### Adding a new metric

1. Implement the metric in `ExecutionMetrics` (or derive from existing fields)
2. Collect it in the `evaluate_config()` loop alongside shortfall and AC objective
3. Add to `StrategySweepStats`
4. Extend the CSV header and row format
