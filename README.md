<p align="center">
  <strong>StrataExec</strong>
</p>

<p align="center">
  Institutional-grade optimal execution engine with discrete event simulation, on-chain liquidity monitoring, Monte Carlo research tooling, and a recurrent RL agent — built in Rust + Python.
</p>

<p align="center">
  <code>163 tests</code> · <code>zero warnings</code> · <code>zero unwrap() in production code</code>
</p>

---

## Overview

StrataExec is a three-mode system:

1. **Live Engine** (`strata-exec`) — Polls on-chain Uniswap V2 reserves in real time, derives mid-price and volatility, then feeds a risk-gated scheduler that decides chunk sizes across three execution strategies.

2. **Research Simulator** (`research-sim`) — Runs Monte Carlo simulations over GBM/GARCH price paths with square-root market impact + transient decay, comparing Heuristic, Optimal, and Adaptive execution. Includes a full parameter sweep framework exporting structured CSV results.

3. **RL Agent** (`rl-env` + Python) — A RecurrentPPO (LSTM) agent trained with `sb3-contrib` against the same Rust simulation engine. The Rust binary exposes a line-delimited JSON interface; `rl/environment.py` wraps it as a Gymnasium env. Supports three training modes: synthetic (GBM/GARCH), historical (real Binance LOB replay), and counterfactual (real LOB adjusted for own-impact via Obizhaeva-Wang).

---

## Table of Contents

- [Architecture](#architecture)
- [Module Reference](#module-reference)
- [Getting Started](#getting-started)
- [Live Engine](#live-engine)
- [Research Simulator](#research-simulator)
- [Experiment Framework](#experiment-framework)
- [Calibration](#calibration)
- [RL Agent](#rl-agent)
- [HTTP API](#http-api)
- [Configuration](#configuration)
- [Mathematical Models](#mathematical-models)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         StrataExec                                  │
│                                                                     │
│  ┌──────────────────┐       ┌──────────────────────────────────┐   │
│  │   Live Engine     │       │     Research Simulator            │   │
│  │   (strata-exec)   │       │     (research-sim)                │   │
│  │                    │       │                                    │   │
│  │  Blockchain RPC ──┤       │  GBM / GARCH Price Paths ─────┐  │   │
│  │  VolEstimator   ──┤       │  SquareRootImpact            ──┤  │   │
│  │  Scheduler      ──┤       │  TransientImpactTracker      ──┤  │   │
│  │  RiskManager    ──┤       │  MultiStrategyRunner         ──┤  │   │
│  │  HTTP API       ──┤       │  DistributionStats           ──┤  │   │
│  └──────────────────┘       │  Experiment Sweep Framework   ──┤  │   │
│                              │  Calibration Engine           ──┤  │   │
│                              └──────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Shared Core   (events · strategies · execution · market) │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow — Live Engine

```
On-Chain Block  →  Fetch Reserves  →  LiquiditySnapshot + Mid-Price
                                            │
                                   VolatilityEstimator (rolling window)
                                            │
                                    Scheduler.on_block(σ, liquidity)
                                            │
                              ┌─────────────┴─────────────┐
                              │         RiskManager        │
                              │  (4-constraint gate)       │
                              └─────────────┬─────────────┘
                                            │
                               Chunk booked  →  ExecutionMetrics
                                            │
                                   AppState (HTTP-visible)
```

### Data Flow — Research Simulator

```
Config (seed, σ, η, λ)  →  GBM/GARCH Simulator
                                     │
                           ┌─────────┴─────────┐
                           │  MultiStrategyRunner │
                           │  ┌─── Heuristic     │
                           │  ├─── Optimal (A-C) │
                           │  └─── AdaptiveOpt   │
                           └─────────┬─────────┘
                                     │
                       Impact Model + Transient Decay
                                     │
                          ExecutionMetrics per strategy
                                     │
                        DistributionStats (MC aggregation)
                                     │
                             CSV / Terminal Output
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `engine/` | Block-polling orchestrator, scheduler, risk manager, state machine |
| `strategies/` | Heuristic (TWAP-like), Almgren–Chriss Optimal, Adaptive Optimal, Regime-aware A-C |
| `market/` | GBM + GARCH simulators, LOB replay, counterfactual LOB, adversarial agents, VPIN, regime classifier, agg-trades reader |
| `execution/` | DES execution engine, square-root impact model, transient impact decay |
| `events/` | `SimEvent` enum, priority queue, order types |
| `analytics/` | `ExecutionMetrics` (VWAP, shortfall, variance, AC objective), `DistributionStats` (VaR, CVaR, skewness, kurtosis) |
| `research/` | `MultiStrategyRunner`, parameter sweep experiment framework |
| `calibration/` | Empirical impact coefficient (Y) fitting from historical trade CSV |
| `api/` | Axum HTTP control plane (`/health`, `/status`, `/metrics`, `/performance`, `/start`, `/stop`) |
| `blockchain/` | `BlockchainClient` trait + `RpcClient` with exponential backoff retry |
| `config/` | Environment-variable based configuration with validation |
| `observability/` | Non-blocking `crossbeam-channel` logging thread |
| `ml/` | _(Placeholder)_ Future ML-based residual predictions |
| `bin/rl_env` | Long-running RL environment server — JSON-over-stdin/stdout interface for Python |

---

## Getting Started

### Prerequisites

- **Rust** 1.70+ (install via [rustup](https://rustup.rs/))
- An Ethereum JSON-RPC endpoint (only for the live engine; not needed for research mode)

### Build

```bash
git clone <repo-url> && cd StrataExec
cargo build --release
```

### Quick Smoke Test

```bash
# Run all 163 tests (no RPC / network needed)
cargo test

# Run a quick research simulation (no .env needed)
cargo run --bin research-sim -- --paths 10
```

---

## Live Engine

The live engine monitors on-chain Uniswap V2 pair reserves and executes notional through a risk-gated scheduler.

### Setup

```bash
cp .env.example .env
# Edit .env — set RPC_URL, PAIR_ADDRESS, and execution parameters
```

### Run

```bash
cargo run                                       # Heuristic mode (default)
EXECUTION_MODE=optimal ETA=0.1 LAMBDA=0.01 cargo run    # Optimal mode
EXECUTION_MODE=adaptive_optimal ETA=0.1 LAMBDA=0.01 cargo run
```

The engine:
1. Fetches ERC-20 decimal places on startup for correct price normalization
2. Polls blocks every 2 seconds
3. Builds `LiquiditySnapshot` from on-chain reserves
4. Derives decimal-normalized mid-price (`(reserve1/10^dec1) / (reserve0/10^dec0)`)
5. Feeds block-over-block returns into a rolling `VolatilityEstimator`
6. Delegates to the `Scheduler` which dispatches to the selected strategy
7. All chunks pass through the `RiskManager` (4-constraint gate) before booking
8. Updates `AppState` atomically for HTTP visibility

### Execution Modes

| Mode | Strategy | Description |
|------|----------|-------------|
| `heuristic` | `HeuristicStrategy` | TWAP-like adaptive chunking — scales base chunk by volatility and liquidity ratios |
| `optimal` | `AlmgrenChriss` | Pre-computed sinh-based schedule minimizing `E[cost] + λ · risk` |
| `adaptive_optimal` | `AdaptiveOptimalStrategy` | Receding-horizon re-computation of the A-C solution at each step using live σ and liquidity |

---

## Research Simulator

The `research-sim` binary runs offline Monte Carlo simulations comparing all three execution strategies against shared price paths.

### Basic Usage

```bash
# Single path, GBM price model
cargo run --bin research-sim

# 500 Monte Carlo paths
cargo run --bin research-sim -- --paths 500

# GARCH(1,1) price model
cargo run --bin research-sim -- --paths 100 --model garch

# With progress reporting
cargo run --bin research-sim -- --paths 500 --progress

# Debug events for first 2 paths only
cargo run --bin research-sim -- --paths 100 --debug --debug-paths 2
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--paths N` | `1` | Number of Monte Carlo paths |
| `--model gbm\|garch` | `gbm` | Price dynamics model |
| `--experiments` | off | Run parameter sweep framework (see [below](#experiment-framework)) |
| `--calibrate <file>` | — | Calibrate impact coefficient Y from historical CSV |
| `--debug` | off | Enable event logging |
| `--debug-paths N` | all | Limit event logging to first N paths |
| `--progress` | off | Print path completion progress |

### Output

- **Terminal**: Per-strategy summary (shortfall, VWAP, variance, AC objective)
- **CSV**: `results/comparison.csv` — step-by-step price and per-strategy remaining/cost trajectory

---

## Experiment Framework

The `research/experiments` module implements a grid-search framework that sweeps across four parameter dimensions, running Monte Carlo simulations at each grid point and exporting structured CSV results.

### Run

```bash
cargo run --bin research-sim -- --experiments --paths 100
```

### Parameter Grids

| Dimension | Grid Values | Output File |
|-----------|-------------|-------------|
| Execution Horizon (days) | `[0.25, 0.5, 1.0, 2.0]` | `results/sweep_horizon.csv` |
| Trade Slices | `[25, 50, 100, 200]` | `results/sweep_trade_chunks.csv` |
| Volatility (σ) | `[0.10, 0.20, 0.30, 0.40]` | `results/sweep_volatility.csv` |
| Impact Coefficient (Y) | `[0.10, 0.50, 1.00, 2.00]` | `results/sweep_impact.csv` |

### Recorded Metrics (per strategy, per grid point)

| Metric | CSV Column | Formula |
|--------|-----------|---------|
| Mean Implementation Shortfall | `MeanImplementationShortfall_Pct` | `Σ(exec_price - arrival_price) × qty / notional` |
| Shortfall Variance | `ImplementationShortfallVariance_Pct2` | `σ²` of shortfall across MC paths |
| CVaR 95% | `CVaR95ImplementationShortfall_Pct` | Mean of tail losses ≥ 95th percentile |
| Mean AC Objective | `MeanACObjective_Pct` | `shortfall% + λ · risk_penalty%` |
| AC Objective Variance | `ACObjectiveVariance_Pct2` | `σ²` of AC objective across MC paths |
| CVaR 95% (AC) | `CVaR95ACObjective_Pct` | Tail risk of AC objective |

### CSV Format

```csv
ParameterName,ParameterValue,Strategy,NumPaths,MeanImplementationShortfall_Pct,...
ExecutionHorizonDays,0.2500,Heuristic,100,0.001234,...
ExecutionHorizonDays,0.2500,Optimal,100,0.000987,...
ExecutionHorizonDays,0.2500,AdaptiveOptimal,100,0.001012,...
```

### Design Principles

- **Does not modify the core simulator** — the experiment framework is self-contained in `research/experiments.rs`
- Uses the same `MultiStrategyRunner`, `GbmSimulator`, `SquareRootImpact`, `TransientImpactTracker`, and `ExecutionMetrics` as the main simulator
- Deterministic seeding (`base_seed = 42 + path_id`) for reproducibility

---

## Calibration

Calibrate the square-root impact coefficient (Y) from historical trade data:

```bash
cargo run --bin research-sim -- --calibrate historical_trades.csv --paths 100
```

### CSV Format

```csv
timestamp_sec,price_before,price_after,trade_size
1.0,100.0,100.05,10000.0
2.0,100.05,100.12,20000.0
```

The engine solves for Y algebraically:
```
Y = (|price_after - price_before| / price_before) / (σ × √(Q / V))
```
and averages across all valid observations. The calibrated Y then overrides the default for all subsequent simulation paths.

---

## RL Agent

The RL layer trains a RecurrentPPO (LSTM) agent to learn execution timing against the same Rust engine used for research simulations.

### Architecture

```
Python (sb3-contrib RecurrentPPO)
       │  line-delimited JSON via stdin/stdout
       ▼
./target/debug/rl-env   (Rust — src/bin/rl_env.rs)
       │
       ├── synthetic mode  →  GBM / GARCH price path
       ├── historical mode →  LobReplay (real Binance snapshots)
       └── counterfactual  →  CounterfactualLob (real LOB + Obizhaeva-Wang impact)
              ├── AggTradesDay  (VPIN / order-flow toxicity)
              ├── MarketRegime  (vol × liq 3×3 classification)
              └── AdversarialBook (FrontRunner + LiquidityWithdrawer)
```

### Protocol

The `rl-env` binary communicates over stdin/stdout with newline-delimited JSON:

```
→ reset:  {"reset": true, "seed": 42}
→ step:   {"action": 7}          # integer 0–19 (fraction of remaining inventory)

← response: {"state": [...], "reward": -0.0023, "done": false, "info": {...}}
```

### Training Modes

| Mode | `--rl-mode` flag | Price source | Observations |
|------|-----------------|--------------|--------------|
| Synthetic | `synthetic` | GBM / GARCH | 8-dim state |
| Historical | `historical` | Binance LOB replay | 8-dim state |
| Counterfactual | `counterfactual` | Real LOB + own-impact correction | 10-dim state (+ VPIN + regime) |

### Prerequisites

```bash
# Build the Rust RL environment binary first
cargo build --bin rl-env

# Set up the Python environment
cd rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install sb3-contrib
```

### Train

```bash
# Counterfactual mode (recommended) — requires Binance BookDepth + AggTrades CSVs
python rl/train.py \
  --mode counterfactual \
  --lob-date 2024-01-15 \
  --agg-date 2024-01-15 \
  --btc-target 50000 \
  --timesteps 500000 \
  --lr-schedule linear \
  --name ppo_lstm_counterfactual

# Synthetic mode (no data files needed)
python rl/train.py --mode synthetic --timesteps 500000

# Adversarial mode (FrontRunner + LiquidityWithdrawer active)
python rl/train.py --mode counterfactual --adversarial --timesteps 500000
```

### Evaluate

```bash
python rl/evaluate.py --model rl/models/ppo_lstm_counterfactual_final \
  --mode counterfactual \
  --lob-date 2024-01-15 \
  --agg-date 2024-01-15 \
  --episodes 50
```

### Model Architecture

| Component | Value |
|-----------|-------|
| Algorithm | RecurrentPPO (`sb3-contrib`) |
| Policy | `MlpLstmPolicy` |
| LSTM hidden size | 64 |
| LSTM layers | 1 |
| Critic LSTM | enabled |
| Actor/Critic heads | `[64, 64]` each |
| Activation | `Tanh` |
| Clip range | 0.2 |
| Entropy coef | 0.005 |

### Market Microstructure Modules

**VPIN** (`market/vpin.rs`) — Easley-López de Prado-O'Hara order-flow toxicity. Classifies trades into buy/sell-initiated via the tick rule, accumulates fixed-size volume buckets, and returns a rolling imbalance signal used as an RL observation.

**Regime** (`market/regime.rs`) — 3×3 vol × liquidity regime classifier. Vol regimes (Low / Normal / High) are relative to σ_ref; liquidity regimes (Deep / Normal / Thin) are relative to the LOB half-spread at episode start.

**Adversarial** (`market/adversarial.rs`) — Two counter-parties that react to observed flow:
- `LiquidityWithdrawer` — pulls bid depth after a large fill, restores it gradually
- `FrontRunner` — detects sustained sell pressure and front-runs a fraction of our order

**CounterfactualLob** (`market/counterfactual_lob.rs`) — Real Binance snapshots adjusted for own-impact using Obizhaeva-Wang (2013): 30% permanent impact + 70% temporary with exponential decay (ρ = 5, half-life ≈ 3.3 h).

**RegimeAC** (`strategies/regime_ac.rs`) — Almgren–Chriss strategy with per-regime λ overrides. Urgency escalates from λ = 10⁻⁵ (low vol + deep book) to λ = 5×10⁻² (high vol + thin book simultaneously).

---

## HTTP API

The live engine exposes an Axum HTTP control plane:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Health check → `{"status": "ok"}` |
| `/status` | `GET` | Engine state, remaining notional, volatility |
| `/metrics` | `GET` | Blocks processed, chunks executed, risk denials |
| `/performance` | `GET` | VWAP, shortfall, variance, AC objective (per execution mode) |
| `/start` | `POST` | Resume execution |
| `/stop` | `POST` | Pause execution (engine keeps polling, skips scheduling) |

### Examples

```bash
curl http://localhost:3000/health
# {"status":"ok"}

curl http://localhost:3000/status
# {"is_running":true,"remaining_notional":8500.0,"last_volatility":0.0213}

curl http://localhost:3000/performance
# {"mode":"optimal","total_executed":1500.0,"vwap":100.23,"implementation_shortfall":3.45,...}

curl -X POST http://localhost:3000/stop
# {"is_running":false}
```

---

## Configuration

All configuration is loaded from environment variables (`.env` file supported via `dotenvy`).

### Required Variables

| Variable | Type | Constraint | Description |
|----------|------|------------|-------------|
| `RPC_URL` | String | — | Ethereum JSON-RPC endpoint |
| `PAIR_ADDRESS` | Address | Checksummed | Uniswap V2 pair contract address |
| `TOTAL_NOTIONAL` | f64 | > 0 | Total notional to execute |
| `BASE_CHUNK` | f64 | > 0 | Base chunk size for heuristic mode |
| `MAX_SLIPPAGE_PCT` | f64 | (0, 1) | Maximum allowed slippage fraction |
| `EMERGENCY_RESERVE_FRAC` | f64 | [0, 1) | Fraction of slippage budget held as emergency reserve |
| `VOL_WINDOW` | usize | ≥ 2 | Rolling window size for volatility estimation |
| `SIGMA_REF` | f64 | > 0 | Reference volatility for chunk scaling |
| `LIQUIDITY_REF` | f64 | > 0 | Reference liquidity depth for chunk scaling |

### Optional Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EXECUTION_MODE` | String | `heuristic` | `heuristic`, `optimal`, or `adaptive_optimal` |
| `ETA` | f64 | `0.0` | Temporary impact coefficient (required > 0 for optimal modes) |
| `LAMBDA` | f64 | `0.0` | Risk aversion parameter (≥ 0) |
| `HORIZON_BLOCKS` | usize | auto | Override horizon for optimal schedule; auto-computed if omitted |
| `LOG_LEVEL` | String | `info` | Tracing log level |
| `HTTP_PORT` | u16 | `3000` | HTTP server bind port |

---

## Mathematical Models

### Price Dynamics

**GBM** — Geometric Brownian Motion:
```
S(t+dt) = S(t) · exp((μ − ½σ²)dt + σ√dt · Z),    Z ~ N(0,1)
```

**GARCH(1,1)** — Time-varying volatility:
```
r(t) = (μ − ½σ²(t))·dt + √(σ²(t)·dt) · Z
σ²(t+1) = ω + α·r²(t) + β·σ²(t)
```
Produces volatility clustering.

### Market Impact

**Square-Root Impact** (empirical):
```
Impact = Y · σ · √(Q / V)
```
Where Y = impact coefficient, Q = trade quantity, V = daily volume.

**Transient Impact Decay**:
```
I(t) = Σᵢ base_impact_i · exp(−ρ · (t − tᵢ))
```
Automatic memory pruning when `exp(−ρ · dt) < 10⁻⁸`.

### Almgren–Chriss Optimal Execution

Inventory trajectory minimizing `E[cost] + λ · Var[cost]`:
```
x(k) = X₀ · sinh(κ(T−k)) / sinh(κT)
κ = √(λσ² / η)
```
When `κ → 0` (no risk aversion), degenerates to TWAP.

### Risk Manager (4-Constraint Gate)

Every chunk must satisfy **all four** constraints:
1. **Global cap** — cumulative slippage ≤ `total_notional × max_slippage_pct`
2. **Running average** — average slippage rate ≤ `max_slippage_pct`
3. **Proportional allocation** — per-chunk fair share of total budget
4. **Emergency reserve** — fraction held back for overruns

### Analytics

- **Implementation Shortfall** = `Σ qty × (exec_price − arrival_price)`
- **Variance** = population variance of per-trade realized costs
- **VaR α** = α-th percentile of the cost distribution
- **CVaR α** = mean of all values ≥ VaR α (expected tail loss)
- **AC Objective** = `E[cost] + λ · Σ(inventory² · σ² · dt)` — the discrete Almgren–Chriss objective

---

## Testing

```bash
# Run all 163 tests
cargo test

# Run only experiment tests
cargo test research::experiments

# Run only analytics tests
cargo test analytics::

# Run with output visible
cargo test -- --nocapture
```

### Test Coverage by Module

| Module | Tests | Focus |
|--------|-------|-------|
| `analytics::distribution` | 7 | Percentile interpolation, CVaR, skewness, kurtosis |
| `analytics::metrics` | 10 | VWAP, shortfall, variance, AC objective, risk penalty |
| `engine::engine` | 3 | Block polling, scheduler integration, decimal normalization |
| `engine::event_loop` | 2 | DES loop correctness, strategy→execution routing |
| `engine::risk` | 5 | Global cap, running average, proportional allocation, emergency draw |
| `events::queue` | 2 | Chronological ordering, FIFO tie-breaking |
| `execution::engine` | 3 | Order fills, partial fills, cancellation |
| `execution::impact` | 2 | Square-root scaling, zero-trade edge case |
| `execution::transient_impact` | 3 | Exponential decay, accumulation, history pruning |
| `market::gbm` | 2 | Deterministic output, price positivity |
| `market::garch` | 4 | Volatility clustering, decay, positivity, determinism |
| `market::drift` | 5 | Edge cases, numerical correctness |
| `market::liquidity` | 4 | Reserve validation, mid-price, depth metric |
| `market::volatility` | 3 | Rolling window, eviction, edge cases |
| `strategies::optimal` | 7 | Horizon computation, schedule sum, front-loading, invalid inputs |
| `research::experiments` | 8 | Config validation, determinism, sensitivity, CSV format |
| `calibration` | 3 | Y-coefficient algebra, zero-trade filtering, multi-trade averaging |
| `blockchain::client` | 3 | Mock client correctness |

---

## Project Structure

```
StrataExec/
├── Cargo.toml
├── .env.example
├── historical_trades.csv       # Sample calibration data
├── README.md                   # ← You are here
│
├── src/
│   ├── main.rs                 # Live engine entry point (tokio async)
│   ├── lib.rs                  # Library crate root
│   │
│   ├── bin/
│   │   ├── research_sim.rs     # Research simulator entry point
│   │   └── rl_env.rs           # RL environment server (JSON over stdin/stdout)
│   │
│   ├── engine/                 # Core orchestration
│   │   ├── engine.rs           # Block-polling loop + Scheduler integration
│   │   ├── event_loop.rs       # Discrete Event Simulation loop
│   │   ├── scheduler.rs        # ExecutionMode dispatch (Heur/Opt/Adaptive)
│   │   ├── risk.rs             # 4-constraint slippage risk manager
│   │   └── state.rs            # Engine lifecycle enum
│   │
│   ├── strategies/             # Execution algorithms
│   │   ├── trait_def.rs        # Strategy trait (event-reactive interface)
│   │   ├── heuristic.rs        # Adaptive TWAP with vol/liq scaling
│   │   ├── optimal.rs          # Almgren–Chriss closed-form + DES wrapper
│   │   ├── adaptive.rs         # Receding-horizon adaptive A-C
│   │   └── regime_ac.rs        # Regime-aware A-C (per-regime λ overrides)
│   │
│   ├── market/                 # Price models & market data
│   │   ├── gbm.rs              # Geometric Brownian Motion + PriceSimulator trait
│   │   ├── garch.rs            # GARCH(1,1) stochastic volatility
│   │   ├── volatility.rs       # Rolling-window volatility estimator
│   │   ├── liquidity.rs        # LiquiditySnapshot (depth, mid-price)
│   │   ├── drift.rs            # Drift estimator
│   │   ├── state.rs            # MarketState (DES-observable)
│   │   ├── lob_replay.rs       # Real Binance LOB snapshot replay
│   │   ├── order_book.rs       # In-memory limit order book
│   │   ├── agg_trades.rs       # Binance aggTrades CSV reader
│   │   ├── counterfactual_lob.rs # LOB + Obizhaeva-Wang own-impact correction
│   │   ├── adversarial.rs      # FrontRunner + LiquidityWithdrawer agents
│   │   ├── vpin.rs             # VPIN order-flow toxicity (Easley et al. 2012)
│   │   └── regime.rs           # Vol × liquidity 3×3 regime classifier
│   │
│   ├── execution/              # Order execution mechanics
│   │   ├── engine.rs           # DES execution engine (submit/fill/cancel)
│   │   ├── impact.rs           # Square-root market impact model
│   │   ├── transient_impact.rs # Exponential decay of past trade impact
│   │   ├── fills.rs            # Fill record structures
│   │   └── mod.rs
│   │
│   ├── events/                 # Event system
│   │   ├── event.rs            # SimEvent enum (14 variants)
│   │   ├── queue.rs            # Min-heap priority queue with FIFO tie-breaking
│   │   └── order.rs            # Order struct (Buy/Sell, limit/market)
│   │
│   ├── analytics/              # Metrics & statistics
│   │   ├── metrics.rs          # ExecutionMetrics (VWAP, shortfall, AC objective)
│   │   └── distribution.rs     # DistributionStats (VaR, CVaR, moments)
│   │
│   ├── research/               # Offline research tooling
│   │   ├── multi_runner.rs     # Multi-strategy shared-path runner
│   │   └── experiments.rs      # Parameter sweep framework + tests
│   │
│   ├── calibration/            # Empirical model calibration
│   │   └── impact_fit.rs       # Y-coefficient fitting from historical CSV
│   │
│   ├── api/                    # HTTP control plane
│   │   └── mod.rs              # Axum routes + AppState
│   │
│   ├── blockchain/             # On-chain interaction
│   │   ├── client.rs           # BlockchainClient trait + RpcClient + retry
│   │   ├── nonce.rs            # Nonce management
│   │   └── tx_builder.rs       # Transaction construction
│   │
│   ├── config/                 # Configuration
│   │   └── mod.rs              # Env-var loading with validation
│   │
│   ├── observability/          # Logging & diagnostics
│   │   └── logger.rs           # Non-blocking crossbeam-channel logger thread
│   │
│   ├── ml/                     # ML (placeholder)
│   │   └── residual.rs
│   │
│   ├── types/                  # Shared type definitions
│   ├── utils/                  # Logging init
│   └── error.rs                # Error types
│
├── rl/                         # Python RL layer
│   ├── requirements.txt        # Python dependencies (pinned)
│   ├── environment.py          # Gymnasium wrapper around rl-env binary
│   ├── train.py                # RecurrentPPO (LSTM) training script
│   └── evaluate.py             # Policy evaluation + CSV export
│
├── results/                    # Rust research output (CSV)
│   ├── comparison.csv
│   ├── sweep_horizon.csv
│   ├── sweep_trade_chunks.csv
│   ├── sweep_volatility.csv
│   ├── sweep_impact.csv
│   ├── garch_sweep_*.csv
│   └── gbm_sweep_*.csv
│
└── TradeData/                  # Raw Binance data (not committed — source locally)
    ├── BookDepth/              # BTCUSDT-bookDepth-<date>.csv
    └── AggTrades/              # BTCUSDT-aggTrades-<date>.csv
```

---

## License

Private — all rights reserved.
