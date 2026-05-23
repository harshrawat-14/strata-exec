<p align="center">
  <strong>StrataExec</strong>
</p>

<p align="center">
  Quantitative execution research system for large-order crypto trading.<br>
  Measures the true cost of executing a large order in a real market — and finds which strategy minimises it.
</p>

<p align="center">
  <code>Rust + Python</code> · <code>389 tests</code> · <code>zero warnings</code> · <code>zero unwrap() in production code</code>
</p>

---

## What This Is

When an institution needs to sell a large position in BTC, doing it all at once collapses the price against them. Splitting it over time reduces market impact but introduces timing risk — prices may move adversely while they wait. The optimal tradeoff is a solved mathematical problem (Almgren-Chriss 2001), but the real-world gap between theory and live market data is not.

StrataExec answers: **how large is that gap, and can a learned policy close it?**

The system has three layers:
- A **Rust simulation engine** implementing five execution strategies with full discrete-event simulation
- A **research pipeline** running Monte Carlo experiments on real Binance order book data (11 dates, 1M+ trades per day)
- A **Python RL agent** (RecurrentPPO with LSTM) trained on a counterfactual LOB environment and evaluated against the closed-form AC optimal

---

## Key Research Findings

All results use BTC-USDT perpetual futures, Binance, 2024. Order size: 50,000 BTC notional (~$2.1B at execution prices).

### Phase 1–2: Strategy Comparison

| Finding | Result |
|---------|--------|
| AC vs TWAP variance reduction | **7.8× lower variance** at σ=40% (500-path GBM sweep) |
| AC in trending market | Pays drift opportunity cost — TWAP outperforms when μ=5% |
| AdaptiveOptimal vs Optimal | AdaptiveOptimal **over-trades** during GARCH volatility clustering (+1.22% IS vs +0.76%) |
| Slice sensitivity | TWAP IS swings 1.07pp across slice counts; AC is slice-invariant |
| VPIN premium | Positive for AC strategies; negative for TWAP — informed flow hurts patient strategies |

### Phase 2B: Calibration Gap (Synthetic vs Real Binance L2)

| Metric | Synthetic Model | Real Binance L2 |
|--------|----------------|----------------|
| Temporary impact | Baseline | **~10× larger** |
| Bid-ask spread | Baseline | **~95× wider** |
| Permanent impact | Strategy-dependent | Strategy-invariant at full liquidation (~0.1–1.7% of IS) |

The synthetic square-root impact model used in the AC derivation significantly understates real market costs. This is the central quantitative finding of Phase 2.

### Phase 3: Null Results (Documented Honestly)

**VPIN as execution signal** — Real aggTrades VPIN ranges 0.05–0.12 with insufficient discrimination to trigger regime-conditional execution. The signal exists but is too weak at daily granularity to improve on static AC.

**Regime-conditional AC** — Greedy receding-horizon regime-switching underperforms the global optimum. The myopic update diverges from the AC solution the longer the horizon.

Both are genuine null results and are reported as such.

### Phase 4: RL Agent Results

The core RL question: can a learned policy outperform the closed-form AC optimal on real market data?

**Final results — RL (LSTM, counterfactual training) vs Static Optimal AC:**

| Date | Regime | AC Optimal | RL Agent | RL wins by |
|------|--------|-----------|---------|-----------|
| 2024-01-15 | calm bull | -0.990% | -0.430% | +0.560pp |
| 2024-03-05 | BTC breakout | -1.228% | -0.459% | +0.769pp |
| 2024-06-10 | quiet consolidation | -0.868% | -0.454% | +0.414pp |
| 2024-08-05 | crash — Yen unwind | -1.071% | -0.482% | +0.589pp |
| 2024-11-06 | post-election surge | -1.212% | -0.478% | +0.734pp |
| **2024-04-15** | BTC correction *(test)* | -1.311% | -0.465% | +0.846pp |
| **2024-07-10** | mid-summer sideways *(test)* | -1.057% | -0.427% | +0.630pp |
| **2024-12-10** | December bull run *(test)* | -1.442% | -0.430% | +1.012pp |

**Bold dates are out-of-sample test dates never used during training.**

Mean sim-to-real degradation: **-0.56pp** (vs synthetic baseline of +0.10%). The RL agent outperforms static AC across all 8 dates including the held-out test set.

---

## Critical Engineering Decisions

The path from first implementation to a working RL agent involved several non-obvious problems. These are documented because the solutions are the technically interesting parts of the project.

### 1. Reward Function Design

**Problem:** The first reward formulation used raw bps slippage without quantity weighting. This made instant liquidation mathematically optimal — selling 245,000 units at 224 bps produced the same reward magnitude as selling 50,000 units at 43 bps, despite the actual dollar cost being 25× larger. The agent learned to dump everything in 7 steps.

**Fix:** Quantity-weighted IS contribution:
```
reward = -(exec_price - mid_price) × fill_qty / (arrival_price × total_inventory)
```
This makes the total episode reward equal to negative IS percentage — bounded, dimensionless, and directly comparable across market conditions. Episode length grew from 7 steps to 1,287+ steps at eval after this fix.

### 2. Book Depth Calibration

**Problem:** Initial `btc_scale=0.0005` gave `qty_factor=2000`, rescaling 2,692 BTC of real depth to 5.3M simulation units. Every action, regardless of size, filled entirely within level 0 at the same price. Fill prices were flat across all action sizes — zero price-impact incentive.

**Fix:** `btc_scale=0.05` (`qty_factor=20`). Level 0 now holds ~53,841 simulation units. Actions up to 5% of inventory fill in level 0; larger actions walk into progressively worse price levels. The 11.4× reward ratio between large and small actions creates a meaningful incentive to spread execution.

### 3. MPS LSTM Crash (PyTorch 2.1)

**Problem:** RecurrentPPO training on Apple MPS crashed consistently at ~500k steps:
```
MPSNDArrayDescriptor sliceDimension error: subRange.start (63)
is not less than length of dimension[2] (1)
```
The Metal LSTM backward kernel allocates a workspace buffer indexed by sequence position. When `episode_length >= lstm_hidden_size` (64), it tries to slice index 63 from a buffer with dimension size 1. `PYTORCH_ENABLE_MPS_FALLBACK=1` did not fix it. Reducing `lstm_hidden_size` to 32 only deferred the crash to `ep_len >= 32`.

**Fix:** `_CPULSTMWrapper` pins the two LSTM submodules (`lstm_actor`, `lstm_critic`) to CPU while all MLP layers stay on MPS. MPS→CPU→MPS transfers are differentiable so gradients flow correctly. The wrapper also proxies all `nn.LSTM` attributes that `sb3-contrib` reads directly (`input_size`, `hidden_size`, `num_layers`, etc.). Throughput: ~49 fps vs ~35 fps CPU-only baseline.

### 4. stdout Pollution in the JSON Wire Protocol

**Problem:** The Rust `rl-env` binary communicates with Python over stdin/stdout as newline-delimited JSON. `AggTradesDay::from_csv` used `println!` for load diagnostics, which wrote to stdout and was read by Python as the reset response, causing `JSONDecodeError` on every episode reset.

**Fix:** Changed all diagnostic output in the rl-env binary from `println!` to `eprintln!` (stderr). This is a subtle class of bug — the protocol and the logging share the same file descriptor, and any stray print breaks the framing.

### 5. Degenerate Policy in Synthetic LOB

**Problem:** Training on an unlimited-depth synthetic LOB, the agent immediately learned to sell everything in one step regardless of reward function. Square-root impact has concavity that makes instant liquidation locally optimal in any LOB with sufficient depth — the agent exploited this in iteration 1 and never escaped.

**Fix:** CounterfactualLob with Obizhaeva-Wang accumulated depth impact. Selling aggressively depletes the visible book across steps, making future fills worse. The agent is forced to consider the sequential consequences of each trade. This was the single most important architectural change — without it, no reward shaping produced meaningful learning.

---

## Overview

StrataExec is a three-mode system:

1. **Live Engine** (`strata-exec`) — Polls on-chain Uniswap V2 reserves in real time, derives mid-price and volatility, then feeds a risk-gated scheduler that decides chunk sizes across three execution strategies.

2. **Research Simulator** (`research-sim`) — Runs Monte Carlo simulations over GBM/GARCH price paths with square-root market impact + transient decay, comparing Heuristic, Optimal, and Adaptive execution. Includes a full parameter sweep framework exporting structured CSV results.

3. **RL Agent** (`rl-env` + Python) — A RecurrentPPO (LSTM) agent trained with `sb3-contrib` against the same Rust simulation engine. The Rust binary exposes a line-delimited JSON interface; `rl/environment.py` wraps it as a Gymnasium env. Supports three training modes: synthetic (GBM/GARCH), historical (real Binance LOB replay), and counterfactual (real LOB adjusted for own-impact via Obizhaeva-Wang).

---

## Table of Contents

- [What This Is](#what-this-is)
- [Key Research Findings](#key-research-findings)
- [Critical Engineering Decisions](#critical-engineering-decisions)
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
| `bin/rl_env` | Long-running RL environment server — JSON-over-stdin/stdout interface for Python |

---

## Getting Started

### Prerequisites

- **Rust** 1.70+ (install via [rustup](https://rustup.rs/))
- An Ethereum JSON-RPC endpoint (only for the live engine; not needed for research mode)

### Build

```bash
git clone https://github.com/harshrawat-14/strata-exec && cd strata-exec
cargo build --release
```

### Quick Smoke Test

```bash
# Run all 389 tests (no RPC / network needed)
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

The `research-sim` binary runs offline Monte Carlo simulations comparing all execution strategies against shared price paths.

### Basic Usage

```bash
# Single path, GBM price model
cargo run --bin research-sim

# 500 Monte Carlo paths
cargo run --bin research-sim -- --paths 500

# GARCH(1,1) price model with real Binance LOB
cargo run --bin research-sim -- --paths 100 --model garch \
  --lob-data TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv \
  --agg-trades TradeData/AggTrades/BTCUSDT-aggTrades-2024-01-15.csv

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
| `--lob-data <file>` | — | Real Binance bookDepth CSV for LOB replay |
| `--agg-trades <file>` | — | Real Binance aggTrades CSV for OFI / VPIN |
| `--experiments` | off | Run parameter sweep framework |
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
./target/release/rl-env   (Rust — src/bin/rl_env.rs)
       │
       ├── synthetic mode  →  GBM / GARCH price path
       ├── historical mode →  LobReplay (real Binance snapshots)
       └── counterfactual  →  CounterfactualLob (real LOB + Obizhaeva-Wang impact)
              ├── AggTradesDay  (OFI per 30-second window)
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

| Mode | `--rl-mode` flag | Price source | State dims |
|------|-----------------|--------------|------------|
| Synthetic | `synthetic` | GBM / GARCH | 8 |
| Historical | `historical` | Binance LOB replay | 8 |
| Counterfactual | `counterfactual` | Real LOB + own-impact correction | 10 |

### State Vector (Counterfactual Mode — 10 dimensions)

| Dim | Signal | Source |
|-----|--------|--------|
| [0] | remaining_inventory / total | rl-env |
| [1] | steps_remaining / total | rl-env |
| [2] | tanh(vol_ratio) | GARCH estimator |
| [3] | tanh(spread_ratio) | Real LOB |
| [4] | tanh(depth_ratio) | Real LOB |
| [5] | OFI — (buy_vol − sell_vol) / total_vol | aggTrades 30s window |
| [6] | tanh(price_drift_10steps) | LOB mid |
| [7] | tanh(last_fill_slippage) | Execution |
| [8] | tanh(permanent_impact_fraction) | OW model |
| [9] | tanh(temporary_impact_fraction) | OW model |

### Prerequisites

```bash
# Build the Rust RL environment binary first
cargo build --release --bin rl-env

# Set up the Python environment
cd rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install sb3-contrib
```

### Train

```bash
# Full training run — 8 dates, 3M steps, linear lr decay (~17 hours on M4)
python rl/train.py \
  --timesteps 3000000 \
  --mode counterfactual \
  --multi-date 2024-01-15,2024-02-20,2024-03-05,2024-06-10,2024-08-05,2024-09-15,2024-10-15,2024-11-06 \
  --btc-target 50000.0 \
  --name ppo_lstm_v4 \
  --lr-schedule linear

# Synthetic mode (no data files needed)
python rl/train.py --mode synthetic --timesteps 500000

# Adversarial mode (FrontRunner + LiquidityWithdrawer active)
python rl/train.py --mode counterfactual --adversarial --timesteps 500000
```

### Evaluate

```bash
python rl/evaluate.py \
  --passive-model rl/models/ppo_lstm_v4_best/best_model \
  --episodes 50 \
  --n-state-dims 10 \
  --include-test-dates
```

### Model Architecture

| Component | Value |
|-----------|-------|
| Algorithm | RecurrentPPO (`sb3-contrib`) |
| Policy | `MlpLstmPolicy` |
| LSTM hidden size | 64 |
| LSTM execution | CPU (`_CPULSTMWrapper` — PyTorch 2.1 MPS fix) |
| LSTM layers | 1 |
| Critic LSTM | enabled |
| Actor/Critic heads | `[64, 64]` each (MPS) |
| Activation | `Tanh` |
| Batch size | 256 |
| Learning rate | 3e-4 → 0 (linear decay) |
| Entropy coef | 0.005 |
| Parallel envs | 8 (SubprocVecEnv) |
| Training device | Apple MPS + CPU LSTM |

### Training Data

| Split | Dates | Purpose |
|-------|-------|---------|
| Training | Jan 15, Feb 20, Mar 05, Jun 10, Aug 05, Sep 15, Oct 15, Nov 06 | Agent training |
| Validation | Jan 15, Mar 05, Jun 10, Aug 05, Nov 06 | Model selection |
| **Test** | **Apr 15, Jul 10, Dec 10** | **Final evaluation only — never seen during training** |

### Market Microstructure Modules

**VPIN** (`market/vpin.rs`) — Easley-López de Prado-O'Hara order-flow toxicity. Classifies trades into buy/sell-initiated via the tick rule, accumulates fixed-size volume buckets, and returns a rolling imbalance signal. Implemented and validated; found to have insufficient discrimination at daily granularity for regime-conditional execution (documented null result, Phase 3).

**Regime** (`market/regime.rs`) — 3×3 vol × liquidity regime classifier. Vol regimes (Low / Normal / High) relative to σ_ref; liquidity regimes (Deep / Normal / Thin) relative to the LOB half-spread at episode start.

**Adversarial** (`market/adversarial.rs`) — Two counter-parties that react to observed flow:
- `LiquidityWithdrawer` — pulls bid depth after a large fill, restores it gradually
- `FrontRunner` — detects sustained sell pressure and front-runs a fraction of our order

**CounterfactualLob** (`market/counterfactual_lob.rs`) — Real Binance snapshots adjusted for own-impact using Obizhaeva-Wang (2013): 30% permanent impact + 70% temporary with exponential decay (ρ = 5, half-life ≈ 200 minutes). This is the training environment that eliminated degenerate instant-liquidation policies.

**RegimeAC** (`strategies/regime_ac.rs`) — Almgren–Chriss strategy with per-regime λ overrides. Urgency escalates from λ = 10⁻⁵ (low vol + deep book) to λ = 5×10⁻² (high vol + thin book). Greedy receding-horizon update diverges from global optimum (documented null result, Phase 3).

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
Produces volatility clustering; used for crash and surge regime simulation.

### Market Impact

**Square-Root Impact** (empirical):
```
Impact = Y · σ · √(Q / V)
```
Where Y = impact coefficient, Q = trade quantity, V = daily volume. Real Binance calibration shows Y ≈ 10× larger than synthetic model assumptions.

**Obizhaeva-Wang Transient Impact**:
```
permanent_reduction += 0.30 × base_impact
temporary_reduction(t) = 0.70 × base_impact × exp(−ρ × steps_elapsed)
ρ = 5,  half-life ≈ 399 steps (200 minutes)
```

### Almgren–Chriss Optimal Execution

Inventory trajectory minimizing `E[cost] + λ · Var[cost]`:
```
x(k) = X₀ · sinh(κ(T−k)) / sinh(κT)
κ = √(λσ² / η)
```
When `κ → 0` (no risk aversion), degenerates to TWAP.

### RL Reward Function

Dimensionless IS contribution per step:
```
reward = -(exec_price - mid_price) × fill_qty / (arrival_price × total_inventory)
       - 0.0002 × (remaining / total)²        [inventory pressure]
       - 0.00005 × vol_ratio² × q_frac²       [variance penalty]
```
Total episode reward ≈ negative IS percentage. Bounded in [-0.05, 0.0] for realistic execution.

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
- **AC Objective** = `E[cost] + λ · Σ(inventory² · σ² · dt)`

---

## Testing

```bash
# Run all 389 tests
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
| `market::counterfactual_lob` | 4 | Reset clears impact, depth depletion, OW decay |
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
├── README.md
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
│   │   ├── agg_trades.rs       # Binance aggTrades CSV reader + OFI series
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
│   └── observability/          # Logging & diagnostics
│       └── logger.rs           # Non-blocking crossbeam-channel logger thread
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
│   └── sweep_impact.csv
│
└── TradeData/                  # Raw Binance data (not committed — source locally)
    ├── BookDepth/              # BTCUSDT-bookDepth-<date>.csv
    └── AggTrades/              # BTCUSDT-aggTrades-<date>.csv
```

---

## License

Private — all rights reserved.