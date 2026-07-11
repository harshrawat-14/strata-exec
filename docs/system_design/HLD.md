# StrataExec — High-Level Design

## System Overview

StrataExec is a quantitative execution research platform that answers one
question empirically: can a reinforcement-learning agent outperform the
closed-form Almgren-Chriss optimal execution solution on real Binance
BTC-USDT perpetual futures data, across diverse market regimes, including
held-out dates never seen during training? The system trains and evaluates
five execution strategies — TWAP, a volatility/liquidity-adjusted Heuristic,
Almgren-Chriss Optimal, a receding-horizon Adaptive AC, and a RecurrentPPO
(LSTM) RL agent — inside a custom counterfactual limit-order-book
environment with an Obizhaeva-Wang market-impact model, then serves the
results through a FastAPI backend and a React frontend, with a pre-computed
demo mode for zero-compute production deployment.

## Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLIENT LAYER                                                             │
│  React + TypeScript + Tailwind + Recharts (frontend/)                    │
│  Pages: Simulator, RLEvaluation, DataUpload, ParameterSweep              │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ HTTPS / REST / WebSocket / SSE
┌───────────────────────────────▼──────────────────────────────────────────┐
│  SERVICE LAYER                                                            │
│  FastAPI (web/main.py)                                                    │
│  Middleware: DemoModeMiddleware → RequestIDMiddleware → CORS             │
│  Routes: auth, dashboard, simulate, evaluate, sweep, upload, websocket,  │
│          progress, demo                                                  │
│  ARQ worker pool (Redis-backed) for background job execution             │
└───────────┬───────────────────────────────────────────────┬──────────────┘
            │ DEMO_MODE=true: served from disk               │ DEMO_MODE=false
            │ (no compute)                                    │ (live compute)
┌───────────▼────────────────┐                  ┌────────────▼──────────────┐
│  DATA LAYER (frozen)        │                  │  COMPUTE LAYER            │
│  demo_results/manifest.json │                  │  Rust binaries (release): │
│  demo_results/simulation/*  │                  │   strata-exec (live engine)│
│  demo_results/evaluation/*  │                  │   research-sim (Monte Carlo)│
│  Read-only JSON, no compute │                  │   rl-env (RL env, JSON IPC)│
└──────────────────────────────┘                  └────────────┬──────────────┘
                                                                │
                                                   ┌────────────▼──────────────┐
                                                   │  MARKET MODELS LAYER       │
                                                   │  CounterfactualLob (OW impact)│
                                                   │  GARCH / GBM / Jump-diffusion │
                                                   │  Adversarial agents (FrontRunner,│
                                                   │  LiquidityWithdrawer)      │
                                                   │  Regime classifier, OFI    │
                                                   └────────────┬──────────────┘
                                                                │
                                                   ┌────────────▼──────────────┐
                                                   │  RL / PYTHON LAYER         │
                                                   │  rl/environment.py (Gymnasium│
                                                   │  wrapper over rl-env subprocess)│
                                                   │  rl/train.py (RecurrentPPO, │
                                                   │  MLflow tracking)          │
                                                   │  rl/evaluate.py (classical +│
                                                   │  RL policy evaluation)     │
                                                   └────────────┬──────────────┘
                                                                │
                                                   ┌────────────▼──────────────┐
                                                   │  DATA LAYER (source)        │
                                                   │  TradeData/BookDepth/*.csv  │
                                                   │  TradeData/AggTrades/*.csv  │
                                                   │  (real Binance BTC-USDT L2) │
                                                   └────────────────────────────┘
```

## Key Design Decisions

1. **Rust + Python process boundary via JSON over stdin/stdout, not PyO3.**
   The Rust simulation core (`rl-env`) and the Python RL stack
   (stable-baselines3 / sb3-contrib) are separate processes communicating
   over newline-delimited JSON. A native binding (PyO3/maturin) would save
   IPC overhead but couples the Rust build toolchain to the Python
   environment's ABI and complicates independent versioning, testing, and
   deployment of the two halves. At the state-vector sizes and step
   frequencies involved (2,880 steps/episode, 12-16 floats/step), JSON IPC
   overhead is negligible relative to model inference time. See
   [ADR-001](ADR.md#adr-001-json-over-stdinstdout-for-ipc).

2. **Counterfactual LOB instead of historical replay.**
   Pure historical replay has no causal feedback — the agent's own fills
   don't affect future snapshots, so an agent can dump its entire position
   with zero simulated consequence. The counterfactual LOB combines real
   Binance L2 snapshots with an accumulated Obizhaeva-Wang impact model, so
   a fill at step *t* measurably depletes the book the agent sees at step
   *t+1*. This is what makes patient execution mechanically more profitable
   than instant liquidation. See
   [ADR-002](ADR.md#adr-002-counterfactual-lob-over-historical-replay).

3. **Pre-computed demo mode instead of live compute on Render's free tier.**
   Training and even single evaluation runs take seconds to hours depending
   on episode count; a free-tier web dyno cannot host live Rust subprocess
   spawning under real traffic without risking timeouts, cold-start
   failures, or resource exhaustion. `DemoModeMiddleware` intercepts
   `/api/simulate` and `/api/evaluate` and serves pre-computed JSON from
   `demo_results/`, so the public demo is instant and free to run, while
   `DEMO_MODE=false` still exercises the full live pipeline locally. See
   [ADR-003](ADR.md#adr-003-pre-computed-demo-mode).

4. **RecurrentPPO (LSTM) instead of DQN or SAC.**
   Optimal execution is fundamentally sequential — the correct action at
   step *t* depends on the agent's own trading history within the episode
   (how much book depth it has already consumed, how the remaining time
   budget has evolved), which a memoryless policy cannot represent. DQN and
   SAC both assume the Markov property holds on the *observed* state; a
   feedforward policy on this state vector cannot infer "I sold 200K units
   in the last 10 steps, the book is thinning, I should slow down." An LSTM
   hidden state carries that history across steps within an episode. See
   [ADR-004](ADR.md#adr-004-recurrentppo-over-dqnsac).

## Data Flow — Simulation Request (`research-sim` / classical strategies)

```
Client → POST /api/simulate {price_model, sigma, lambda, notional, horizon}
  → DemoModeMiddleware (DEMO_MODE=true: match preset key, return demo_results/simulation/*.json)
  → [live path, DEMO_MODE=false] web/routes/simulate.py → ARQ job
    → subprocess: ./target/release/research-sim --model garch --paths 500 ...
    → Monte Carlo sweep over TWAP / Heuristic / Optimal / AdaptiveOptimal
    → results written to results/*.csv, job marked complete
  → Client polls /api/jobs/{id}/state or subscribes to /api/jobs/{id}/progress (SSE)
  → Client fetches /api/compare/{job_id} for final strategy comparison
```

## Data Flow — Evaluation Request (RL agent / unified comparison)

```
Client → POST /api/evaluate {dates: [...]}
  → DemoModeMiddleware (DEMO_MODE=true: return demo_results/evaluation/{date}.json)
  → [live path, DEMO_MODE=false] web/routes/evaluate.py → ARQ job
    → rl/evaluate.py: load_model() → evaluate_model_on_date() per date
      → rl/environment.py spawns ./target/release/rl-env --rl-mode historical ...
      → 50 episodes/date, seeds 0-49, JSON reset/step exchange per episode
      → IS%, CVaR-95, action distribution, forced-liquidation rate computed
  → Client fetches /api/evaluate/result/{job_id}
```

## Phase S — Systems Rework: Caching, Allocations, Batched Transport

Measured, not asserted (see [ADR-006](ADR.md#adr-006-arrow-ipc-over-a-unix-domain-socket-for-the-batched-hot-loop-transport-plus-per-date-data-caching),
[phase_s_baseline.md](phase_s_baseline.md), [phase_s_regression_report.md](phase_s_regression_report.md)).
Scoped to this offline research platform, not live-trading infra (see Phase S's explicit
scope boundary in the project plan) — no live-clock abstraction, no lock-free live
dispatch, no distributed training added here.

**What the baseline profiling actually found** (before assuming the transport was the
bottleneck): `RlEnv::reset()` — not `step()` — dominated episode wall time. It re-parses
the entire historical CSV (1M+-row aggTrades, 28.8K-row bookDepth) from disk on *every*
episode, ~239ms and ~99% of a historical episode's allocations, while the 2880-step
`step()` loop that follows it is comparatively free at the wall-clock level (16
allocations/step). The per-step JSON transport ADR-001 called "negligible" turned out to
still be true — the redundant per-episode I/O was the real cost, and was never measured
before.

**Changes made, in the order their measured impact motivated them:**

1. **Per-date `MarketDataCache`** — parse each date's CSV once per process, `Arc`-share
   the immutable snapshot data across every subsequent `reset()` (and across a batched
   env's N instances). This is the change that mattered most.
2. **Hot-loop allocation reduction** in `build_state()` (pre-sized `Vec`, no
   grow-reallocations), `write_msg()` (`serde_json::to_writer` straight to the stream,
   no intermediate `String`), and the input loop (`read_line` into a reused buffer
   instead of `BufRead::lines()`'s per-line allocation).
3. **Graceful degradation**: a missing/corrupt historical file or a LOB replay that
   yields zero snapshots now force-terminates the episode (`done=true`, a defined
   `DATA_INGESTION_FAILURE_PENALTY` reward, an explanatory `info.error`) instead of
   panicking and killing the whole training process.
4. **`BatchedRlEnv` + Arrow RecordBatch transport** (ADR-006) — N `RlEnv` instances per
   process, sharing one `MarketDataCache`, driven over a Unix domain socket
   (`--batch-size N --socket-path PATH`) using length-prefixed Arrow IPC messages
   (`rl/batched_environment.py` is the Python client). Additive: the default
   single-instance JSON-over-stdio path (`environment.py` / `rl/evaluate.py`'s every
   existing call site) is unchanged and still the one every current comparison table
   uses.

**Measured throughput** (2024-01-15, `TWAPPolicy`, fixed-horizon episodes; steps/sec is
env `step()` calls/sec, not wall-clock-independent):

| Configuration | steps/sec | episodes/sec |
|---|---:|---:|
| Before Phase S (JSON-over-stdio, no cache) | 12,083 | 4.20 |
| After caching + allocation fixes (JSON-over-stdio, single instance) | ~44,000–46,000 | ~15.3–16.1 |
| Batched (Arrow/socket), batch_size=1 | 1,186 | — |
| Batched (Arrow/socket), batch_size=8, aggregate | 18,740 | — |
| Batched (Arrow/socket), batch_size=32, aggregate | 63,972 | — |

**Honest reading of the batched numbers, not the headline-friendly one:** the caching
fix alone (row 2) already delivers most of the win and is *faster per-instance* than the
batched path at small N. The Arrow/socket protocol has real fixed per-call overhead
(Python-side serialization + one socket round-trip per `step_batch()` call) that is
*not* amortized until N is large enough — at `batch_size=1` the batched path is ~40x
*slower* than the equivalent single-instance JSON call once caching is warm. Aggregate
throughput does scale close to linearly with N (1,186 → 18,740 → 63,972 at N=1/8/32),
which is the actual value proposition: fewer OS processes, one array handoff instead of
N separate ones, not a smaller per-step constant. This is reported plainly rather than
picking the most flattering N — reducing that fixed per-call overhead (e.g. reusing
Arrow buffers instead of allocating a fresh `BufferOutputStream` per call) is a clear,
identified next step, not claimed as already solved.

**Regression-verified against Phase 0**: `evaluate_all_strategies_on_date` run against
the pre-Phase-S binary (`a7fe081`) and the post-Phase-S binary produced byte-for-byte
identical `mean_IS`/`cvar_95`/`std_IS`/`is_sharpe` for all 5 strategies — confirming
none of the above changed simulation semantics (full detail in
[phase_s_regression_report.md](phase_s_regression_report.md)).

## Scalability Notes

- **Horizontal (API layer):** `web/main.py` is stateless per-request aside
  from the ARQ pool handle in `app.state`; multiple FastAPI pods behind a
  load balancer is a straightforward scale-out, provided they share the
  same Redis instance (job queue) and Postgres/SQLite (uploaded models,
  users).
- **Worker tier:** ARQ workers (`web/arq_worker.py`) consume the same Redis
  queue and can scale independently of the API pods — CPU/GPU-bound
  simulation and evaluation jobs run there, not in the request thread.
- **Compute isolation:** each `StrataExecEnv` spawns its own `rl-env`
  subprocess; there is no shared mutable state between concurrent
  evaluations, so evaluation jobs parallelize trivially across worker
  processes.
- **Demo mode sidesteps scaling entirely** for the public-facing deployment
  — no compute, no worker tier, no Redis dependency required — which is why
  it exists as a first-class deployment mode rather than a test fixture.
- **Batched training envs** (Phase S): a training loop that needs many
  parallel environment instances should use `BatchedRlEnv`/
  `rl/batched_environment.py` rather than N separate `StrataExecEnv`
  subprocesses — one `rl-env` process holds N instances sharing one
  `MarketDataCache`, avoiding N-way redundant CSV parsing.

## Related Documents

- [LLD.md](LLD.md) — struct layouts, wire protocol, state vector, action space
- [ADR.md](ADR.md) — the 6 architecture decisions with alternatives and consequences
- [phase_s_baseline.md](phase_s_baseline.md) — Phase S's before-measurement
- [phase_s_regression_report.md](phase_s_regression_report.md) — Phase S's regression-verify against Phase 0
