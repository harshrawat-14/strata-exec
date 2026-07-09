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

## Related Documents

- [LLD.md](LLD.md) — struct layouts, wire protocol, state vector, action space
- [ADR.md](ADR.md) — the 5 architecture decisions with alternatives and consequences
