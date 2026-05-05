# Engine

> Core orchestration layer — block polling, scheduling, risk management, and discrete event simulation.

---

## Overview

The engine module contains two complementary execution models:

1. **Live Block Loop** (`engine.rs`) — Polls on-chain blocks via RPC, derives market state, and feeds the scheduler.
2. **DES Simulator** (`event_loop.rs`) — Priority-queue based discrete event simulation for offline research.

Both share the same `Scheduler`, `RiskManager`, and state primitives.

---

## Components

### `engine.rs` — Live Engine

The `Engine` struct is the central orchestrator for production execution.

**Lifecycle**: `Idle → Running → (runs indefinitely until ctrl-c)`

**Per-Block Pipeline**:

```
┌──────────────────────────────────────────────────────────────┐
│  1. Fetch reserves       client.get_uniswap_v2_reserves()   │
│  2. Liquidity snapshot   LiquiditySnapshot::new(r0, r1)     │
│  3. Decimal normalize    (r1/10^dec1) / (r0/10^dec0)        │
│  4. Vol estimation       vol_estimator.update(return)        │
│  5. Scheduler dispatch   scheduler.on_block(σ, liquidity)   │
│  6. Risk gate            RiskManager 4-constraint check      │
│  7. Metrics update       metrics.record_trade(chunk, ...)   │
│  8. AppState publish     atomics + RwLock for HTTP layer     │
└──────────────────────────────────────────────────────────────┘
```

**Decimal Normalization**: The engine fetches ERC-20 `decimals()` on startup to correctly normalize reserve ratios:
```
price = (reserve1 / 10^decimals_token1) / (reserve0 / 10^decimals_token0)
```
This is critical for pairs like USDC/WETH (6 vs 18 decimals).

---

### `event_loop.rs` — DES Simulator

The `DesSimulator` processes events from a min-heap priority queue in chronological order.

**Event Routing**:
```
Event Queue → pop()
    │
    ├─ MarketState.handle()     → price/vol/liquidity updates
    ├─ ExecutionEngine.handle() → order fills, partial fills
    └─ Strategy.on_event()      → new orders (scheduled at t+ε)
```

**Termination**: The loop exits when `SimEvent::SimulationEnd` is encountered or the queue empties.

**Causality**: Follow-on events (fills, strategy orders) are scheduled slightly after the current time (`t + 1e-6` for orders, `t + 1e-9` for fills) to maintain proper causal ordering.

---

### `scheduler.rs` — Scheduler

The scheduler is a pure state machine that dispatches to the selected execution strategy.

**Execution Modes**:

| Mode | Strategy | Schedule Type |
|------|----------|---------------|
| `Heuristic` | `HeuristicStrategy` | On-demand adaptive |
| `Optimal` | `AlmgrenChriss` | Pre-computed once |
| `AdaptiveOptimal` | `AdaptiveOptimalStrategy` | Receding horizon |

**Risk Integration**: Every proposed chunk passes through `RiskManager::request_execute_chunk()`. In heuristic mode, a denied chunk is retried at 50% size. In optimal/adaptive modes, denied chunks are simply dropped.

**Completion**: The scheduler tracks `remaining_notional` and sets `completed = true` when inventory ≤ 10⁻⁸.

---

### `risk.rs` — RiskManager

Multi-constraint slippage risk manager enforcing **four simultaneous invariants**:

| # | Constraint | Formula |
|---|-----------|---------|
| 1 | Global cap | `used_slippage ≤ total_notional × max_slippage_pct` |
| 2 | Running average | `avg_slippage_rate ≤ max_slippage_pct` |
| 3 | Proportional allocation | `per_chunk ≤ (chunk / total) × budget` |
| 4 | Emergency reserve | `proportional + remaining_emergency` ceiling |

**The tightest constraint wins**: `allowed = min(proportional_ceiling, running_cap, remaining_budget)`

**Emergency reserve**: A fraction of the total budget is held back. Chunks can overshoot their proportional share by drawing from this reserve, but the reserve is finite and tracked.

---

### `state.rs` — EngineState

Simple lifecycle enum:
```rust
pub enum EngineState {
    Idle,                 // Before run()
    Running,              // Active block polling
    AwaitingConfirmation, // (Reserved for future tx confirmation)
    Aborted,              // (Reserved for fatal errors)
}
```

---

## Configuration Parameters

| Parameter | Used By | Effect |
|-----------|---------|--------|
| `total_notional` | Scheduler, Risk | Total amount to execute |
| `base_chunk` | Heuristic | Base size before vol/liq adjustment |
| `sigma_ref` | Heuristic, Adaptive | Reference volatility for scaling |
| `liquidity_ref` | Heuristic, Adaptive | Reference liquidity for scaling |
| `eta` | Optimal, Adaptive | Temporary impact coefficient |
| `lambda` | Optimal, Adaptive | Risk aversion (Almgren–Chriss) |
| `max_slippage_pct` | Risk | Hard cap on cumulative slippage |
| `reserve_frac` | Risk | Emergency reserve fraction |
| `poll_interval` | Engine | Block polling cadence (default 2s) |
