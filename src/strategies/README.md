# Strategies

> Pluggable execution algorithms that control how the total notional is sliced into individual trade chunks.

---

## Strategy Trait

All strategies implement the event-reactive `Strategy` trait (`trait_def.rs`):

```rust
pub trait Strategy {
    fn name(&self) -> &str;
    fn on_event(&mut self, event: &SimEvent, market: &MarketState) -> Vec<Order>;
    fn is_complete(&self) -> bool;
}
```

Strategies **observe** simulation events and **react** by emitting `Order`s. They don't touch the execution engine directly — orders are injected into the event queue by the DES loop.

---

## Available Strategies

### 1. Heuristic (`heuristic.rs`)

**Approach**: Adaptive TWAP — scales a base chunk size by volatility and liquidity ratios.

**Chunk Sizing**:
```
liquidity_factor = clamp(liq / liq_ref, 0.25, 2.0)
volatility_factor = clamp(σ_ref / σ, 0.25, 2.0)
chunk = base_chunk × liquidity_factor × volatility_factor
```

**Behavior**:
- More liquidity → larger chunks (up to 2×)
- Higher volatility → smaller chunks (down to 0.25×)
- Simple, robust, no model assumptions

**Slippage Estimate**: `α × (chunk / liq) + β × σ` with `α=0.01, β=0.005`

**When to use**: As a baseline or when model parameters (η, λ) are uncertain.

---

### 2. Optimal (`optimal.rs`)

**Approach**: Almgren–Chriss closed-form solution — pre-computes the entire trade schedule at the start.

**Model** (`AlmgrenChriss`):

The optimal inventory trajectory minimizes `E[cost] + λ · Var[cost]`:

```
x(k) = X₀ · sinh(κ(T−k)) / sinh(κT)
q(k) = x(k) − x(k+1)                    # per-step trade size
κ = √(λσ²/η)                             # urgency parameter
```

When `λ = 0` or `σ = 0`, the schedule degenerates to linear TWAP.

**Horizon Computation**: If not overridden, `T = ceil(3 / (κ + ε))`, clamped to `[1, 200]`.

**Schedule Properties**:
- `Σ q(k) = X₀` exactly (rounding absorbed by final trade)
- Front-loads execution when risk aversion is high
- Boundary conditions: `x(0) = X₀`, `x(T) = 0`

**When to use**: When you have reliable estimates of η and λ and the market structure is stable.

---

### 3. Adaptive Optimal (`adaptive.rs`)

**Approach**: Receding-horizon re-computation of the A-C solution at each time step using **live** volatility and liquidity.

**Key Difference from Optimal**: Instead of computing one schedule at `t=0`, the adaptive strategy recomputes `q(t)` at every step using the current `σ(t)` and `liq(t)`:

```
η_t = η × clamp(liq_ref / liq_t, 0.25, 4.0)    # liquidity-adjusted impact
κ_t = √(λ × σ_t² / η_t)                          # time-varying urgency
q_t = remaining − remaining × sinh(κ_t(T_rem−1)) / sinh(κ_t × T_rem)
```

**Behavior**:
- Reacts to volatility spikes by accelerating execution
- Adjusts for liquidity changes by scaling the impact coefficient
- Falls back to TWAP when `κ_t → 0`

**When to use**: In GARCH-style markets where volatility and liquidity fluctuate.

---

## Comparison Matrix

| Property | Heuristic | Optimal | Adaptive |
|----------|-----------|---------|----------|
| Planning | None (reactive) | Once at t=0 | Receding horizon |
| Model | Rule-based | Almgren–Chriss | Almgren–Chriss |
| Volatility | Scales chunk size | Fixed at init | Live re-estimation |
| Liquidity | Scales chunk size | Not used post-init | Live adjustment |
| Profile | Roughly uniform | Front-loaded | Context-dependent |
| Parameters | `base_chunk, σ_ref, liq_ref` | `η, λ, σ` | `η, λ, σ_ref, liq_ref` |
| Risk-Aversion | Implicit via vol scaling | Explicit (λ) | Explicit (λ) + adaptive |
| Degenerate Case | — | TWAP when λ=0 | TWAP when κ≈0 |

---

## Integration Points

### In the Scheduler (Live Engine)

The `Scheduler` in `engine/scheduler.rs` dispatches to each strategy via `ExecutionMode`:

```rust
match self.mode {
    ExecutionMode::Heuristic       => self.on_block_heuristic(volatility, liquidity),
    ExecutionMode::Optimal         => self.on_block_optimal(volatility, liquidity),
    ExecutionMode::AdaptiveOptimal => self.on_block_adaptive_optimal(volatility, liquidity),
}
```

All chunks are gated by `RiskManager::request_execute_chunk()`. Denied chunks are retried at half size (heuristic mode only).

### In the DES Engine (Simulation)

The `DesSimulator` in `engine/event_loop.rs` calls `strat.on_event()` for each event, and strategies respond with `Order` vectors:

```
SimEvent::TimeTick → strategy.on_event() → Vec<Order>
     ↓
SimEvent::SubmitOrder → ExecutionEngine → fill/partial
     ↓
SimEvent::OrderFilled → strategy.on_event() → record_fill()
```

### In MultiStrategyRunner (Research)

The `MultiStrategyRunner` runs all three strategies against the same price path. Each strategy has its own `Scheduler`, `ExecutionMetrics`, `SquareRootImpact`, and `TransientImpactTracker` — completely isolated from each other.
