# StrataExec — Low-Level Design

## CounterfactualLob

`src/market/counterfactual_lob.rs`. Wraps a real Binance L2 replay and
tracks the agent's own accumulated market impact so that trades measurably
affect the book the agent sees on the next step.

```rust
pub struct CounterfactualLobConfig {
    pub permanent_fraction: f64,   // 0.3 — OW-style permanent impact share
    pub temporary_fraction: f64,   // 0.7 — decays exponentially
    pub rho: f64,                  // 5.0 — decay rate; half-life ~400 steps at dt=1/2880
    pub dt: f64,                   // 1.0 / total_steps
    pub btc_scale: f64,            // simulation-units -> BTC (e.g. 0.05)
    pub adv_btc: f64,              // average daily volume in BTC (240_000.0 default)
    pub impact_coefficient: f64,   // Y in the square-root magnitude formula (0.1 default, uncalibrated)
    pub sigma: f64,                // volatility term in the same formula (1.0 default = no-op)
}
```
Naming note (Phase 2R): the permanent/temporary decay structure IS genuine
Obizhaeva-Wang-style dynamics (each trade's temporary contribution tracked
with its own timestamp and decayed individually — verified by reading
`total_residual_impact()`). The instantaneous MAGNITUDE, however, is a
square-root law (`impact_coefficient * sigma * sqrt(participation)`), not
OW's linear form — a separate, well-supported empirical model. Calling the
whole thing "Obizhaeva-Wang" mislabels the magnitude half.
`CounterfactualLobConfig::calibrated(..)` fits `impact_coefficient`/`sigma`
to real trade data (via `calibration::impact_fit::CalibrationEngine`,
bridged from Binance aggTrades + LOB snapshot mids) instead of using the
hardcoded defaults.
```rust

pub struct CounterfactualLob {
    replay: LobReplay,
    config: CounterfactualLobConfig,
    permanent_depth_reduction: f64,          // accumulated, never decays, capped at 0.95
    temporary_impacts: Vec<(f64, f64)>,      // (time_when_traded, impact_fraction)
    current_step: usize,
    pub depth_ref: f64,                      // first snapshot's total bid depth (calibration ref)
    current_book: Option<OrderBook>,         // current real Binance snapshot, pre-impact
    simulation_mid_price: f64,                // fixed at 100.0; real prices rescaled to this
}
```

Key methods: `current_adjusted_book()` (real book minus accumulated impact),
`record_trade_and_advance(qty)` (records a fill's impact, advances the
replay cursor), `permanent_impact()` / `temporary_impact_total()` (exposed
to the state vector as dims [8]/[9]), `total_residual_impact()`.

## Fill Execution Algorithm (counterfactual mode, per step)

```
1. adjusted_book = cf_lob.current_adjusted_book()
     (real Binance snapshot at current replay position, minus
      permanent_depth_reduction and un-decayed temporary_impacts)
2. book_mid = adjusted_book.mid_price()
3. intended_qty = ACTION_FRACTIONS[action] * remaining_inventory
4. if intended_qty > 0:
     fill_result = adjusted_book.fill_market_sell(intended_qty)
       (walks the bid ladder level by level until intended_qty is filled
        or depth is exhausted; returns filled_qty, avg_fill_price)
5. cf_lob.record_trade_and_advance(intended_qty)
     -> permanent_depth_reduction += permanent_fraction * (qty / adv_btc-scaled)
     -> temporary_impacts.push((current_time, temporary_fraction * ...))
     -> advance replay cursor to the next real snapshot
6. remaining_inventory -= filled_qty
   total_notional_received += filled_qty * avg_fill_price
7. reward = (fill_price - book_mid) * filled_qty / (arrival_price * total_inventory)
            [if no fill this step: small negative variance/inventory-pressure penalty]
8. done = (remaining_inventory <= 0.1% of total_inventory)
       or (step_idx >= total_steps, with forced liquidation of any residual)
       or (cf_lob.is_exhausted(), i.e. replay ran out of snapshots)
9. next_state = build_state()  (see State Vector table below)
```

## JSON Wire Protocol

Newline-delimited JSON over the `rl-env` subprocess's stdin/stdout. One
message in, one message out, per call.

**Reset request:**
```json
{"reset": true, "seed": 42}
```

**Reset response:**
```json
{
  "state": [0.0, 1.0, 0.0, 0.12, 0.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
  "reward": 0.0,
  "done": false,
  "info": {
    "step": 0,
    "remaining": 1000000.0,
    "arrival_price": 100.0,
    "action_9_fraction": 0.05,
    "episode_horizon": 2481,
    "total_inventory": 1000000.0,
    "size_fraction": 1.0,
    "n_state_dims": 12,
    "vol_regime": 0.5,
    "liq_regime": 0.5,
    "time_features_active": false,
    "btc_scale": 0.05,
    "adv_btc": 240000.0,
    "depth_ref": 53841.2
  }
}
```

**Step request:**
```json
{"action": 9}
```

**Step response:**
```json
{
  "state": [0.95, 0.999, 0.01, 0.10, 0.30, 0.02, -0.01, -0.05, 0.001, 0.002, 0.5, 0.5],
  "reward": -0.0000234,
  "done": false,
  "info": {
    "step": 1,
    "remaining": 950000.0,
    "fill_qty": 50000.0,
    "fill_price": 99.98,
    "is_contribution": -0.0002,
    "regime": "quiet"
  }
}
```

On the final step of an episode, `info` additionally carries
`total_IS_pct`, `total_is_dollar`, and (only if inventory remained above the
0.1% threshold at the time horizon) `forced_liquidation_qty`.

If `reset()`'s data-ingestion (LOB/aggTrades file load) fails — missing file, or a
replay that parses to zero valid snapshots — the response is instead
`{"reward": -1.0, "done": true, "state": [0.0, ...] (correct width), "info":
{"error": "<message>", "terminated_reason": "data_ingestion_failure"}}` rather than the
process panicking (Phase S graceful degradation; see `DATA_INGESTION_FAILURE_PENALTY`
in `rl_env.rs`).

## Batched Wire Protocol (Phase S, ADR-006)

Additive to the JSON protocol above, used only when `rl-env` is invoked with
`--batch-size N --socket-path PATH` (`BatchedRlEnv` / `rl/batched_environment.py`).
Speaks Arrow RecordBatch IPC over a Unix domain socket instead of JSON-over-stdio, one
persistent connection, N environment instances stepped together per request.

**Framing**: every message (both directions) is a 4-byte little-endian length prefix
followed by exactly that many bytes of a self-contained Arrow IPC *stream* buffer
(schema + one `RecordBatch` + EOS marker), parsed from an in-memory buffer rather than
read directly off the live socket. This is required, not a stylistic choice — an earlier
version wrote Arrow IPC bytes straight to the socket and relied on Arrow's own EOS
marker to delimit messages; both `arrow-rs`'s and `pyarrow`'s stream readers may read
ahead past that marker when the underlying source is a live, continuing byte stream
(observed concretely: `pyarrow.ipc.open_stream` corrupted the *second* request on a
shared connection). Framing each message to an exact, bounded byte range removes any
possibility of over-reading into the next message.

**Reset request** — one `RecordBatch`, schema `{seed: uint64}`, N rows (one seed per
instance).

**Step request** — one `RecordBatch`, schema `{action: int32}`, N rows.

**Response** (both reset and step) — one `RecordBatch`, schema
`{s0: float32, s1: float32, ..., s{D-1}: float32, reward: float32, done: bool}`, N rows
— column-oriented so the Python side reads each state dimension as one zero-copy
`to_numpy(zero_copy_only=True)` buffer view (`D` = 12 or 16, matching the JSON protocol's
state width). The server distinguishes a reset vs. step request by column name (`seed`
vs. `action`) rather than a separate message-type field.

`BatchedRlEnv` shares one `MarketDataCache` (`Rc<RefCell<_>>`) across its N `RlEnv`
instances, so a date replayed by multiple instances in the batch is parsed from disk
once between them, not N times (see `MarketDataCache`'s doc comment in `rl_env.rs` and
[phase_s_baseline.md](phase_s_baseline.md) for the measurement this design responds to).

## State Vector (12-dim v5 / 16-dim v6)

| Index | Name | Formula | Range | Rationale |
|---|---|---|---|---|
| 0 | `q_frac` | `remaining / total_inventory` | [0, 1] | How much inventory is left to sell |
| 1 | `t_frac` | `steps_remaining / total_steps` | [0, 1] | How much time budget is left |
| 2 | `vol_ratio` | `tanh(current_sigma / sigma_ref)` | [0, 1) | Current vol relative to reference regime |
| 3 | `spread_ratio` | `tanh(half_spread / spread_ref)` | [0, 1) | Current spread relative to reference |
| 4 | `depth_ratio` | `tanh(depth_1pct / depth_ref)` | [0, 1) | Current book depth relative to reference |
| 5 | `ofi` | 30s order-flow imbalance from aggTrades | [-1, 1] | Short-term buy/sell pressure signal |
| 6 | `drift` | `tanh((mid_now - mid_10steps_ago) / mid_10steps_ago / sigma_ref)` | (-1, 1) | Recent price momentum |
| 7 | `slippage` | `tanh(last_slippage / spread_ref)` | (-1, 1) | How adverse the last fill was |
| 8 | `perm_impact` | `tanh(cf_lob.permanent_impact() / 0.5)` | [0, 1) | Accumulated permanent impact (0 outside counterfactual mode) |
| 9 | `temp_impact` | `tanh(cf_lob.temporary_impact_total() / 0.3)` | [0, 1) | Current un-decayed temporary impact (0 outside counterfactual mode) |
| 10 | `vol_regime` | discrete `{0.0, 0.5, 1.0}` | {0, 0.5, 1} | Low/normal/high vol classification |
| 11 | `liq_regime` | discrete `{0.0, 0.5, 1.0}` | {0, 0.5, 1} | Thin/normal/deep liquidity classification |
| 12† | `time_of_day` | `(timestamp_ms % 86_400_000) / 1000 / 86_400` | [0, 1] | UTC fraction of day — Binance liquidity varies sharply by hour |
| 13† | `intraday_vol_ratio` | `tanh(current_sigma / sigma_ref - 1)` | (-1, 1) | Intraday vol vs. daily reference |
| 14† | `sin_time` | `sin(2π · time_of_day)` | [-1, 1] | Cyclical encoding (00:00 ≈ 23:59) |
| 15† | `cos_time` | `cos(2π · time_of_day)` | [-1, 1] | Cyclical encoding, orthogonal to sin |

† Dims 12-15 exist only when `--no-time-features` is not passed (v6, the
new default). `--no-time-features` restores the exact 12-dim v5 vector for
evaluating pre-v6 checkpoints.

## RecurrentPPO Architecture

```
Observation (12 or 16 floats)
        │
        ▼
  MlpLstmPolicy (sb3-contrib)
        │
   ┌────┴────┐
   │  Actor   │              │  Critic  │
   │  LSTM    │              │  LSTM    │
   │ hidden=64│              │ hidden=64│   (v6 planned: 128 hidden, 2 layers)
   │ 1 layer  │              │ 1 layer  │
   └────┬────┘              └────┬────┘
        ▼                        ▼
   MLP [64, 64]              MLP [64, 64]
   Tanh activation            Tanh activation
        ▼                        ▼
  Softmax(20 actions)        Value estimate V(s)
```

`policy_kwargs`: `net_arch=dict(pi=[64,64], vf=[64,64])`,
`lstm_hidden_size=64`, `n_lstm_layers=1`, `enable_critic_lstm=True`.
Trained with `ent_coef` up to 0.05 (v5) to keep the policy exploratory
through ~1.5M steps — v4 collapsed to a near-deterministic policy by 500K
steps with a lower coefficient.

## `_CPULSTMWrapper` (PyTorch 2.1 MPS Bug)

**Root cause:** PyTorch 2.1's MPS (Apple GPU) backend has a broken LSTM
kernel — it crashes with an `MPSNDArrayDescriptor sliceDimension` error
whenever `sequence_length >= lstm_hidden_size`. Training on an M-series Mac
with the full model on `mps` device is otherwise desirable for the MLP
layers' throughput.

**Fix:** `_CPULSTMWrapper` (in `rl/train.py`) wraps the actor/critic
`nn.LSTM` submodules, forcing only those submodules to run on CPU while
leaving everything else (the surrounding MLP layers, which dominate
compute) on MPS. It proxies every attribute sb3-contrib reads directly
(`input_size`, `hidden_size`, `num_layers`, `batch_first`, `dropout`,
`bidirectional`) so the policy code sees a transparent drop-in
replacement, and moves tensors CPU↔MPS on each forward call.

**Performance:** the CPU↔MPS transfer overhead is small because LSTM
inputs at this state-vector size are tiny; the MLP layers, which stay on
MPS, dominate total compute time.

**Permanent fix:** upgrade to PyTorch ≥ 2.3, where the MPS LSTM kernel was
rewritten. The wrapper is safe to leave in place after upgrading — it adds
negligible overhead — so no code needs to change when the upgrade happens,
only the pinned `torch` version.

## `load_model()` State Dict Key Remapping

**Problem:** `_CPULSTMWrapper` (used during training when `device == "mps"`)
renames the state dict keys of the wrapped LSTM submodules from
`lstm_actor.weight_ih_l0` (sb3-contrib's expected name) to
`lstm_actor.lstm.weight_ih_l0` (one level deeper, because the LSTM is now
nested inside the wrapper as `self.lstm`). A model saved after MPS training
has this shifted key structure baked into `policy.pth` inside the model
zip; loading it directly with `RecurrentPPO.load()` on a machine without
the wrapper (e.g. `device='cpu'` at evaluation time) fails to match the
plain `MlpLstmPolicy`'s expected keys.

**Fix (`rl/evaluate.py:load_model()`):**
1. Open the model zip, read `policy.pth` (not `pytorch_variables.pth`) as
   raw bytes.
2. `torch.load()` the state dict, rewrite every key:
   `lstm_actor.lstm.` → `lstm_actor.`, `lstm_critic.lstm.` → `lstm_critic.`.
3. Re-serialize the patched state dict, write it into a temporary copy of
   the zip (all other files copied through unchanged).
4. `RecurrentPPO.load()` the patched zip with `device='cpu'`.

`load_model()` also auto-detects whether a zip is a recurrent model at all
(inspects `policy.pth` for any key containing `"lstm"`) and falls back to
plain `PPO.load()` for non-recurrent checkpoints.

## Action Space (V6, 20 discrete actions)

| Action | Fraction of remaining inventory | Action | Fraction |
|---|---|---|---|
| 0 | 0.000 (wait) | 10 | 0.060 |
| 1 | 0.005 | 11 | 0.075 |
| 2 | 0.010 | 12 | 0.100 |
| 3 | 0.015 | 13 | 0.125 |
| 4 | 0.020 | 14 | 0.150 |
| 5 | 0.025 | 15 | 0.200 |
| 6 | 0.030 | 16 | 0.300 |
| 7 | 0.035 | 17 | 0.500 |
| 8 | 0.040 | 18 | 0.750 |
| 9 | 0.050 | 19 | 1.000 (liquidate all) |

`nearest_action(target_frac, fractions)` maps a classical policy's desired
continuous sell fraction to the nearest discrete action by absolute
difference — this is how `TWAPPolicy`, `HeuristicPolicy`, `ACOptimalPolicy`,
and `AdaptiveACPolicy` (all in `rl/evaluate.py`) share the exact same
20-action discrete space as the RL agent, so all five strategies are
directly comparable step-for-step.

## Related Documents

- [HLD.md](HLD.md) — component diagram, data flow, scalability
- [ADR.md](ADR.md) — the 6 architecture decisions with alternatives and consequences
