# StrataExec — Architecture Decision Records

## ADR-001: JSON over stdin/stdout for IPC

**Status:** Accepted

**Decision:** The Rust simulation core (`rl-env` binary) and the Python RL
stack communicate via newline-delimited JSON messages over the subprocess's
stdin/stdout, rather than a native language binding.

**Alternatives considered:**
- PyO3 / maturin native bindings (Rust compiled as a Python extension module)
- gRPC or another RPC framework over a local socket
- Shared memory / memory-mapped state

**Rationale:** JSON over stdin/stdout requires no shared build toolchain
between Rust and Python, no ABI compatibility constraints, and is trivially
debuggable (any message can be logged, replayed, or hand-constructed for a
test). The state vector is 12-16 floats and messages are exchanged once per
simulation step (not once per tick of a live feed), so serialization
overhead is negligible relative to policy inference time. A native binding
would reduce that overhead further but would couple Rust compiler/toolchain
versions to the Python environment's ABI and complicate independently
testing, versioning, and deploying the two halves — a cost not justified by
the IPC overhead this workload actually incurs.

**Consequences:** Every new field added to the state vector or info payload
must be threaded through both the Rust `OutMsg`/`InMsg` structs and the
Python `StrataExecEnv` wrapper by hand; there is no schema shared at compile
time between the two languages. This is an accepted ongoing cost.

---

## ADR-002: Counterfactual LOB over historical replay

**Status:** Accepted (supersedes pure historical replay, generation 1→2)

**Decision:** Train and evaluate on a counterfactual LOB — real Binance L2
snapshots combined with an accumulated Obizhaeva-Wang impact model — rather
than pure historical replay of the raw order book.

**Alternatives considered:**
- Pure historical replay (agent's fills have no effect on future snapshots)
- Fully synthetic unlimited-depth LOB with a square-root impact model

**Rationale:** Pure historical replay has no causal feedback: an agent's own
fill at step *t* does not change what it sees at step *t+1*, so instant
full liquidation carries zero simulated consequence and is trivially
"optimal." The fully-synthetic unlimited-depth approach (generation 1) has
the opposite problem — the square-root impact formula is concave in trade
size, so for a fixed total quantity, splitting into *n* steps costs *more*
(∝ √n) than trading all at once (∝ 1), which again makes instant
liquidation the reward-maximizing policy, just for a different reason. The
counterfactual LOB fixes both: real snapshots preserve authentic price
levels and spread dynamics, and the accumulated OW impact (permanent +
decaying temporary) means a large fill measurably depletes the book the
agent faces on the next step, making patient execution mechanically more
profitable.

**Consequences:** Episode length grew from 2-7 steps (both prior
approaches) to up to 2,481 steps once the degenerate-liquidation exploit
was closed — validating that the fix worked. The impact model's parameters
(permanent/temporary split, decay rate, book-depth calibration) become a
first-class calibration surface that must be validated against real market
data, which is what generation 3 (calibration gap finding: temporary impact
~10× larger, spreads ~95× wider than the synthetic proxy assumed) did.

---

## ADR-003: Pre-computed demo mode

**Status:** Accepted

**Decision:** Serve a live public demo (Render + Vercel) from pre-computed
JSON results (`demo_results/`) via `DemoModeMiddleware`, rather than running
live Rust subprocess computation on every request.

**Alternatives considered:**
- Live compute on Render's free tier for every simulate/evaluate request
- Paid always-on compute tier sized for worst-case concurrent load
- No public demo; local-only

**Rationale:** A free-tier web dyno cannot reliably host on-demand Rust
subprocess spawning under real internet traffic — cold starts, timeout
limits, and concurrent-request resource contention would make the demo
flaky or force a paid tier disproportionate to a portfolio/research
project's actual traffic. Pre-computing a fixed set of simulation presets
and evaluation dates and serving them as static JSON makes the public demo
instant, free, and impossible to crash from load, while `DEMO_MODE=false`
still exercises the exact same live pipeline for local development and
custom parameters.

**Consequences:** The public demo can only show the pre-computed presets
and dates in `demo_results/manifest.json` — custom parameter combinations
return a "run locally" message rather than a live result. `demo_results/`
must be treated as a frozen, deliberately-updated artifact (never a side
effect of an unrelated script run), since it is the sole source of truth
for what production actually serves.

---

## ADR-004: RecurrentPPO over DQN/SAC

**Status:** Accepted

**Decision:** Train the RL agent with RecurrentPPO (an LSTM-augmented
policy/value network), rather than a memoryless algorithm like DQN or SAC
with a feedforward network.

**Alternatives considered:**
- DQN with a feedforward MLP (generation 1 approach)
- SAC (continuous-action variant, would require redesigning the discrete
  20-action space)
- Feedforward PPO with a stacked-frame observation (last *k* states
  concatenated) as a manual substitute for memory

**Rationale:** Optimal execution is sequential by nature: the correct
action at step *t* depends on the agent's own trading history within the
episode — how much book depth it has already consumed, how the remaining
time budget has evolved, whether recent fills have been adverse. A
feedforward policy conditioned only on the current state vector cannot
represent "I sold 200K units in the last 10 steps, the book is thinning, I
should slow down," because that fact is not recoverable from any single
state snapshot, only from the trajectory. An LSTM hidden state carries that
trajectory information across steps within an episode without manually
engineering a fixed-length history window.

**Consequences:** Training requires explicit LSTM state management
(`episode_start` flags, hidden/cell state threading) in both `train.py` and
`evaluate.py`'s `_run_episode()`, and introduced the PyTorch 2.1 MPS LSTM
bug that `_CPULSTMWrapper` works around (see ADR-005). Classical baseline
policies (TWAP, Heuristic, AC Optimal, Adaptive AC) do not need this
machinery since they are stateless functions of the current observation,
but they implement the same `.predict(obs, state, episode_start,
deterministic)` interface so `_run_episode()` can run any of the five
strategies identically.

---

## ADR-005: `_CPULSTMWrapper` for the MPS PyTorch 2.1 bug

**Status:** Accepted (workaround; superseded once PyTorch ≥ 2.3 is adopted)

**Decision:** Wrap the RecurrentPPO policy's actor/critic `nn.LSTM`
submodules in `_CPULSTMWrapper`, forcing only those submodules to execute
on CPU while the rest of the model runs on Apple Silicon's MPS backend.

**Alternatives considered:**
- Train entirely on CPU (correct, but loses MPS throughput for the MLP layers)
- Downgrade to a PyTorch version with a working MPS LSTM kernel
- Wait for / immediately upgrade to a fixed PyTorch release

**Rationale:** PyTorch 2.1's MPS LSTM kernel crashes with an
`MPSNDArrayDescriptor sliceDimension` error whenever
`sequence_length >= lstm_hidden_size` — exactly the regime this project
trains in. Training fully on CPU was measured to be meaningfully slower for
the MLP-heavy layers. Moving only the LSTM submodules to CPU isolates the
broken code path while keeping the throughput-dominant MLP layers on MPS.

**Consequences:** Every model saved after MPS training has shifted state
dict key names for the wrapped LSTM submodules (`lstm_actor.lstm.*` instead
of `lstm_actor.*`), which `rl/evaluate.py`'s `load_model()` must detect and
remap before loading on a machine without the wrapper (see LLD.md). This
workaround should be removed (or simply left in place, since it is
harmless overhead) once the project's pinned PyTorch version is upgraded to
≥ 2.3, where the underlying MPS LSTM kernel was rewritten.
