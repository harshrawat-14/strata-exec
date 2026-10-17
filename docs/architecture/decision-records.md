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
snapshots combined with an accumulated transient-impact model (an
Obizhaeva-Wang-style permanent/temporary decay structure paired with a
square-root instantaneous magnitude — see the README's "Market Impact"
section for the Phase 2R naming correction) — rather than pure historical
replay of the raw order book.

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
remap before loading on a machine without the wrapper (see low-level-design.md). This
workaround should be removed (or simply left in place, since it is
harmless overhead) once the project's pinned PyTorch version is upgraded to
≥ 2.3, where the underlying MPS LSTM kernel was rewritten.

---

## ADR-006: Arrow IPC over a Unix domain socket for the batched hot-loop transport, plus per-date data caching

**Status:** Accepted

**Context — this partially revises ADR-001.** ADR-001 accepted JSON-over-stdio on the
grounds that "serialization overhead is negligible relative to policy inference time."
Phase S's baseline measurement (`docs/performance/baseline-profiling.md`) both confirms
and complicates that claim: `step()`'s JSON marginal cost is indeed small in absolute
terms (16 allocations/call, ~12,000 steps/sec on a single instance) — but `reset()`
dominates total episode wall time (~103% of it), because it re-parses the entire
multi-million-row historical CSV from disk on every episode. **The measured bottleneck
is not the transport codec; it's redundant per-episode I/O and parsing.** This ADR
covers both fixes together because the vectorized batched env (Phase S task 3) is the
natural place to do both: N instances sharing one cached, pre-parsed dataset, and one
batched state handoff to Python instead of N separate round-trips.

**Decision:**
1. **Cache parsed per-date market data** (`AggTradesDay`, `LobReplay`) behind a
   process-local cache keyed by date, shared via `Arc` across every `RlEnv` instance
   (including all N instances inside a batched env) that replays that date. Parse once
   per date per process, not once per episode.
2. **Replace JSON-over-stdio with Arrow RecordBatch IPC over a Unix domain socket** for
   the batched `step_batch`/`reset_batch` hot path. The single-instance JSON protocol
   from ADR-001 is left in place for the non-batched `rl-env` binary (still used by
   `rl/evaluate.py`'s per-strategy episodes, where the transport cost was never the
   bottleneck and JSON's debuggability is worth keeping) — this is additive, not a
   full replacement.

**Alternatives considered (for item 2):**
- **True shared-memory ring buffer** (a hand-rolled `mmap`'d region with manual
  producer/consumer sequence numbers). Highest theoretical throughput and the only
  option that is *literally* zero-copy end-to-end (no bytes cross a socket at all).
  Rejected for this codebase: manual synchronization primitives are exactly the kind of
  code where a subtle bug silently corrupts state rather than crashing loudly — a bad
  trade on a research platform that depends on Global Rule 10's per-instance RNG
  isolation being airtight, and platform-specific enough (macOS vs Linux shared-memory
  APIs differ) to add real portability risk. Given the baseline shows the per-step
  transport cost is not the bottleneck, the extra correctness risk isn't bought by a
  correspondingly large measured win.
- **Cap'n Proto / FlatBuffers.** Also zero-copy on deserialization (direct struct access
  on the wire format, no unpacking pass), and would work. Rejected in favor of Arrow
  because our actual payload shape — N environment instances × a fixed-width float
  state vector, plus per-instance reward/done scalars — is exactly Arrow's home turf
  (a columnar `RecordBatch`), whereas Cap'n Proto/FlatBuffers have no batch-of-numeric-
  arrays abstraction and would require hand-rolling that layout inside a schema. Arrow
  also has substantially richer Python-side tooling for a research platform: a captured
  message can be loaded and inspected interactively via `pyarrow` (schema, column
  stats, `to_pandas()`) without writing a decoder, which JSON already gave us and this
  choice does not give up.
- **Arrow RecordBatch, chosen.** Deserialization on the Python side is a buffer-view
  reinterpretation via `pyarrow`, not a parse (this is the conventional meaning of
  "Arrow enables zero-copy" — the *decode* step is zero-copy even though bytes still
  cross the socket, unlike shared memory). Natural fit for `step_batch(actions: &[Action])
  -> StateBatch` (a `RecordBatch` with N rows). Strong library support on both sides
  (`arrow-rs`, `pyarrow`). Self-describing schema keeps debuggability close to JSON's,
  which matters more here than in a live-trading system (explicit scope boundary, see
  Phase S framing) — correctness and the ability to inspect a wire message by hand
  outweigh squeezing out the last-mile latency a hand-rolled shared-memory buffer would
  give, especially since that latency was never the measured bottleneck.

**Rationale for the socket (not stdio) as the batched channel:** the existing
stdin/stdout pipe is line-oriented and shared with the single-instance JSON protocol; a
dedicated Unix domain socket keeps the batched Arrow protocol as a separate, independently
testable channel without overloading the meaning of a line on stdout, and is a small,
well-understood step from "subprocess pipe" (no shared-memory allocator, no cross-platform
POSIX shm quirks).

**Consequences:** Two transports now exist side by side — JSON-over-stdio (ADR-001,
unchanged, still used by the non-batched single-episode path) and Arrow-over-socket
(this ADR, used by the new batched env). This is accepted as the correct scope: Phase S's
explicit boundary is an offline research platform, not a live-trading rewrite, and
`rl/evaluate.py`'s existing single-episode call sites do not need to change to get the
throughput win, since that win doesn't come from the transport for those call sites in
the first place (see baseline report). The per-date cache must be invalidated per
process, not persisted across `rl-env` invocations — this is intentional: it caches
within one training run's process lifetime, not across runs, so a changed CSV file on
disk is always picked up by the next process start.
