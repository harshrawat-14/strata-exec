# Phase S — Baseline Measurement Report

Measured before any Phase S implementation change, on the pre-Phase-S JSON-over-stdio
`rl-env` binary (commit `a7fe081`, macOS/Darwin, M-series). All numbers below are
reproducible via `docs/system_design/phase_s_bench_baseline.py` and the `dhat-heap`
Cargo feature added for this report.

## 1. Steady-state throughput (Python `StrataExecEnv` → `rl-env` subprocess)

Driven exactly as `rl/evaluate.py`'s `_run_episode` drives it: `TWAPPolicy`,
deterministic, real `2024-01-15` historical data, `fixed_steps`/`fixed_size` so episode
length is constant. One subprocess reused across all 20 episodes (this measures
per-step/per-reset marginal cost, not process startup).

```
date=2024-01-15 episodes=20 total_steps=57580
wall_time_s=4.7654
steps_per_sec=12082.9
episodes_per_sec=4.1969
mean_episode_time_s=0.2383
mean_steps_per_episode=2879.0
```

## 2. Where the episode time actually goes

This is the finding that reframes the rest of Phase S. Isolating `reset()` alone
(10 calls, same date) against a full episode (`reset()` + 2880 `step()` calls):

```
mean reset() time:               239.14 ms
full episode (reset+2880 steps): 231.51 ms
reset() as pct of episode:       103.3%
```

**`reset()` alone accounts for ~100% of a full episode's wall time.** The 2880-step
`step()` loop that follows it is, at the wall-clock level, comparatively free. This
directly contradicts the assumption (reasonable a priori, and the one this project's
plan started from) that the per-step JSON transport is the throughput bottleneck.

## 3. Why: `reset()` re-parses the entire historical dataset from disk every episode

Allocation profile via `dhat-heap` (`cargo build --release --bin rl-env --features
dhat-heap`), one `reset()` + a 2879-step episode on `2024-01-15`:

| | blocks (allocations) | bytes |
|---|---:|---:|
| `RlEnv::reset()` (`AggTradesDay::from_csv`, `LobReplay::from_csv`) | 4,320,684 | 302,831,384 (~303 MB) |
| `RlEnv::step()` (`build_state()`, per-step JSON encode/decode) | 46,060 | 3,552,362 (~3.6 MB) |
| process startup / CLI parsing / stdio locking (one-time) | 11,544 | 2,631,973 |

`reset()` is **99% of total allocations by count and by bytes** for this run. The top
individual allocation sites are all inside `AggTradesDay::from_csv` (the aggTrades file
for `2024-01-15` has **1,036,190 rows** — parsed with `csv::StringRecordsIter`, `String`
allocations per field, and a `str::to_lowercase()` call per row) and `LobReplay::from_csv`
(28,800 rows). **Every single episode reset re-parses these multi-million-row CSV files
from scratch**, even though the same date is replayed across many episodes in both
evaluation (`evaluate_all_strategies_on_date` runs 50 episodes per strategy per date) and
training (`--multi-date` samples repeatedly from a fixed date list across millions of
timesteps).

`step()` alone: **16.0 allocations/call, 1,234 bytes/call** — non-trivial for a hot loop
called ~2,880 times per episode and potentially billions of times over a full training
run, but two orders of magnitude smaller in both count and bytes than one `reset()` call.

## 4. What this means for Phase S's priorities

The master prompt's Phase S plan (replace JSON-over-stdio with a zero-copy transport,
eliminate `step()` allocations) is still worth doing and is still in scope — 16
allocations/step compounds over a full training run, and a slow transport format is a
real, measurable cost. But **it is not the dominant lever**. The dominant lever, by a
factor of ~94x in allocation count and ~85x in bytes, is that `reset()` re-parses
multi-million-row CSVs from disk on every episode instead of once per date. Per Global
Rule 6 (trust the repo/measurements over the prompt's a priori assumptions), Phase S's
implementation plan adds a step the original plan didn't anticipate:

1. **Cache parsed per-date market data across resets** (new, highest-impact item found
   by this baseline). `AggTradesDay`/`LobReplay` should be parsed once per date and
   shared (e.g. `Arc`) across every subsequent `reset()` call for that date, including
   across the batched env's N parallel instances. This is the fix the measurement
   actually calls for.
2. **Zero-copy transport for the state vector / LOB snapshot handoff** (as planned) —
   real, but a secondary win now that `reset()`'s cost has been isolated.
3. **Eliminate `step()`'s 16 allocations/call** (as planned) — pre-allocate `build_state()`'s
   buffer and avoid the `String` round-trip through `serde_json::to_string`, validated by
   re-running this same `dhat-heap` profile after the change.

All three are implemented below; the throughput table in the HLD update reports
before/after numbers for all three, not just the transport.

## 5. Reproduction

```bash
# Throughput
rl/.venv/bin/python docs/system_design/phase_s_bench_baseline.py

# Allocation profile (writes dhat-heap.json to CWD; view at https://nnethercote.github.io/dh_view/dh_view.html)
cargo build --release --bin rl-env --features dhat-heap
# then drive it via the same StrataExecEnv path, or a raw stdin/stdout script,
# for a SHORT episode -- dhat's per-allocation backtrace capture is expensive
# enough on this CSV-parse-heavy workload that a full-length run takes minutes,
# not milliseconds (learned the hard way while producing this report).
cargo build --release --bin rl-env   # rebuild the plain binary afterward
```
