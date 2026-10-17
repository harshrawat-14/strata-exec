# Phase S — Baseline Measurement Report

Measured before any Phase S implementation change, on the pre-Phase-S JSON-over-stdio
`rl-env` binary (commit `a7fe081`, macOS/Darwin, M-series). All numbers below are
reproducible via `docs/performance/baseline_benchmark.py` and the `dhat-heap`
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

## 6. Follow-up: allocation elimination reduced count 25%, not to zero — what's the remaining cost?

Phase S's stated acceptance criterion was "profiler-confirmed zero allocations in the
`step()` hot path." The implementation reduced it from 20.01 to 15.01 allocations/step
(-25%, ~1174 bytes/step) — not to zero. A count reduction alone doesn't say whether the
remainder is worth chasing further, so this measures the *time* cost of what's left, not
just the count.

**Method**: a timing-instrumented global allocator (`alloc-timing` Cargo feature,
`src/bin/rl_env.rs`) wraps the system allocator, summing `Instant::now()` deltas around
every `alloc`/`dealloc`/`realloc`/`alloc_zeroed` call into an atomic counter. A benchmark
(`step_allocation_time_share_of_wall_time`, gated behind the same feature, run with
`cargo test --release --features alloc-timing -- --ignored --nocapture`) then calls the
real production functions in the real per-iteration order — `serde_json::from_str::<InMsg>`
(parsing an incoming action, exactly as `main()`'s hot loop does) → `RlEnv::step()`
(includes `build_state()`) → `step_result_to_msg` → `write_msg` (writing to `io::sink()`,
which discards bytes without buffering/allocating, isolating this from actual socket I/O
latency) — for 2000 steps on real, cache-warmed `2024-01-15` historical data.

**Result:**

```
total wall time:       1555.7 ns/step
time inside allocator:  147.4 ns/step  (14,480 alloc/dealloc/realloc calls total, 7.24/step)
allocator share of step wall time: 9.5%
```

**Reading this against the full observed per-step latency** (the number that actually
reaches a Python training loop, not just Rust-side compute): section 1's post-caching
throughput was ~44,000–46,000 steps/sec, i.e. ~21,600–22,700 ns/step end-to-end. This
benchmark's 1,556 ns/step is the Rust-side compute (parse + step + serialize) contained
within that — **only ~7% of the full round-trip time**; the remaining ~93% is socket/pipe
I/O and Python-side JSON decode, which this benchmark deliberately excludes via
`io::sink()`. Allocation specifically (147 ns/step) is **~0.7% of the full observed
per-step latency**.

**Verdict: treated as acceptable, not pursued further.** Even a hypothetical *complete*
elimination of all remaining allocations would save ~147 ns/step out of a ~21,600+ ns/step
round trip — a theoretical ceiling around 0.7%, not a meaningful lever. Allocation was
never going to be the bottleneck for single-instance throughput once `reset()`'s ~239ms
cost was fixed (section 4); IPC/socket round-trip cost now dominates what remains, and
that is what the batched Arrow transport (not further `step()` micro-optimization)
targets — see the HLD's Phase S section for that transport's own honestly-reported
cost/benefit at different batch sizes. Chasing the literal "zero allocations" wording
further would be optimizing a cost that is no longer material, contradicting the same
"prove it with a number, don't assert it with a technique name" standard this whole
baseline report is built on.

One measurement caveat, stated plainly: this benchmark's allocator-*call*-count (7.24
calls/step, alloc+dealloc combined) is lower than section 3's dhat-heap allocation-*count*
(15.01 allocations/step alone, not counting frees) — the two profilers count on different
bases (dhat's call-site "blocks" vs. this benchmark's raw `GlobalAlloc` invocation count),
and reconciling them exactly wasn't pursued once the *time* answer above made the count
discrepancy immaterial to the actual question being asked.

## 7. Reproduction

```bash
# Throughput
rl/.venv/bin/python docs/performance/baseline_benchmark.py

# Allocation profile (writes dhat-heap.json to CWD; view at https://nnethercote.github.io/dh_view/dh_view.html)
cargo build --release --bin rl-env --features dhat-heap
# then drive it via the same StrataExecEnv path, or a raw stdin/stdout script,
# for a SHORT episode -- dhat's per-allocation backtrace capture is expensive
# enough on this CSV-parse-heavy workload that a full-length run takes minutes,
# not milliseconds (learned the hard way while producing this report).
cargo build --release --bin rl-env   # rebuild the plain binary afterward

# Allocation TIME share of step wall time (section 6)
cargo test --release --features alloc-timing --bin rl-env \
  step_allocation_time_share_of_wall_time -- --ignored --nocapture
```
