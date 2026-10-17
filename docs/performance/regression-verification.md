# Phase S — Regression Verification Against Phase 0

Phase S's implementation (per-date market data caching, hot-loop allocation elimination,
graceful degradation on bad data, and the additive batched-env/Arrow transport) must not
change simulation semantics — only performance. This is a hard gate per the Phase S plan:
"the first question any reviewer of this work should ask, and the answer needs to be a
number, not an assurance."

## Method

Built `rl-env` at two commits and ran the actual Phase 0 unified comparison harness
(`evaluate_all_strategies_on_date`, driven via `docs/performance/regression_check.py`)
against each binary in turn, same Python code both times (only the compiled Rust binary
changed):

- **BEFORE**: commit `a7fe081` ("Phase 0: declare crash-day holdout split..."), the last
  commit before any Phase S work — built via `git worktree add /tmp/strataexec-before
  a7fe081 && cargo build --release --bin rl-env`.
- **AFTER**: HEAD at the time of this check (`9cdcd42`), including all of Phase S's
  caching, allocation, graceful-degradation, and batched-transport changes.

Both binaries were copied in turn to `target/release/rl-env` (the path `environment.py`
hardcodes) and driven by the identical current `rl/evaluate.py`, evaluating all 5
strategies (TWAP, Heuristic, AC Optimal, Adaptive AC, RL Agent using
`ppo_lstm_v5_adaptive_best`) on `2024-01-15`, 10 episodes, same seeds (0–9).

## Result

```
BEFORE                                          AFTER
twap:        mean_IS=-0.439257  cvar_95=-0.439736   twap:        mean_IS=-0.439257  cvar_95=-0.439736
heuristic:   mean_IS=-0.439512  cvar_95=-0.439721   heuristic:   mean_IS=-0.439512  cvar_95=-0.439721
ac_optimal:  mean_IS=-0.437392  cvar_95=-0.441100   ac_optimal:  mean_IS=-0.437392  cvar_95=-0.441100
adaptive_ac: mean_IS=-0.440078  cvar_95=-0.442350   adaptive_ac: mean_IS=-0.440078  cvar_95=-0.442350
rl_agent:    mean_IS=-0.534946  cvar_95=-0.537281   rl_agent:    mean_IS=-0.534946  cvar_95=-0.537281
```

`diff` on the full output (mean_IS, cvar_95, std_IS, is_sharpe for all 5 strategies):
**zero differences, byte-for-byte identical.**

## Verdict

**PASSED.** Phase S's changes did not shift any strategy's IS numbers — not even at
floating-point precision, let alone beyond tolerance. This confirms the caching layer,
the `build_state()`/`write_msg()`/input-loop allocation changes, and the graceful
`Result`-based error handling on the LOB/aggTrades load path are behavior-preserving for
the successful (non-error) path, which is the only path Phase 0's harness exercises.

## Scope note (original)

This first check verified the single-instance JSON-over-stdio path — the one Phase 0's
harness (`evaluate_all_strategies_on_date`) and every existing evaluation call site
actually use. It did **not** cover the new batched Arrow-transport path (`BatchedRlEnv` /
`rl/batched_environment.py`), and specifically did not run through an SB3 `VecEnv`
wrapper — no such wrapper existed yet at the time this section was first written, and
`rl/train.py`'s own wrapper is non-functional for a reason predating and unrelated to
Phase S (a `no_regime_impact` kwarg mismatch, deferred to Phase 2). **This gap was flagged
in review and is closed below.**

## Follow-up: regression-verify through the actual SB3 VecEnv training wrapper

A standalone-harness pass, however clean, doesn't establish that the batched-env → VecEnv
adapter (batch ordering, reset semantics, done-flag handling, observation dtype/shape) is
correct — that's a different code path from `evaluate_all_strategies_on_date` entirely,
and exactly where a silent regression would hide. No such adapter existed before this
follow-up: `BatchedStrataExecVecEnv` (`rl/batched_vec_env.py`) was built specifically to
close this gap, implementing SB3's `VecEnv` abstract interface (`reset`, `step_async`,
`step_wait`, `close`, `get_attr`, `set_attr`, `env_method`, `env_is_wrapped`) around
`BatchedStrataExecEnv`.

**The auto-reset-on-done contract is the specific risk.** SB3's `VecEnv` requires that
when one instance's episode ends, *only that instance* resets (with the new episode's
observation spliced into its slot and the true final observation stashed in
`info["terminal_observation"]`), while every other in-progress instance continues
untouched. `BatchedRlEnv` had no way to do this — `reset_batch` always reset all N
instances together. Built `BatchedRlEnv::reset_indices` (Rust) plus a matching
`reset_indices` method (`rl/batched_environment.py`) and a third wire-protocol request
shape (`{index: uint32, seed: uint64}`) specifically for this, with its own correctness
test (`reset_indices_only_resets_specified_instances` — proves untouched instances'
trajectories are bit-identical to continuing them directly, not silently reset).

**Test** (`rl/tests/test_batched_vec_env.py::test_batched_vec_env_matches_single_instance_json_path`):
drove `BatchedStrataExecVecEnv` with `batch_size=4`, real `2024-01-15` historical data,
and — critically — `fixed_steps=False, fixed_size=False` (real training config:
randomised per-episode horizons), so different instances genuinely finish episodes at
different global step counts, actually exercising the auto-reset path (Phase 0's harness
always uses `fixed_steps=True`, where every instance finishes together and this whole
distinction is invisible). Ran 400 steps, logging every reset (initial + auto-reset) with
its seed via an instrumentation hook (`VecEnv.reset_log`), then replayed each of those
exact per-slot episode segments through the single-instance JSON path
(`environment.py`'s `StrataExecEnv`, the same transport `evaluate.py` uses) with the same
seed and the same per-step actions, comparing every observation array.

**Result:**

```
batched-VecEnv vs single-instance JSON: 84 episode segments, 1604 observation points, all matched exactly
```

84 independent episodes across the 4 slots over 400 steps (confirming randomised horizons
did cause staggered episode endings — the auto-reset path was genuinely exercised, not
skipped), 1,604 observation points, `np.array_equal` on every one (exact match, not just
within tolerance). Additionally smoke-tested with the real `ppo_lstm_v5_adaptive_best`
`RecurrentPPO` model driving `predict()` through the VecEnv for 30 steps, confirming
observation shape `(4, 12)` and dtype `float32` are exactly what the policy expects.

**Verdict: PASSED.** The batched-env → VecEnv adapter Phase 2's retrain will actually
train through has now been regression-verified, not just the standalone harness. Phase 2
is no longer blocked on this.
