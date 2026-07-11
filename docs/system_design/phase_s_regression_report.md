# Phase S — Regression Verification Against Phase 0

Phase S's implementation (per-date market data caching, hot-loop allocation elimination,
graceful degradation on bad data, and the additive batched-env/Arrow transport) must not
change simulation semantics — only performance. This is a hard gate per the Phase S plan:
"the first question any reviewer of this work should ask, and the answer needs to be a
number, not an assurance."

## Method

Built `rl-env` at two commits and ran the actual Phase 0 unified comparison harness
(`evaluate_all_strategies_on_date`, driven via `docs/system_design/phase_s_regression_check.py`)
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

## Scope note

This verifies the single-instance JSON-over-stdio path — the one Phase 0's harness
(`evaluate_all_strategies_on_date`) and every existing evaluation call site actually use.
The new batched Arrow-transport path (`BatchedRlEnv` / `rl/batched_environment.py`) is a
genuinely new code path with its own tests (numeric equivalence against N single-instance
`RlEnv`s, proven in `src/bin/rl_env.rs`'s `batched_env_reset_and_step_match_n_single_instances`
test) rather than a regression target, since nothing used it before Phase S.

This check was **not** run through `rl/train.py`'s SB3 `VecEnv` wrapper, because
`rl/train.py` is currently non-functional for a reason predating and unrelated to Phase S
(the Phase -1 audit found it passes a `no_regime_impact` kwarg `StrataExecEnv.__init__`
doesn't accept — a `TypeError` on every invocation). Fixing that is explicitly scoped to
Phase 2's start, by prior agreement — this regression check instead used
`evaluate_all_strategies_on_date`, which is Phase 0's actual hardened harness and the one
every existing comparison table in this project is built on.
