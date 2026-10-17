# Disk cleanup manifest — 2026-09-07

Record of what was deleted, why, and how to regenerate it. Written **before** the deletions were
executed, so nothing is removed without a provenance trail. Nothing in this list was cited by any
document, result table, test, or `.gitignore` whitelist — each candidate was grep-checked against
`docs/`, `rl/*.py`, `rl/tests/`, `README.md`, and `.gitignore` first.

## Summary

| category | reclaimed | reversible? |
|---|---:|---|
| `target/debug/` (Rust debug build artifacts) | ~10 GB | yes — `cargo build` |
| smoke/throwaway RL checkpoints (11 dirs + 7 zips) | ~9 MB | no — but none were referenced by any result |
| smoke/throwaway TensorBoard logs (10 dirs) | ~660 KB | no — training-time telemetry only |

## Explicitly NOT deleted (checked and kept)

- **`rl/results/*.POSTSTEP0_DISCARDED.csv.bak`** — `docs/experiments/05-tail-risk-validation.md:256-257` states
  these are *"kept, not deleted, as the evidence trail for the provenance finding"* (the discarded
  post-Step-0 run that first revealed the binary-version issue). Deliberate evidence, not clutter.
- **`target/release/`** — contains `target/release/rl-env`, the binary every Python evaluation and
  training run drives over stdin/stdout. A full `cargo clean` would have removed it and silently
  broken the next eval until rebuilt. Only `target/debug/` was removed.
- **`rl/models/smoke_v5_final.zip`** — git-tracked, whitelisted in `.gitignore:25`, and named in
  `README.md:893` as a model the ARQ worker loads. Despite the "smoke" name, it is a live dependency.
- **`strataexec.db`** — tracked in git; flagged as questionable in the repo analysis but NOT touched
  here, since deciding whether to untrack it needs a look at its contents first.

## Deleted: RL checkpoints

All are smoke tests, device-confirmation runs, or isolation experiments — none produced a number in
any results table. `best_mean_reward` recorded here so the runs remain auditable after deletion.

| checkpoint | size | created | best_mean_reward | purpose |
|---|---:|---|---:|---|
| `ppo_lstm_v6r_constlr_isolation_best/` | 752K | 2026-07-15 | −0.00466032 | LR-schedule isolation test |
| `ppo_lstm_v6r_entfloor_smoke_best/` | 752K | 2026-07-15 | −0.0046717 | entropy-floor smoke test |
| `ppo_lstm_v6r_integrated_smoke_best/` | 752K | 2026-07-16 | −0.00725102 | integrated smoke test |
| `ppo_lstm_v6r_mps_deviceconfirm_best/` | 752K | 2026-07-16 | −0.00475236 | MPS device confirmation |
| `ppo_lstm_v6r_smoketest_best/` | 752K | 2026-07-13 | (none) | smoke test |
| `smoke_v5_best/` | 0B | 2026-05-26 | (empty) | empty dir |
| `smoke_mps_best/` | 0B | 2026-05-22 | (empty) | empty dir |
| `smoke_test_lstm_best/` | 0B | 2026-05-22 | (empty) | empty dir |
| `phase3_smoke_best/` | 0B | 2026-08-13 | (empty) | empty dir |
| `lstm32_crash_test_best/` | 0B | 2026-05-22 | (empty) | empty dir |
| `mps_crash_test_best/` | 0B | 2026-05-22 | (empty) | empty dir |

Loose `_final.zip` counterparts also deleted: `phase3_smoke_final`, `smoke_mps_final`,
`smoke_test_lstm_final`, `ppo_lstm_v6r_smoketest_final`, `ppo_lstm_v6r_entfloor_smoke_final`,
`ppo_lstm_v6r_integrated_smoke_final`, `ppo_lstm_v6r_constlr_isolation_final` (~708–752K each).

## Deleted: TensorBoard logs

`rl/logs/` entries for the same throwaway runs: `smoke_v5`, `smoke_mps`, `smoke_test_lstm`,
`lstm32_crash_test`, `mps_crash_test`, `ppo_lstm_v6r_smoketest`, `ppo_lstm_v6r_entfloor_smoke`,
`ppo_lstm_v6r_integrated_smoke`, `ppo_lstm_v6r_mps_deviceconfirm`, `ppo_lstm_v6r_constlr_isolation`.
(4K–320K each.) `rl/logs/` is gitignored; these were never version-controlled.

## Checkpoints retained (the ones results actually cite)

`ppo_lstm_v5_adaptive_best` (git-tracked, README-cited) · `ppo_ablation_12dim_cf_best` /
`ppo_ablation_16dim_cf_best` (the 12-vs-16-dim ablation) · `ppo_lstm_v6_time_features_best` ·
`ppo_lstm_v6r_time_features_best` · `phase3_step2_16dim_v2_best` ·
`phase3_step2b_allcoverage_best` (numbers cited across Phase 3 docs) ·
`phase3_step4_capacity_entropy_best` (Step A headline) · `phase3_stepB_bc_pretrain` /
`phase3_stepB_warmstart_finetune_best` (Step B) · `ppo_counterfactual_v1/v2/v3_best` ·
`ppo_passive_best` / `ppo_passive_v2_best` · `ppo_lstm_v1`–`v4` lineage · `rl/models/checkpoints/`.

## How to regenerate

- `target/debug/`: `cargo build` (the release binary was untouched, so no eval workflow is affected).
- Smoke checkpoints: not worth regenerating — they were throwaway validation runs whose only purpose
  was to confirm a code path ran, and each was superseded by the real run that followed it.

## Still outstanding after this cleanup (from the repo analysis, not acted on)

1. `strataexec.db` (7.1 MB) is committed to git — needs a contents check before untracking.
2. `rl/batched_vec_env.py` / `rl/batched_environment.py` (Phase S Arrow transport) are **not used by
   `train.py` or `evaluate.py`** — only by their own tests. ADR-006 describes this path as adopted;
   it is not. Either wire it in or correct the ADR.
3. `README.md` still carries a stale **⚠️ UNVERIFIED** banner on the Phase 4 RL section, superseded
   by the V2 split → Step 2b → Step 4 → Step A chain.
4. ~11 `phase2r_*` one-off scripts could move to `rl/experiments/archive/` for legibility.
5. `scratch/` is empty; `historical_trades.csv` is a 4-line toy fixture at repo root.
