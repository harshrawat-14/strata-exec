# rl/experiments — frozen experiment scripts

One-off scripts behind the results in `docs/experiments/`. They are kept so
every published number can be reproduced, and are **not** maintained as
library code: new work belongs in `rl/` modules, not here.

Run from the repository root, e.g.

```
python rl/experiments/phase3_stepA_full_holdout_eval.py --model rl/models/<ckpt>/best_model
```

Each script puts `rl/` on `sys.path` itself, so bare imports such as
`import evaluate` keep working. Outputs go to `rl/results/`; see
`rl/results/PROVENANCE.md` before re-running one, since several checkpoints
only reproduce against the simulator binary they were trained on.

| Prefix | Docs |
|---|---|
| `phase1_*` | `docs/experiments/01-tail-risk-metrics.md` |
| `phase2_*`, `phase2r_*` | `docs/experiments/02`–`04`, `docs/methodology/standing-issues.md` |
| `phase3_tail_validation_*` | `docs/experiments/05-tail-risk-validation.md` |
| `phase3_step2_*` | `docs/experiments/06-state-signal-repair.md`, `07-*` |
| `phase3_step3_*` | `docs/experiments/08-sqrt-impact-baseline.md` |
| `phase3_stepA_*` | `docs/experiments/10-full-holdout-evaluation-result.md` |
| `phase3_stepB_*` | `docs/experiments/11-imitation-warm-start-result.md` |
