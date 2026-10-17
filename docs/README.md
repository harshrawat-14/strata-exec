# StrataExec Documentation

Documentation is organised by **purpose**, not by the chronological phase it was produced in.
Phase identifiers (Phase 1, 2R, 3, Step A/B, etc.) are retained *inside* each document so provenance
and supersession chains stay traceable — but you shouldn't need to know the project's internal phase
numbering to find what you're looking for.

## Start here

| If you want to… | Read |
|---|---|
| Understand what the system is and how it fits together | [architecture/high-level-design.md](architecture/high-level-design.md) |
| Know why a design choice was made | [architecture/decision-records.md](architecture/decision-records.md) |
| See the current headline research result | [experiments/10-full-holdout-evaluation-result.md](experiments/10-full-holdout-evaluation-result.md) |
| Know what's still unresolved or contested | [methodology/standing-issues.md](methodology/standing-issues.md) |
| See what went wrong before, and why | [history/timeline.md](history/timeline.md) · [lessons.md](history/lessons.md) |
| See what's planned next | [proposals/research-roadmap.md](proposals/research-roadmap.md) |

## `architecture/` — how the system is built

| Document | Contents |
|---|---|
| [high-level-design.md](architecture/high-level-design.md) | System decomposition, Rust/Python boundary, data flow |
| [low-level-design.md](architecture/low-level-design.md) | Module-level detail, key types, invariants |
| [decision-records.md](architecture/decision-records.md) | ADRs — numbered architectural decisions and their rationale |

## `performance/` — throughput and systems engineering

Profiling and optimisation of the simulation/training loop (internally: Phase S).

| Document | Contents |
|---|---|
| [baseline-profiling.md](performance/baseline-profiling.md) | Where time actually went before optimisation — the bottleneck was `reset()`'s CSV re-parse, not the transport |
| [regression-verification.md](performance/regression-verification.md) | Proof the optimisation didn't change behaviour, under both harness and real training config |
| `baseline_benchmark.py`, `regression_check.py` | The runnable benchmark and regression scripts |

## `experiments/` — research results, chronological

Numbered in the order they were run. **Later documents supersede earlier ones**; several early
findings were retracted and the retractions are preserved deliberately rather than deleted.

| # | Document | Finding |
|---|---|---|
| 01 | [tail-risk-metrics.md](experiments/01-tail-risk-metrics.md) | CVaR/tail-risk metric definitions and first measurements |
| 02 | [reward-objective-audit.md](experiments/02-reward-objective-audit.md) | **Large, contains a major retraction.** Reward-hacking discovery; the "evaluate on the training objective, not a reporting metric" lesson |
| 03 | [ac-baseline-calibration.md](experiments/03-ac-baseline-calibration.md) | How the Almgren-Chriss baseline's parameters are chosen honestly |
| 04 | [crash-severity-expansion.md](experiments/04-crash-severity-expansion.md) | Dataset expansion 11 → 25 dates; severity-scaling pattern replaces the coarse event-window one |
| 05 | [tail-risk-validation.md](experiments/05-tail-risk-validation.md) | CVaR edge scales with severity but does not survive mean-centering |
| 06 | [state-signal-repair.md](experiments/06-state-signal-repair.md) | Retrain on repaired state — negative result, root-caused to a training-data bug |
| 07 | [full-date-coverage-retrain.md](experiments/07-full-date-coverage-retrain.md) | Full 16-date coverage restores the crash advantage |
| 08 | [sqrt-impact-baseline.md](experiments/08-sqrt-impact-baseline.md) | A numerically-optimised baseline proves AC was not a strawman on impact-model shape |
| 09 | [policy-capacity-and-entropy.md](experiments/09-policy-capacity-and-entropy.md) | Underfit policy fixed; the pacing dissociation survives, and strengthens |
| 10 | [full-holdout-evaluation-predictions.md](experiments/10-full-holdout-evaluation-predictions.md) · [result](experiments/10-full-holdout-evaluation-result.md) | **Current headline.** All 9 held-out dates: 4/4 crash-tier wins, 0/5 calm/normal |
| 11 | [imitation-warm-start-predictions.md](experiments/11-imitation-warm-start-predictions.md) · [result](experiments/11-imitation-warm-start-result.md) | **Negative result.** Cloning AC gives the calm-day floor but destroys the crash-day edge |
| 12 | [forced-liquidation-at-data-exhaustion.md](experiments/12-forced-liquidation-at-data-exhaustion.md) | Simulator fix. AC/RL unaffected; Adaptive-AC flattered (up to 0.46 pp held-out); sqrt-impact flattered on training dates only, Step 3 stands |

Predictions files are **pre-registered** — written and committed before the corresponding experiment
was run, so the result cannot be retrofitted to the outcome.

## `methodology/` — how the research is kept honest

| Document | Contents |
|---|---|
| [standing-issues.md](methodology/standing-issues.md) | Living register of known limitations, open questions, and contested claims |
| [platform-audit.md](methodology/platform-audit.md) | Adversarial audit of the evaluation platform; numbered findings, several still open |

## `proposals/` — planned and deferred work

| Document | Contents |
|---|---|
| [research-roadmap.md](proposals/research-roadmap.md) | Current plan, sequencing, and explicitly deferred items |
| [residual-action-space.md](proposals/residual-action-space.md) | Design sketch for a learned-residual action space (not implemented) |

## `maintenance/` — repository housekeeping

| Document | Contents |
|---|---|
| [disk-cleanup-2026-09-07.md](maintenance/disk-cleanup-2026-09-07.md) | What was deleted, why, and how to regenerate it |

## `history/` — mistakes, corrections and decisions

| Document | Contents |
|---|---|
| [timeline.md](history/timeline.md) | Every bug, retracted claim, methodology error and key decision, in date order |
| [lessons.md](history/lessons.md) | The ten recurring patterns behind them, and the check that now guards each |

---

## Conventions

- **Supersession is explicit.** When a result is overturned, the original document stays and says so.
  Nothing is silently rewritten to look correct in hindsight.
- **Retractions are labelled as retractions**, not softened to "unresolved."
- **Pre-registration** is used for experiments where the outcome could otherwise be rationalised
  after the fact.
- **Phase identifiers live inside documents**, not in filenames or folder names.
