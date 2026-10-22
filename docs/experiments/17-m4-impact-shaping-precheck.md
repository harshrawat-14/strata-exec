# 17 — M4 (impact-potential shaping): stopped by its pre-check

Written 2026-10-22. No model was trained.

## The idea

After M0–M3, the working explanation was: a trade's main cost lands on later fills,
through impact that decays with a ~400-step half-life (30% permanent), beyond PPO's
~17-step credit reach — so RL cannot see what its own impact costs. M4 would charge that
cost immediately with potential-based shaping (policy-invariant; Ng et al. 1999):

    Phi(s) = -k * q * (P + T * D(n))      (training/shaping.py)

## The pre-check (rule fixed before running)

`rl/experiments/m4_fit_impact_potential.py`, TRAINING_DATES_V3 only: behaviour policies AC,
AC run ahead by 0.1 and 0.3, and M0; 5 starts per date; at every 10th step, impact
exposure x = q·(P + T·D(n)) and realised cost-to-go G (sum of later env rewards). Fit
`G = c0 + a·q + b·x`. **Use k = −b only if b < 0 and |t| > 5; otherwise stop.**

## Result (`rl/results/m4_impact_potential_fit.json`)

| | value |
|---|---|
| observations | 36,449 |
| b (cost per unit impact exposure) | −1.28e−5, **t = −0.27** |
| a (cost per unit remaining inventory) | −0.0054 |
| R² | 0.79 |
| mean impact exposure | 0.38 (99th pct 0.87) |

**Rejected.** Given the remaining inventory, the impact already in the book does not
measurably predict the cost still to come. Remaining inventory alone explains most of it.

## What this changes

The "RL cannot see its own delayed impact cost" explanation offered after M3 is **not
supported by the data** and is withdrawn. RL's execution gap to AC (M0: 0.13–0.24 pp on
every validation date) remains real but unexplained. M4 is not trained; `training/shaping.py`
is kept only because the pre-check uses its exposure function.

Candidate explanations still open, to test before the next training run:
- the agent's **value function** misjudges the cost of holding inventory (checkable by
  comparing M0's value estimates with realised cost-to-go);
- the **action granularity** near the small fractions AC uses makes AC-like pacing hard to
  express without chatter;
- the per-trade cost is **convex in trade size**, so the large trades RL makes are
  expensive immediately — a cost RL does see, which would point to an optimisation
  problem rather than a visibility one.
