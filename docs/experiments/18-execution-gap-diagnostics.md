# 18 — Why does RL execute worse than AC? Three diagnostics

Written 2026-10-22. No model trained. TRAINING_DATES_V3 only, protocol v3, 5 evenly spaced
starts per date (`rl/experiments/execution_gap_diagnostics.py`; CSVs
`rl/results/execution_gap_diagnostics_*`). Validation and held-out dates untouched.

## A. Policies side by side (means over 12 dates × 5 starts)

| policy | IS % | execution % | reward | avg sell time | forced dump | action switches / episode |
|---|---:|---:|---:|---:|---:|---:|
| AC (closed-loop) | **−0.33** | −0.46 | −0.0047 | 0.45 | 0% | **837** |
| always sell 0.3% of remaining | −0.43 | −0.56 | −0.0057 | 0.17 | 0.3% | 0 |
| M0 | −0.54 | −0.64 | −0.0064 | 0.29 | 0% | 83 |
| always sell 0.5% of remaining | −0.76 | −0.77 | −0.0077 | 0.10 | 0% | 0 |
| always sell 0.1% of remaining | −0.85 | **−0.39** | −0.0111 | 0.35 | 15% | 0 |
| always sell 0.8% of remaining | −1.10 | −1.10 | −0.0110 | 0.07 | 0% | 0 |

- A constant one-action rule beats M0 on IS and on M0's own training reward.
- AC's pacing uses actions 1/2/0/3 (0.1/0.3/0/0.5% of remaining) and switches **837 times
  in a 1,920-step episode**: the action table is coarse exactly where AC operates, so
  AC-quality pacing needs very fine, high-frequency control.

## B. Value function — candidate not supported (and a flaw in this diagnostic)

The diagnostic compared M0's critic V(s) with returns of the *decoded* deterministic
policy and found corr 0.13. That comparison is flawed: the critic estimates value under the
*sampling* policy it was trained with. PPO's own training metric is the right test, and it
shows a good critic: **explained variance 0.70–0.93** throughout M0's training. Candidate 1
(the agent misjudges holding cost) is not supported.

## C. Per-trade cost is strongly convex in size

| fill size (share of order) | ≤ 0.05% | 0.05–0.1% | 0.1–0.3% | **0.3–1%** |
|---|---:|---:|---:|---:|
| cost per unit (bp below mid) | 46 | 46 | 57 | **124** |
| fills | 300,229 | 161,935 | 56,329 | 15,627 |

AC never trades above 0.3% of the order; M0 does on 3.2% of its fills, at ~2.7× the
per-unit cost. This cost is immediate and therefore visible in the reward.

## Reading (hypothesis consistent with A and C; not yet tested)

During training the policy samples actions, including occasional large ones (3–5% of
what is left), which become the expensive >0.3% fills when much inventory remains; while
AC-quality pacing requires precise high-frequency alternation among the smallest buckets.
The action table makes the good behaviour hard to express and the costly mistakes easy.
Proposed next: a **rate-based action space** (sell k × order/horizon per step), where
steady pacing needs no switching and the largest trade is bounded by the order size.
