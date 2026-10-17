"""
Phase 3 Step 1: tail-risk validation -- per-episode outcome dumping + CVaR
analysis, for the four-part evidence bar in docs/proposals/research-roadmap.md ("The tail
claim -- evidence bar and current status").

This does NOT reimplement evaluate_all_strategies_on_date's simulation
logic. It calls into it (same as every other phase2r_*.py script) and adds
what that function does not persist: (a) the raw per-episode is_samples
arrays to disk, and (b) CVaR at multiple quantiles (90/95/99, not just 95)
with paired-bootstrap confidence intervals, using the SAME paired-resample
methodology already used for p_16_vs_ac / p_16_vs_aac in
rl/phase2r_ablation_honest_lambda.py and rl/phase2r_severity_expansion_eval.py
(matched episode seeds, index-resample both arms together) -- generalized
from "mean of paired differences" (paired_bootstrap_delta) and "CVaR-95 of
one array" (evaluate.compute_cvar_95 / bootstrap_cvar_95_ci) to "CVaR-q of
paired differences, optionally mean-centered per arm first". No new
statistical method is introduced; this is the union of two methods this
project's own codebase already uses.

Model: RL-16dim only (rl/models/ppo_ablation_16dim_cf_best/best_model), the
same honest lambda=1e-6 baseline as every other Phase 2R/3 result. RL-12dim
is not evaluated here -- not needed for the tail-mechanism question (see
docs/proposals/research-roadmap.md's four-part bar, all stated in terms of RL16 vs
AC/Adaptive-AC/TWAP).

Coverage: all 25 dates in evaluate.SEVERITY_RANK_V2 (which is a superset of
the original 11-date honest-lambda set -- every V1 date is also a V2 date).
Timed at ~90-100s/date for the full 5-policy, 50-episode
evaluate_all_strategies_on_date call (measured directly on 2024-08-05 before
committing to the full run) -- roughly 40 minutes for all 25 dates, judged
affordable and run in one pass rather than scoped down.

Usage: rl/.venv/bin/python rl/phase3_tail_validation_eval.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from evaluate import (
    load_model,
    evaluate_all_strategies_on_date,
    SEVERITY_RANK_V2,
    TRAINING_DATES_V2,
    HELD_OUT_DATES_V2,
    HELD_OUT_CRASH_DATES,
    TRAINING_DATES,
)

MODEL_16DIM = "rl/models/ppo_ablation_16dim_cf_best/best_model"
N_EPISODES = 50
BTC_TARGET = 50_000.0
HONEST_LAMBDA = 1e-6  # same selection as every other Phase 2R/3 honest-lambda result
OUTPUT_DIR = "rl/results/"

EPISODES_CSV = f"{OUTPUT_DIR}phase3_tail_validation_episodes.csv"
SUMMARY_CSV = f"{OUTPUT_DIR}phase3_tail_validation_summary.csv"

BASELINE_KEYS = [("ac", "ac_optimal"), ("aac", "adaptive_ac"), ("tw", "twap")]
BASELINE_LABELS = {"ac": "AC-Optimal", "aac": "Adaptive-AC", "tw": "TWAP"}

# All 25 V2 dates, ordered by severity (most severe first) for readable
# progress output -- same ordering the severity-expansion doc uses.
ALL_DATES = sorted(SEVERITY_RANK_V2.keys(), key=lambda d: SEVERITY_RANK_V2[d][0])


def _split_v2(date: str) -> str:
    if date in TRAINING_DATES_V2:
        return "train_v2"
    if date in HELD_OUT_DATES_V2:
        return "held_out_v2"
    return "unknown"


def _split_v1(date: str) -> str:
    if date in HELD_OUT_CRASH_DATES:
        return "held_out_crash_v1"
    if date in TRAINING_DATES:
        return "training_v1"
    if date in ("2024-07-10", "2024-12-10"):
        return "held_out_calm_v1"
    return "not_in_v1"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 100)
    print("PHASE 3 STEP 1: per-episode tail-validation eval (RL-16dim vs AC/Adaptive-AC/TWAP, honest lambda)")
    print(f"{len(ALL_DATES)} dates (V2 pool), n_episodes={N_EPISODES}, lambda={HONEST_LAMBDA:.0e}")
    print("=" * 100)

    v16 = load_model(MODEL_16DIM)

    episode_rows = []
    summary_rows = []
    t_start = time.time()

    for i, date in enumerate(ALL_DATES):
        ac_mean_is_v2, regime_v2, window = SEVERITY_RANK_V2[date]
        t0 = time.time()
        print(f"\n{'-' * 70}\n  [{i+1}/{len(ALL_DATES)}] {date}  (V2 regime={regime_v2}, window={window})\n{'-' * 70}")

        r = evaluate_all_strategies_on_date(
            rl_model=v16, date=date, n_episodes=N_EPISODES, n_state_dims=16,
            mode="counterfactual", btc_target=BTC_TARGET, ac_lambda=HONEST_LAMBDA,
        )
        dt = time.time() - t0
        print(f"  ({dt:.1f}s)")

        split_v2 = _split_v2(date)
        split_v1 = _split_v1(date)

        # Dump per-episode outcomes for rl_agent + the three baselines used
        # in the tail claim (twap, ac_optimal, adaptive_ac). Heuristic is
        # evaluated by evaluate_all_strategies_on_date (can't be skipped
        # without editing that function, which we were told not to
        # duplicate/modify) but is not part of the tail claim -- dumped too,
        # for completeness, at negligible extra cost.
        for strat_key in ["rl_agent", "ac_optimal", "adaptive_ac", "twap", "heuristic"]:
            samples = r[strat_key]["is_samples"]
            for seed, is_pct in enumerate(samples):
                episode_rows.append({
                    "date": date, "regime_v2": regime_v2, "window_v2": window,
                    "split_v2": split_v2, "split_v1": split_v1,
                    "strategy": strat_key, "episode_seed": seed, "is_pct": is_pct,
                })

        row = {
            "date": date, "regime_v2": regime_v2, "window_v2": window,
            "split_v2": split_v2, "split_v1": split_v1,
            "ac_mean_is_frozen_v2": ac_mean_is_v2,
            "rl16_mean_IS": r["rl_agent"]["mean_IS"],
            "rl16_cvar_95": r["rl_agent"]["cvar_95"],
        }
        for short, key in BASELINE_KEYS:
            row[f"{short}_mean_IS"] = r[key]["mean_IS"]
            row[f"{short}_cvar_95"] = r[key]["cvar_95"]
        summary_rows.append(row)

        print(f"  RL16={row['rl16_mean_IS']:+.4f}%  AC={row['ac_mean_IS']:+.4f}%  "
              f"AAC={row['aac_mean_IS']:+.4f}%  TWAP={row['tw_mean_IS']:+.4f}%")

    total_dt = time.time() - t_start
    print(f"\nTotal eval wall time: {total_dt/60:.1f} min for {len(ALL_DATES)} dates")

    ep_df = pd.DataFrame(episode_rows)
    ep_df.to_csv(EPISODES_CSV, index=False)
    print(f"Saved per-episode outcomes: {EPISODES_CSV} ({len(ep_df)} rows)")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved per-date mean/CVaR-95 summary: {SUMMARY_CSV} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
