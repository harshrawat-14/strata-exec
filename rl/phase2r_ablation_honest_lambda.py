"""
Phase 2R close-out gate: re-run the Task 4 ablation deliverable (RL-12dim vs
RL-16dim vs AC-Optimal vs Adaptive-AC vs TWAP, all 11 clean dates, with paired
bootstrap significance) using the HONEST lambda (1e-6, selected from the 7
TRAINING_DATES' aggregate outcome only -- see rl/phase2r_lambda_grid_full.csv
and docs/phase2r_objective_bias_finding.md, "The lambda-selection protocol was
hindsight") instead of AC_DEFAULT_LAMBDA=0.001 (the original strawman) or any
per-date hindsight-selected lambda.

This is the version of the Task 4 ablation table that should be treated as
current: the original used a single un-swept lambda (strawman); a later pass
used per-date-optimal lambda (hindsight); this uses one lambda, chosen without
ever looking at a held-out date's own outcome, applied uniformly -- the same
standard RL itself is held to.

Structure mirrors rl/phase2r_ablation_report.py exactly (same bootstrap
method, same date/regime table, same summary sections) so the two are
directly comparable side by side.

Usage: rl/.venv/bin/python rl/phase2r_ablation_honest_lambda.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from evaluate import (
    load_model,
    evaluate_all_strategies_on_date,
    HELD_OUT_CRASH_DATES,
    TRAINING_DATES,
)

MODEL_16DIM = "rl/models/ppo_ablation_16dim_cf_best/best_model"
MODEL_12DIM = "rl/models/ppo_ablation_12dim_cf_best/best_model"
N_EPISODES = 50
OUTPUT_DIR = "rl/results/"
BTC_TARGET = 50_000.0
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0

# Selected from the 7 TRAINING_DATES' aggregate mean AND median AC/AAC mean_IS
# across the full lambda grid -- robust, unchanged even excluding the
# 2024-11-06 outlier. See rl/results/phase2r_lambda_grid_full.csv.
HONEST_LAMBDA = 1e-6

DATE_REGIMES = {
    "2024-01-15": "calm bull",
    "2024-02-20": "moderate volatility",
    "2024-03-05": "crash - flash liquidation cascade",
    "2024-04-15": "BTC correction (moderate stress)",
    "2024-06-10": "quiet consolidation",
    "2024-07-10": "mid-summer sideways",
    "2024-08-05": "crash - Yen unwind",
    "2024-09-15": "moderate volatility",
    "2024-10-15": "moderate volatility",
    "2024-11-06": "post-election surge",
    "2024-12-10": "December bull run",
}

BASELINE_KEYS = [("ac", "ac_optimal"), ("aac", "adaptive_ac"), ("tw", "twap")]
BASELINE_LABELS = {"ac": "AC-Optimal", "aac": "Adaptive-AC", "tw": "TWAP"}


def _available_dates() -> list:
    dates = []
    for d in sorted(DATE_REGIMES.keys()):
        path = f"TradeData/BookDepth/BTCUSDT-bookDepth-{d}.csv"
        if os.path.exists(path):
            dates.append(d)
        else:
            print(f"  SKIPPING {d} -- LOB file not found")
    return dates


def _classify(date: str) -> str:
    if date in HELD_OUT_CRASH_DATES:
        return "held_out_crash"
    if date in TRAINING_DATES:
        return "training"
    return "held_out_calm"


def paired_bootstrap_delta(a_samples, b_samples, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    a = np.asarray(a_samples)
    b = np.asarray(b_samples)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    point = float(diff.mean())
    if n == 0 or diff.std() < 1e-12:
        return {"point": point, "ci_lower": point, "ci_upper": point, "p_value": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    resampled = diff[idx].mean(axis=1)
    ci_lower = float(np.percentile(resampled, 2.5))
    ci_upper = float(np.percentile(resampled, 97.5))
    p_value = float((resampled <= 0).mean())
    return {"point": point, "ci_lower": ci_lower, "ci_upper": ci_upper, "p_value": p_value}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 104)
    print(f"PHASE 2R HONEST-LAMBDA ABLATION: RL-12dim vs RL-16dim vs AC-Optimal vs Adaptive-AC vs TWAP")
    print(f"AC/Adaptive-AC lambda = {HONEST_LAMBDA:.0e} (selected from TRAINING_DATES only -- no")
    print(f"per-date hindsight), all 11 clean dates, post-correction simulator")
    print("=" * 104)

    print("Loading models...")
    v16 = load_model(MODEL_16DIM)
    v12 = load_model(MODEL_12DIM)

    dates = _available_dates()
    rows = []
    any_inconsistency = False

    for date in dates:
        regime = DATE_REGIMES[date]
        split = _classify(date)
        print(f"\n{'-' * 70}\n  {date}  ({regime})  [{split}]\n{'-' * 70}")

        print("  [12-dim]")
        r12 = evaluate_all_strategies_on_date(
            rl_model=v12, date=date, n_episodes=N_EPISODES, n_state_dims=12,
            mode="counterfactual", btc_target=BTC_TARGET, ac_lambda=HONEST_LAMBDA,
        )
        print("  [16-dim]")
        r16 = evaluate_all_strategies_on_date(
            rl_model=v16, date=date, n_episodes=N_EPISODES, n_state_dims=16,
            mode="counterfactual", btc_target=BTC_TARGET, ac_lambda=HONEST_LAMBDA,
        )

        rl12, rl16 = r12["rl_agent"], r16["rl_agent"]

        baselines = {}
        for short, key in BASELINE_KEYS:
            b12, b16 = r12[key], r16[key]
            consistent = abs(b12["mean_IS"] - b16["mean_IS"]) < 1e-9
            if not consistent:
                any_inconsistency = True
                print(f"  *** WARNING: {BASELINE_LABELS[short]} differs between the 12-dim and "
                      f"16-dim calls on {date} (diff={b12['mean_IS']-b16['mean_IS']:.6f}))")
            baselines[short] = b12

        row = {
            "date": date, "regime": regime, "split": split,
            "rl12_mean_IS": rl12["mean_IS"], "rl12_cvar_95": rl12["cvar_95"],
            "rl16_mean_IS": rl16["mean_IS"], "rl16_cvar_95": rl16["cvar_95"],
            "delta_16_vs_12_mean_IS": rl16["mean_IS"] - rl12["mean_IS"],
        }
        for short, _ in BASELINE_KEYS:
            b = baselines[short]
            row[f"{short}_mean_IS"] = b["mean_IS"]
            row[f"{short}_cvar_95"] = b["cvar_95"]
            for arm_label, arm in [("16", rl16), ("12", rl12)]:
                boot = paired_bootstrap_delta(arm["is_samples"], b["is_samples"])
                row[f"delta_{arm_label}_vs_{short}_mean_IS"] = arm["mean_IS"] - b["mean_IS"]
                row[f"delta_{arm_label}_vs_{short}_ci_lower"] = boot["ci_lower"]
                row[f"delta_{arm_label}_vs_{short}_ci_upper"] = boot["ci_upper"]
                row[f"delta_{arm_label}_vs_{short}_p_value"] = boot["p_value"]
                row[f"rl{arm_label}_beats_{short}"] = arm["mean_IS"] > b["mean_IS"]
        rows.append(row)

        print(f"  RL-12dim: mean_IS={rl12['mean_IS']:+.4f}%   RL-16dim: mean_IS={rl16['mean_IS']:+.4f}%")
        for short, _ in BASELINE_KEYS:
            b = baselines[short]
            print(f"  {BASELINE_LABELS[short]:12s}: mean_IS={b['mean_IS']:+.4f}%")
            for arm_label in ("16", "12"):
                d = row[f"delta_{arm_label}_vs_{short}_mean_IS"]
                p = row[f"delta_{arm_label}_vs_{short}_p_value"]
                beats = row[f"rl{arm_label}_beats_{short}"]
                pstr = f"{p:.4f}" if p is not None else "n/a"
                print(f"    RL-{arm_label}dim vs {BASELINE_LABELS[short]}: delta={d:+.4f}pp  "
                      f"p={pstr}  {'BEATS' if beats else 'loses to'}")

    if not any_inconsistency:
        print("\n*** Consistency check passed on every date. ***")

    df = pd.DataFrame(rows)
    csv_path = f"{OUTPUT_DIR}phase2r_ablation_honest_lambda.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull honest-lambda ablation table saved: {csv_path}")

    print("\n" + "=" * 104)
    print(f"SUMMARY -- RL16dim vs AC-Optimal @ honest lambda={HONEST_LAMBDA:.0e}, by split")
    print("=" * 104)
    for split_name, split_key in [
        ("HELD-OUT CRASH (n=2)", "held_out_crash"),
        ("HELD-OUT CALM (n=2)", "held_out_calm"),
        ("IN-SAMPLE TRAINING (n=7)", "training"),
    ]:
        sub = df[df["split"] == split_key]
        if len(sub) == 0:
            continue
        print(f"\n{split_name}:")
        print(sub[["date", "regime", "rl16_mean_IS", "ac_mean_IS", "aac_mean_IS", "tw_mean_IS",
                    "rl16_beats_ac", "rl16_beats_aac"]].to_string(index=False))
        for short, _ in BASELINE_KEYS:
            n16 = sub[f"rl16_beats_{short}"].sum()
            n12 = sub[f"rl12_beats_{short}"].sum()
            print(f"  RL-16dim beats {BASELINE_LABELS[short]}: {n16}/{len(sub)}   "
                  f"RL-12dim beats {BASELINE_LABELS[short]}: {n12}/{len(sub)}")

    print("\n" + "=" * 104)
    print("OVERALL RL16dim vs AC-Optimal @ honest lambda, all 11 dates:")
    print(f"  {df['rl16_beats_ac'].sum()}/11 wins   |   vs Adaptive-AC: {df['rl16_beats_aac'].sum()}/11 wins")
    print("=" * 104)


if __name__ == "__main__":
    main()
