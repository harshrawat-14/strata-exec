"""
Phase 2R Task 4 — the actual 16-dim vs 12-dim ablation deliverable.

This is the FIRST post-correction result (see rl/results/PROVENANCE.md):
produced against the counterfactual simulator AFTER Phase 2R's Task 2 (impact
calibration), Task 3 (drift restoration), and the spread_ref scale-bug fix.
Do not compare these numbers against any pre-Phase-2R report (v5's headline
numbers, phase1_tail_risk.csv, unified_comparison.csv, failure_analysis.csv,
the old phase2_ablation.csv) -- the simulator itself changed underneath them.

Both arms trained with an IDENTICAL config (matched-conditions discipline):
same 7 training dates, same entropy floor, same CPU device, same n_envs,
same lr_schedule, same full 3M-step schedule, same resume-safe/multi-date
reward-based best-model selection -- differing ONLY in state dims (16 vs 12).

mode="counterfactual" is used for evaluation (not the historical-mode
default), per the explicit Task 4 decision: historical mode has no cf_lob,
so dims 8/9 (own-impact fractions) would be structurally dead there, testing
the 16-dim arm with two of its own distinguishing features disabled.

best_model (not _final) is used for both arms: the run logs show _final
drifted somewhat toward waiting on several dates in the 16-dim arm
specifically (entropy floor slowed but did not perfectly eliminate the
drift), while best_model (reward-selected, resume-safe, all-dates) stayed
healthy in both arms -- using best_model is precisely what that fix exists
for.

Usage: rl/.venv/bin/python rl/phase2r_ablation_report.py
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

DATE_REGIMES = {
    "2024-01-15": "calm bull",
    "2024-02-20": "moderate volatility",
    "2024-03-05": "crash - flash liquidation cascade",
    "2024-04-15": "BTC correction (moderate stress)",
    "2024-06-10": "quiet consolidation",
    "2024-08-05": "crash - Yen unwind",
    "2024-09-15": "moderate volatility",
    "2024-10-15": "moderate volatility",
    "2024-11-06": "post-election surge",
}


def _available_dates() -> list:
    dates = []
    for d in sorted(DATE_REGIMES.keys()):
        path = f"TradeData/BookDepth/BTCUSDT-bookDepth-{d}.csv"
        if os.path.exists(path):
            dates.append(d)
        else:
            print(f"  SKIPPING {d} -- LOB file not found")
    return dates


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 92)
    print("PHASE 2R TASK 4 ABLATION: 12-dim vs 16-dim, POST-CORRECTION counterfactual simulator")
    print("=" * 92)
    print(
        "\n*** PROVENANCE ***\n"
        "This is the FIRST post-correction result (calibrated impact model, restored\n"
        "drift, fixed spread_ref -- see rl/results/PROVENANCE.md). Do NOT compare against\n"
        "any pre-Phase-2R report.\n"
        "\n*** MATCHED CONDITIONS ***\n"
        "Both arms: identical config (7 training dates, entropy floor, CPU device, n_envs=2,\n"
        "linear LR, full 3M-step schedule, resume-safe/multi-date reward-based best-model\n"
        "selection) -- differ ONLY in state dims (16 vs 12). Evaluated in COUNTERFACTUAL mode\n"
        "(not historical) so dims 8/9 are live for the 16-dim arm, per the Task 4 gate-3\n"
        "decision. best_model used for both arms (see module docstring).\n"
    )

    print("Loading models...")
    v16 = load_model(MODEL_16DIM)
    v12 = load_model(MODEL_12DIM)

    dates = _available_dates()
    rows = []

    for date in dates:
        regime = DATE_REGIMES[date]
        is_crash_holdout = date in HELD_OUT_CRASH_DATES
        is_training_date = date in TRAINING_DATES
        print(f"\n{'-' * 70}\n  {date}  ({regime})"
              f"  [{'HELD-OUT crash' if is_crash_holdout else ('training' if is_training_date else 'other')}]"
              f"\n{'-' * 70}")

        print("  [12-dim]")
        r12 = evaluate_all_strategies_on_date(
            rl_model=v12, date=date, n_episodes=N_EPISODES, n_state_dims=12,
            mode="counterfactual", btc_target=BTC_TARGET,
        )
        print("  [16-dim]")
        r16 = evaluate_all_strategies_on_date(
            rl_model=v16, date=date, n_episodes=N_EPISODES, n_state_dims=16,
            mode="counterfactual", btc_target=BTC_TARGET,
        )

        rl12 = r12["rl_agent"]
        rl16 = r16["rl_agent"]

        rows.append({
            "date": date,
            "regime": regime,
            "held_out_crash_date": is_crash_holdout,
            "training_date": is_training_date,
            "v12_mean_IS": rl12["mean_IS"],
            "v16_mean_IS": rl16["mean_IS"],
            "delta_mean_IS": rl16["mean_IS"] - rl12["mean_IS"],
            "v12_cvar_95": rl12["cvar_95"],
            "v16_cvar_95": rl16["cvar_95"],
            "delta_cvar_95": rl16["cvar_95"] - rl12["cvar_95"],
            "v12_cvar_95_ci": f"[{rl12['cvar_95_ci_lower']:.4f}, {rl12['cvar_95_ci_upper']:.4f}]",
            "v16_cvar_95_ci": f"[{rl16['cvar_95_ci_lower']:.4f}, {rl16['cvar_95_ci_upper']:.4f}]",
            "v12_forced_liq": rl12["forced_liquidation_rate"],
            "v16_forced_liq": rl16["forced_liquidation_rate"],
            "twap_mean_IS": r12["twap"]["mean_IS"],
            "ac_optimal_mean_IS": r12["ac_optimal"]["mean_IS"],
        })

        print(f"  12-dim: mean_IS={rl12['mean_IS']:.4f}%  cvar_95={rl12['cvar_95']:.4f}%  "
              f"forced_liq={rl12['forced_liquidation_rate']:.2f}")
        print(f"  16-dim: mean_IS={rl16['mean_IS']:.4f}%  cvar_95={rl16['cvar_95']:.4f}%  "
              f"forced_liq={rl16['forced_liquidation_rate']:.2f}")
        print(f"  delta:  mean_IS={rl16['mean_IS']-rl12['mean_IS']:+.4f}pp  "
              f"cvar_95={rl16['cvar_95']-rl12['cvar_95']:+.4f}pp")

    df = pd.DataFrame(rows)
    csv_path = f"{OUTPUT_DIR}phase2r_ablation_postcorrection.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull ablation table saved: {csv_path}")

    print("\n" + "=" * 92)
    print("SUMMARY -- mean across all dates")
    print("=" * 92)
    print(f"12-dim mean_IS avg:  {df['v12_mean_IS'].mean():.4f}%")
    print(f"16-dim mean_IS avg:  {df['v16_mean_IS'].mean():.4f}%")
    print(f"delta avg:           {df['delta_mean_IS'].mean():+.4f}pp")
    print()
    print(f"12-dim cvar_95 avg:  {df['v12_cvar_95'].mean():.4f}%")
    print(f"16-dim cvar_95 avg:  {df['v16_cvar_95'].mean():.4f}%")
    print(f"delta avg:           {df['delta_cvar_95'].mean():+.4f}pp")

    print("\n" + "=" * 92)
    print("SUMMARY -- held-out crash dates only (2024-08-05, 2024-03-05)")
    print("Both arms trained on the IDENTICAL corrected split -- these are genuinely")
    print("out-of-sample for BOTH arms, a fair apples-to-apples generalization check.")
    print("=" * 92)
    crash_df = df[df["held_out_crash_date"]]
    print(crash_df[["date", "regime", "v12_mean_IS", "v16_mean_IS", "delta_mean_IS",
                    "v12_cvar_95", "v16_cvar_95", "delta_cvar_95",
                    "v12_forced_liq", "v16_forced_liq"]].to_string(index=False))

    print("\n" + "=" * 92)
    print("forced_liquidation_rate by arm (0.0 = agent always self-liquidated; watch for")
    print("either arm re-collapsing to blanket-wait, which would invalidate its numbers)")
    print("=" * 92)
    print(df[["date", "v12_forced_liq", "v16_forced_liq"]].to_string(index=False))


if __name__ == "__main__":
    main()
