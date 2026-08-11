"""
Phase 2 ablation: 12-dim checkpoint (no time features) vs 16-dim checkpoint
(time features s12-s15 included), same eval harness (evaluate_all_strategies_on_date),
same dates, same episode count. This is the deliverable Phase 2's acceptance
criteria call for.

Both models are run through the identical counterfactual-LOB harness; only
the "rl_agent" row differs between the two calls (TWAP/Heuristic/AC Optimal/
Adaptive AC don't depend on the RL checkpoint, so they're identical between
the two runs and only kept for context, not re-analyzed here).

HONESTY NOTES (both must survive into the report, not just this docstring):
1. v5 (12-dim) was trained under the OLD, uncorrected date split -- every
   date evaluated below, including both crash dates, was in its training
   data. v6 (16-dim) was trained under the CORRECTED split
   (TRAINING_DATES / HELD_OUT_CRASH_DATES) -- for v6, 2024-08-05 and
   2024-03-05 are genuinely out-of-sample. This is an asymmetry the ablation
   table must label explicitly per row, not gloss over: a "v6 does better/
   worse on the crash days" finding partly reflects this split difference,
   not purely the 12-vs-16-dim state representation.
2. v5's exact original hyperparameters predate mlflow logging and are not
   recoverable. v6 was trained with the codebase's current default
   hyperparameters as the best-available "same code" proxy -- not a
   provably byte-identical replication of whatever produced v5. Any
   observed delta could be partly attributable to this, not purely the
   state-vector change.

Usage: rl/.venv/bin/python rl/phase2_ablation_report.py
"""
from __future__ import annotations

import os

import pandas as pd

from evaluate import (
    load_model,
    evaluate_all_strategies_on_date,
    HELD_OUT_CRASH_DATES,
    TEST_DATES,
)

V5_MODEL_PATH = "rl/models/ppo_lstm_v5_adaptive_best/best_model"
V6_MODEL_PATH = "rl/models/ppo_lstm_v6_time_features_best/best_model"
N_EPISODES = 50
OUTPUT_DIR = "rl/results/"

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

    print("=" * 90)
    print("PHASE 2 ABLATION: 12-dim (no time features) vs 16-dim (time features s12-s15)")
    print("=" * 90)
    print(
        "\n*** HONESTY NOTES ***\n"
        "1. v5 (12-dim) trained on the OLD split -- every date below was in its training\n"
        "   data, crash days included. v6 (16-dim) trained on the CORRECTED split\n"
        "   (TRAINING_DATES/HELD_OUT_CRASH_DATES) -- 2024-08-05 and 2024-03-05 are\n"
        "   genuinely out-of-sample FOR V6 ONLY. Labeled per-row below.\n"
        "2. v5's exact original hyperparameters predate mlflow logging and are not\n"
        "   recoverable; v6 uses the codebase's current default hyperparameters as the\n"
        "   best-available 'same code' proxy, not a provable byte-identical replication.\n"
        "   Any delta below could be partly attributable to this, not purely the\n"
        "   12-vs-16-dim state representation change.\n"
    )

    print("Loading models...")
    v5_model = load_model(V5_MODEL_PATH)
    v6_model = load_model(V6_MODEL_PATH)

    dates = _available_dates()
    rows = []

    for date in dates:
        regime = DATE_REGIMES[date]
        print(f"\n{'-' * 70}\n  {date}  ({regime})\n{'-' * 70}")

        print("  [v5, 12-dim]")
        v5_results = evaluate_all_strategies_on_date(
            rl_model=v5_model, date=date, n_episodes=N_EPISODES, n_state_dims=12,
        )
        print("  [v6, 16-dim]")
        v6_results = evaluate_all_strategies_on_date(
            rl_model=v6_model, date=date, n_episodes=N_EPISODES, n_state_dims=16,
        )

        v5_rl = v5_results["rl_agent"]
        v6_rl = v6_results["rl_agent"]

        # v5: every date was in its (old, uncorrected) training set.
        # v6: only HELD_OUT_CRASH_DATES are genuinely out-of-sample.
        v5_in_sample = date not in TEST_DATES  # old convention: only the 3 TEST_DATES were out-of-sample
        v6_in_sample = date not in HELD_OUT_CRASH_DATES

        rows.append({
            "date": date,
            "regime": regime,
            "v5_in_sample": v5_in_sample,
            "v6_in_sample": v6_in_sample,
            "v5_mean_IS": v5_rl["mean_IS"],
            "v6_mean_IS": v6_rl["mean_IS"],
            "delta_mean_IS": v6_rl["mean_IS"] - v5_rl["mean_IS"],
            "v5_cvar_95": v5_rl["cvar_95"],
            "v6_cvar_95": v6_rl["cvar_95"],
            "delta_cvar_95": v6_rl["cvar_95"] - v5_rl["cvar_95"],
            "v5_cvar_95_ci": f"[{v5_rl['cvar_95_ci_lower']:.4f}, {v5_rl['cvar_95_ci_upper']:.4f}]",
            "v6_cvar_95_ci": f"[{v6_rl['cvar_95_ci_lower']:.4f}, {v6_rl['cvar_95_ci_upper']:.4f}]",
            "v5_forced_liq": v5_rl["forced_liquidation_rate"],
            "v6_forced_liq": v6_rl["forced_liquidation_rate"],
            # Classical baselines (identical between v5/v6 runs, same seeds/episodes) --
            # kept for context, not re-analyzed as part of the ablation itself.
            "twap_mean_IS": v5_results["twap"]["mean_IS"],
            "ac_optimal_mean_IS": v5_results["ac_optimal"]["mean_IS"],
        })

        print(f"  v5 (12-dim): mean_IS={v5_rl['mean_IS']:.4f}%  cvar_95={v5_rl['cvar_95']:.4f}%")
        print(f"  v6 (16-dim): mean_IS={v6_rl['mean_IS']:.4f}%  cvar_95={v6_rl['cvar_95']:.4f}%")
        print(f"  delta:       mean_IS={v6_rl['mean_IS']-v5_rl['mean_IS']:+.4f}pp  "
              f"cvar_95={v6_rl['cvar_95']-v5_rl['cvar_95']:+.4f}pp")

    df = pd.DataFrame(rows)
    csv_path = f"{OUTPUT_DIR}phase2_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull ablation table saved: {csv_path}")

    print("\n" + "=" * 90)
    print("SUMMARY -- mean across all dates")
    print("=" * 90)
    print(f"v5 (12-dim) mean_IS avg:  {df['v5_mean_IS'].mean():.4f}%")
    print(f"v6 (16-dim) mean_IS avg:  {df['v6_mean_IS'].mean():.4f}%")
    print(f"delta avg:                {df['delta_mean_IS'].mean():+.4f}pp")
    print()
    print(f"v5 (12-dim) cvar_95 avg:  {df['v5_cvar_95'].mean():.4f}%")
    print(f"v6 (16-dim) cvar_95 avg:  {df['v6_cvar_95'].mean():.4f}%")
    print(f"delta avg:                {df['delta_cvar_95'].mean():+.4f}pp")

    print("\n" + "=" * 90)
    print("SUMMARY -- v6-out-of-sample dates only (2024-08-05, 2024-03-05)")
    print("These are the ONLY rows where v6 is evaluated on data it never trained on.")
    print("v5's numbers on these same dates are in-sample (old split) -- NOT a fair")
    print("apples-to-apples generalization comparison, shown for reference only.")
    print("=" * 90)
    oos_df = df[~df["v6_in_sample"]]
    print(oos_df[["date", "regime", "v5_mean_IS", "v6_mean_IS", "delta_mean_IS",
                   "v5_cvar_95", "v6_cvar_95", "delta_cvar_95"]].to_string(index=False))


if __name__ == "__main__":
    main()
