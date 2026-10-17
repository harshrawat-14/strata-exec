"""
Phase 3 Step B, part 2: verify the BC-pretrained checkpoint's floor guarantee
BEFORE any RL fine-tuning. If the behavior-cloned policy doesn't reproduce
AC-Optimal's performance on held-out dates, fine-tuning from it is building on
a broken base -- stop and fix the clone, don't proceed to RL fine-tuning.

Same harness as Step A (rl/experiments/phase3_stepA_full_holdout_eval.py) so numbers are
directly comparable to the existing AC-Optimal column: same 9 HELD_OUT_DATES_V2,
same honest lambda, same n_episodes.

Usage:
    rl/.venv/bin/python rl/experiments/phase3_stepB_verify_floor.py \
        --model rl/models/phase3_stepB_bc_pretrain/bc_pretrained
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from evaluate import (
    load_model,
    evaluate_all_strategies_on_date,
    query_calibration,
    HELD_OUT_DATES_V2,
    SEVERITY_RANK_V2,
    TRAINING_DATES_V2,
)

HONEST_LAMBDA = 1e-6
BTC_TARGET = 50_000.0
N_STATE_DIMS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-episodes", type=int, default=20)
    args = ap.parse_args()

    assert not (set(TRAINING_DATES_V2) & set(HELD_OUT_DATES_V2))

    model = load_model(args.model)

    rows = []
    for date in HELD_OUT_DATES_V2:
        regime = SEVERITY_RANK_V2[date][1]
        calib = query_calibration(date, btc_target=BTC_TARGET)
        results = evaluate_all_strategies_on_date(
            rl_model=model,
            date=date,
            n_episodes=args.n_episodes,
            n_state_dims=N_STATE_DIMS,
            mode="counterfactual",
            btc_target=BTC_TARGET,
            ac_lambda=HONEST_LAMBDA,
            include_sqrt_impact_optimal=False,
        )
        for strategy_key, r in results.items():
            rows.append({"date": date, "regime": regime, "strategy": r["strategy"],
                         "mean_IS": r["mean_IS"]})
        print(f"  done {date} ({regime})")

    df = pd.DataFrame(rows)
    df.to_csv("rl/results/phase3_stepB_floor_check.csv", index=False)

    pivot = df.pivot(index=["date", "regime"], columns="strategy", values="mean_IS")
    print("\nmean_IS (BC-pretrained model, no RL fine-tuning yet):")
    print(pivot.round(4).to_string())

    if "RL Agent" in df.strategy.unique() and "AC Optimal" in df.strategy.unique():
        rl = df[df.strategy == "RL Agent"].set_index("date")["mean_IS"]
        ac = df[df.strategy == "AC Optimal"].set_index("date")["mean_IS"]
        delta = (rl - ac).sort_values()
        print("\nBC-clone - AC-Optimal (should be near 0 if the clone worked; "
              "large negative = clone failed, do NOT fine-tune from this checkpoint):")
        for date, d in delta.items():
            print(f"    {date} ({SEVERITY_RANK_V2[date][1]:6}) {d:+.4f}")
        mean_gap = delta.abs().mean()
        print(f"\nmean |gap| = {mean_gap:.4f} pp")
        if mean_gap > 1.0:
            print("FLOOR CHECK FAILED: clone is not close to AC. Do not proceed to "
                  "RL fine-tuning from this checkpoint without investigating why.")
        else:
            print("Floor check passed (gap small): reasonable to proceed to RL fine-tuning.")


if __name__ == "__main__":
    main()
