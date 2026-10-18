"""
M0 under protocol v3: validation-date comparison (docs/experiments/13-m0-protocol-v3-*).

Models, both evaluated under v3 (1,920-step episodes, 20 evenly spaced starts):
  M0      -- the Step 4 recipe retrained under v3 (fixed seed stream, 12 training
             dates, selection on the validation dates)
  Step 4  -- the legacy headline checkpoint, trained on the replayed-seed bug.
             Diagnostic only: M0 vs Step 4 shows what fixing the training data
             changes, holding the eval protocol fixed.
Baselines at their v3 honest lambdas (evaluation/protocol.py).

VALIDATION_DATES_V3 only. Held-out dates are not touched.

    python rl/experiments/m0_v3_validation_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import pandas as pd

from evaluation.dates import SEVERITY_RANK_V2, VALIDATION_DATES_V3
from evaluation.episode import load_model
from evaluation.harness import evaluate_all_strategies_on_date
from evaluation.protocol import AC_LAMBDA, ADAPTIVE_AC_LAMBDA, EPISODE_STEPS, N_EVAL_EPISODES

MODELS = {
    "M0": "rl/models/m0_v3_best/best_model",
    "Step 4 (legacy-trained)": "rl/models/phase3_step4_capacity_entropy_best/best_model",
}
OUT = "rl/results/m0_v3_validation.csv"


def main():
    rows = []
    for model_name, path in MODELS.items():
        model = load_model(path)
        for date in VALIDATION_DATES_V3:
            print(f"\n=== {model_name} | {date} ({SEVERITY_RANK_V2[date][1]}) ===")
            res = evaluate_all_strategies_on_date(
                rl_model=model, date=date, n_episodes=N_EVAL_EPISODES, n_state_dims=16,
                mode="counterfactual", btc_target=50_000.0,
                ac_lambda=AC_LAMBDA, adaptive_ac_lambda=ADAPTIVE_AC_LAMBDA,
                episode_steps=EPISODE_STEPS,
                include_sqrt_impact_optimal=(model_name == "M0"),
            )
            for r in res.values():
                strategy = r["strategy"]
                if strategy == "RL Agent":
                    strategy = model_name
                elif model_name != "M0":
                    continue  # baselines are identical across models; keep one copy
                rows.append(dict(date=date, regime=SEVERITY_RANK_V2[date][1], strategy=strategy,
                                 mean_IS=r["mean_IS"], std_IS=r["std_IS"], cvar_95=r["cvar_95"],
                                 forced_liquidation_rate=r["forced_liquidation_rate"],
                                 n_episodes=r["n_episodes"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    piv = df.pivot(index=["date", "regime"], columns="strategy", values="mean_IS")
    print("\nmean_IS, validation dates, protocol v3:")
    print(piv.round(4).to_string())
    for m in MODELS:
        print(f"\n{m} - AC Optimal (positive = RL better):")
        print((piv[m] - piv["AC Optimal"]).round(4).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
