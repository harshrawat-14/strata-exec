"""
Phase 2R severity expansion: evaluate the 14 new event-window 2024 dates
against RL-12dim / RL-16dim, at the SAME honest lambda=1e-6 used for the
original 11-date ablation (rl/phase2r_ablation_honest_lambda.py) -- selected
from TRAINING_DATES only, never from any held-out date's own outcome,
including these new ones.

Native 2024 Binance pipeline -- zero recalibration risk, none of the
Tardis/LUNA/FTX confounds. This is the gate: if the honest-lambda
regime-conditional pattern (RL wins severe stress, loses calm/moderate)
holds or strengthens across these 14 new days, LUNA/FTX becomes worth
considering. If it washes out, LUNA/FTX is moot.

Guardrails (explicit, both enforced in the reporting below):
  1. Every day in every window is reported, not just the sharp ones.
  2. 2024-04-15, 2024-07-10, 2024-08-05 (already in the original 11-date set)
     are the pre-existing tail/member days of windows A/B/C respectively --
     their honest-lambda results are pulled from the existing ablation CSV,
     not re-evaluated, and are reported as part of their window's episode,
     not as a separate/new observation.

Usage: rl/.venv/bin/python rl/phase2r_severity_expansion_eval.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from evaluate import load_model, evaluate_all_strategies_on_date

MODEL_16DIM = "rl/models/ppo_ablation_16dim_cf_best/best_model"
MODEL_12DIM = "rl/models/ppo_ablation_12dim_cf_best/best_model"
N_EPISODES = 50
BTC_TARGET = 50_000.0
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0
HONEST_LAMBDA = 1e-6  # same selection as the original 11-date honest-lambda ablation

OUTPUT_DIR = "rl/results/"

# Window membership, in event order. Days marked "existing" are already in
# rl/results/phase2r_ablation_honest_lambda.csv -- pulled from there, not
# re-run, per the no-double-counting guardrall.
WINDOWS = {
    "A": {
        "label": "Iran-Israel escalation",
        "days": {
            "2024-04-12": "new", "2024-04-13": "new", "2024-04-14": "new",
            "2024-04-15": "existing",
        },
    },
    "B": {
        "label": "German govt BTC sales / Mt. Gox fear",
        "days": {
            "2024-07-04": "new", "2024-07-05": "new", "2024-07-06": "new",
            "2024-07-07": "new", "2024-07-08": "new", "2024-07-09": "new",
            "2024-07-10": "existing",
            "2024-07-11": "new", "2024-07-12": "new",
        },
    },
    "C": {
        "label": "Jobs-report fear -> Aug 5 culmination",
        "days": {
            "2024-08-02": "new", "2024-08-03": "new", "2024-08-04": "new",
            "2024-08-05": "existing",
        },
    },
}

NEW_DATES = [d for w in WINDOWS.values() for d, status in w["days"].items() if status == "new"]


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


def evaluate_new_date(v12, v16, date: str) -> dict:
    print(f"\n{'-' * 70}\n  {date}\n{'-' * 70}")
    r12 = evaluate_all_strategies_on_date(
        rl_model=v12, date=date, n_episodes=N_EPISODES, n_state_dims=12,
        mode="counterfactual", btc_target=BTC_TARGET, ac_lambda=HONEST_LAMBDA,
    )
    r16 = evaluate_all_strategies_on_date(
        rl_model=v16, date=date, n_episodes=N_EPISODES, n_state_dims=16,
        mode="counterfactual", btc_target=BTC_TARGET, ac_lambda=HONEST_LAMBDA,
    )
    rl12, rl16 = r12["rl_agent"], r16["rl_agent"]
    ac, aac, tw = r16["ac_optimal"], r16["adaptive_ac"], r16["twap"]

    row = {
        "date": date,
        "rl12_mean_IS": rl12["mean_IS"], "rl16_mean_IS": rl16["mean_IS"],
        "ac_mean_IS": ac["mean_IS"], "aac_mean_IS": aac["mean_IS"], "tw_mean_IS": tw["mean_IS"],
    }
    for arm_label, arm in [("16", rl16), ("12", rl12)]:
        for short, base in [("ac", ac), ("aac", aac), ("tw", tw)]:
            boot = paired_bootstrap_delta(arm["is_samples"], base["is_samples"])
            row[f"delta_{arm_label}_vs_{short}"] = arm["mean_IS"] - base["mean_IS"]
            row[f"p_{arm_label}_vs_{short}"] = boot["p_value"]
            row[f"rl{arm_label}_beats_{short}"] = arm["mean_IS"] > base["mean_IS"]

    print(f"  RL-12dim: {rl12['mean_IS']:+.4f}%   RL-16dim: {rl16['mean_IS']:+.4f}%   "
          f"AC(honest): {ac['mean_IS']:+.4f}%   AAC(honest): {aac['mean_IS']:+.4f}%   "
          f"TWAP: {tw['mean_IS']:+.4f}%")
    p16ac = row['p_16_vs_ac']
    print(f"  RL16 vs AC: delta={row['delta_16_vs_ac']:+.4f}pp p={p16ac if p16ac is None else f'{p16ac:.4f}'} "
          f"{'BEATS' if row['rl16_beats_ac'] else 'loses to'}")
    return row


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading models...")
    v16 = load_model(MODEL_16DIM)
    v12 = load_model(MODEL_12DIM)

    print(f"\nEvaluating {len(NEW_DATES)} new event-window dates at honest lambda={HONEST_LAMBDA:.0e}")
    new_rows = {}
    for date in NEW_DATES:
        new_rows[date] = evaluate_new_date(v12, v16, date)

    new_df = pd.DataFrame(list(new_rows.values()))
    new_df.to_csv(f"{OUTPUT_DIR}phase2r_severity_expansion_new_days.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR}phase2r_severity_expansion_new_days.csv")

    # Pull existing-day results (04-15, 07-10, 08-05) from the prior honest-lambda ablation.
    existing = pd.read_csv(f"{OUTPUT_DIR}phase2r_ablation_honest_lambda.csv").set_index("date")

    def existing_row(date):
        e = existing.loc[date]
        row = {
            "date": date,
            "rl12_mean_IS": e["rl12_mean_IS"], "rl16_mean_IS": e["rl16_mean_IS"],
            "ac_mean_IS": e["ac_mean_IS"], "aac_mean_IS": e["aac_mean_IS"], "tw_mean_IS": e["tw_mean_IS"],
        }
        for arm_label in ("16", "12"):
            for short in ("ac", "aac", "tw"):
                row[f"delta_{arm_label}_vs_{short}"] = e[f"delta_{arm_label}_vs_{short}_mean_IS"]
                row[f"p_{arm_label}_vs_{short}"] = e[f"delta_{arm_label}_vs_{short}_p_value"]
                row[f"rl{arm_label}_beats_{short}"] = bool(e[f"rl{arm_label}_beats_{short}"])
        return row

    print(f"\n{'=' * 104}\nFULL PER-WINDOW REPORT (guardrails enforced: every day shown; existing episode\n"
          f"members pulled from the prior ablation, not re-run or double-counted)\n{'=' * 104}")

    all_rows = []
    for wname, w in WINDOWS.items():
        print(f"\n### Window {wname} -- {w['label']} ###")
        print(f"{'date':>12s} {'status':>9s} {'RL16':>9s} {'RL12':>9s} {'AC(honest)':>11s} "
              f"{'AAC(honest)':>12s} {'TWAP':>9s} {'RL16 vs AC (p)':>18s}")
        for date, status in w["days"].items():
            row = new_rows[date] if status == "new" else existing_row(date)
            row["window"] = wname
            row["window_label"] = w["label"]
            row["status"] = status
            all_rows.append(row)
            p = row["p_16_vs_ac"]
            pstr = f"{p:.4f}" if p is not None else "n/a"
            verdict = "BEATS" if row["rl16_beats_ac"] else "loses"
            print(f"{date:>12s} {status:>9s} {row['rl16_mean_IS']:>+9.4f} {row['rl12_mean_IS']:>+9.4f} "
                  f"{row['ac_mean_IS']:>+11.4f} {row['aac_mean_IS']:>+12.4f} {row['tw_mean_IS']:>+9.4f} "
                  f"  {row['delta_16_vs_ac']:+.4f} ({pstr}) {verdict}")
        wdf = pd.DataFrame([r for r in all_rows if r["window"] == wname])
        n = len(wdf)
        n_beat_ac = wdf["rl16_beats_ac"].sum()
        n_beat_aac = wdf["rl16_beats_aac"].sum()
        print(f"  Window {wname} summary: RL16 beats honest-AC on {n_beat_ac}/{n} days, "
              f"honest-AAC on {n_beat_aac}/{n} days")

    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(f"{OUTPUT_DIR}phase2r_severity_expansion_full.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR}phase2r_severity_expansion_full.csv ({len(full_df)} rows)")

    print(f"\n{'=' * 104}\nOVERALL (18 days across 3 windows, all individually reported above)\n{'=' * 104}")
    print(f"RL16 beats honest-AC: {full_df['rl16_beats_ac'].sum()}/{len(full_df)}   "
          f"RL16 beats honest-AAC: {full_df['rl16_beats_aac'].sum()}/{len(full_df)}")


if __name__ == "__main__":
    main()
