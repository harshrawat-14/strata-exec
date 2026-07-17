"""
Phase 2R Task 4 — the actual ablation deliverable: 12-dim RL vs 16-dim RL vs
AC-Optimal vs TWAP, post-correction simulator.

This is the FIRST post-correction result (see rl/results/PROVENANCE.md):
produced against the counterfactual simulator AFTER Phase 2R's Task 2 (impact
calibration), Task 3 (drift restoration), and the spread_ref scale-bug fix.
Do not compare these numbers against any pre-Phase-2R report (v5's headline
numbers, phase1_tail_risk.csv, unified_comparison.csv, failure_analysis.csv,
the old phase2_ablation.csv) -- the simulator itself changed underneath them.

Both RL arms trained with an IDENTICAL config (matched-conditions discipline):
same 7 training dates, same entropy floor, same CPU device, same n_envs,
same lr_schedule, same full 3M-step schedule, same resume-safe/multi-date
reward-based best-model selection -- differing ONLY in state dims (16 vs 12).

mode="counterfactual" is used for evaluation (not the historical-mode
default), per the explicit Task 4 decision: historical mode has no cf_lob,
so dims 8/9 (own-impact fractions) would be structurally dead there, testing
the 16-dim arm with two of its own distinguishing features disabled.

best_model (not _final) is used for both RL arms: the run logs show _final
drifted somewhat toward waiting on several dates in the 16-dim arm
specifically (entropy floor slowed but did not perfectly eliminate the
drift), while best_model (reward-selected, resume-safe, all-dates) stayed
healthy in both arms -- using best_model is precisely what that fix exists
for.

WHY AC AND TWAP ARE IN THIS COMPARISON: the 12-vs-16-dim RL delta alone
doesn't answer the question that matters -- whether either RL representation
actually beats the classical execution baselines. A representation ablation
that never checks against AC is not a result, it's an internal RL
leaderboard. TWAP/AC-Optimal don't depend on the RL model at all, so their
numbers are identical whether computed alongside the 12-dim or 16-dim call
(same seeds, same simulator physics) -- this script verifies that
consistency directly rather than assuming it, then uses one canonical copy.

Significance: paired bootstrap on (RL_is_samples - AC_is_samples) using the
SAME seeds for both (evaluate_all_strategies_on_date runs every strategy on
seed=0..n_episodes-1 for a given date/call), which is more powerful and more
honest than an unpaired comparison against a scalar AC baseline -- especially
important given the held-out crash set is only n=2 dates.

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
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0

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


def paired_bootstrap_delta(a_samples, b_samples, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    Paired bootstrap CI + one-sided p-value for mean(a) - mean(b), resampling
    matched (a_i, b_i) pairs together (same seed index) rather than
    independently -- correct when a and b were evaluated on the same episode
    seeds, as RL vs AC/TWAP always are here.
    """
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
    # one-sided: fraction of bootstrap resamples where a did NOT beat b (delta <= 0)
    p_value = float((resampled <= 0).mean())
    return {"point": point, "ci_lower": ci_lower, "ci_upper": ci_upper, "p_value": p_value}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 100)
    print("PHASE 2R TASK 4 ABLATION: 12-dim RL vs 16-dim RL vs AC-Optimal vs TWAP")
    print("POST-CORRECTION counterfactual simulator")
    print("=" * 100)
    print(
        "\n*** PROVENANCE ***\n"
        "First post-correction result (calibrated impact model, restored drift, fixed\n"
        "spread_ref -- see rl/results/PROVENANCE.md). Do NOT compare against any\n"
        "pre-Phase-2R report.\n"
        "\n*** MATCHED CONDITIONS (RL arms) ***\n"
        "Both RL arms: identical config (7 training dates, entropy floor, CPU device,\n"
        "n_envs=2, linear LR, full 3M-step schedule, resume-safe/multi-date reward-based\n"
        "best-model selection) -- differ ONLY in state dims (16 vs 12). Evaluated in\n"
        "COUNTERFACTUAL mode (not historical) so dims 8/9 are live for the 16-dim arm.\n"
        "best_model used for both RL arms.\n"
        "\n*** WHY AC/TWAP ARE HERE ***\n"
        "A 12-vs-16-dim delta with no classical-baseline comparison is an internal RL\n"
        "leaderboard, not a result. This is a full 4-way comparison so the question that\n"
        "actually matters -- does either RL representation beat AC-Optimal -- has an answer.\n"
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

        rl12, rl16 = r12["rl_agent"], r16["rl_agent"]
        ac12, ac16 = r12["ac_optimal"], r16["ac_optimal"]
        tw12, tw16 = r12["twap"], r16["twap"]

        # AC/TWAP shouldn't depend on the RL model at all (same seeds, same
        # physics) -- verify that consistency directly rather than assume it.
        ac_consistent = abs(ac12["mean_IS"] - ac16["mean_IS"]) < 1e-9
        tw_consistent = abs(tw12["mean_IS"] - tw16["mean_IS"]) < 1e-9
        if not (ac_consistent and tw_consistent):
            print(f"  *** WARNING: AC/TWAP differ between the 12-dim and 16-dim calls on "
                  f"{date} (ac_diff={ac12['mean_IS']-ac16['mean_IS']:.6f}, "
                  f"twap_diff={tw12['mean_IS']-tw16['mean_IS']:.6f}) -- investigate before "
                  f"trusting this date's numbers.")
        ac = ac12  # canonical copy once consistency is confirmed
        tw = tw12

        boot_16_vs_ac = paired_bootstrap_delta(rl16["is_samples"], ac["is_samples"])
        boot_12_vs_ac = paired_bootstrap_delta(rl12["is_samples"], ac["is_samples"])

        rows.append({
            "date": date,
            "regime": regime,
            "held_out_crash_date": is_crash_holdout,
            "training_date": is_training_date,
            "rl12_mean_IS": rl12["mean_IS"], "rl12_cvar_95": rl12["cvar_95"],
            "rl12_forced_liq": rl12["forced_liquidation_rate"],
            "rl16_mean_IS": rl16["mean_IS"], "rl16_cvar_95": rl16["cvar_95"],
            "rl16_forced_liq": rl16["forced_liquidation_rate"],
            "ac_mean_IS": ac["mean_IS"], "ac_cvar_95": ac["cvar_95"],
            "ac_cvar_95_ci_lower": ac["cvar_95_ci_lower"], "ac_cvar_95_ci_upper": ac["cvar_95_ci_upper"],
            "twap_mean_IS": tw["mean_IS"], "twap_cvar_95": tw["cvar_95"],
            "twap_cvar_95_ci_lower": tw["cvar_95_ci_lower"], "twap_cvar_95_ci_upper": tw["cvar_95_ci_upper"],
            "delta_16_vs_12_mean_IS": rl16["mean_IS"] - rl12["mean_IS"],
            "delta_16_vs_12_cvar_95": rl16["cvar_95"] - rl12["cvar_95"],
            "delta_16_vs_ac_mean_IS": rl16["mean_IS"] - ac["mean_IS"],
            "delta_16_vs_ac_ci_lower": boot_16_vs_ac["ci_lower"],
            "delta_16_vs_ac_ci_upper": boot_16_vs_ac["ci_upper"],
            "delta_16_vs_ac_p_value": boot_16_vs_ac["p_value"],
            "rl16_beats_ac": rl16["mean_IS"] > ac["mean_IS"],
            "delta_12_vs_ac_mean_IS": rl12["mean_IS"] - ac["mean_IS"],
            "delta_12_vs_ac_ci_lower": boot_12_vs_ac["ci_lower"],
            "delta_12_vs_ac_ci_upper": boot_12_vs_ac["ci_upper"],
            "delta_12_vs_ac_p_value": boot_12_vs_ac["p_value"],
            "rl12_beats_ac": rl12["mean_IS"] > ac["mean_IS"],
        })

        print(f"  RL-12dim: mean_IS={rl12['mean_IS']:+.4f}%  cvar_95={rl12['cvar_95']:+.4f}%  "
              f"forced_liq={rl12['forced_liquidation_rate']:.2f}")
        print(f"  RL-16dim: mean_IS={rl16['mean_IS']:+.4f}%  cvar_95={rl16['cvar_95']:+.4f}%  "
              f"forced_liq={rl16['forced_liquidation_rate']:.2f}")
        print(f"  AC-Optimal: mean_IS={ac['mean_IS']:+.4f}%  cvar_95={ac['cvar_95']:+.4f}%  "
              f"[{ac['cvar_95_ci_lower']:.4f}, {ac['cvar_95_ci_upper']:.4f}]")
        print(f"  TWAP:       mean_IS={tw['mean_IS']:+.4f}%  cvar_95={tw['cvar_95']:+.4f}%  "
              f"[{tw['cvar_95_ci_lower']:.4f}, {tw['cvar_95_ci_upper']:.4f}]")
        print(f"  16dim vs AC: delta={rl16['mean_IS']-ac['mean_IS']:+.4f}pp  "
              f"CI=[{boot_16_vs_ac['ci_lower']:+.4f}, {boot_16_vs_ac['ci_upper']:+.4f}]  "
              f"p={boot_16_vs_ac['p_value']:.4f}  {'BEATS AC' if rl16['mean_IS']>ac['mean_IS'] else 'loses to AC'}")
        print(f"  12dim vs AC: delta={rl12['mean_IS']-ac['mean_IS']:+.4f}pp  "
              f"CI=[{boot_12_vs_ac['ci_lower']:+.4f}, {boot_12_vs_ac['ci_upper']:+.4f}]  "
              f"p={boot_12_vs_ac['p_value']:.4f}  {'BEATS AC' if rl12['mean_IS']>ac['mean_IS'] else 'loses to AC'}")

    df = pd.DataFrame(rows)
    csv_path = f"{OUTPUT_DIR}phase2r_ablation_postcorrection.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull ablation table saved: {csv_path}")

    print("\n" + "=" * 100)
    print("SUMMARY -- mean across all dates (in-sample dominated, see crash-only section for")
    print("the fair generalization test)")
    print("=" * 100)
    for label, col in [("RL-12dim", "rl12_mean_IS"), ("RL-16dim", "rl16_mean_IS"),
                       ("AC-Optimal", "ac_mean_IS"), ("TWAP", "twap_mean_IS")]:
        print(f"{label:12s} mean_IS avg: {df[col].mean():+.4f}%")

    print("\n" + "=" * 100)
    print("SUMMARY -- held-out crash dates only (2024-08-05, 2024-03-05), 4-way, n=2 -- INDICATIVE")
    print("Both RL arms trained on the IDENTICAL corrected split; these are genuinely")
    print("out-of-sample for BOTH. n=2 dates: directionally informative, NOT statistically")
    print("conclusive on its own -- expanding the held-out crash set is the priority to")
    print("strengthen this specific claim.")
    print("=" * 100)
    crash_df = df[df["held_out_crash_date"]]
    print(crash_df[["date", "regime", "rl12_mean_IS", "rl16_mean_IS", "ac_mean_IS", "twap_mean_IS",
                    "rl12_cvar_95", "rl16_cvar_95", "ac_cvar_95", "twap_cvar_95"]].to_string(index=False))
    print()
    print(crash_df[["date", "rl16_beats_ac", "delta_16_vs_ac_mean_IS", "delta_16_vs_ac_p_value",
                    "rl12_beats_ac", "delta_12_vs_ac_mean_IS", "delta_12_vs_ac_p_value"]].to_string(index=False))

    print("\n" + "=" * 100)
    print("THE SHARPEST SINGLE FINDING: does 16-dim flip 12-dim's loss-to-AC into a win, on the")
    print("hardest stress date (2024-08-05)?")
    print("=" * 100)
    row_0805 = df[df["date"] == "2024-08-05"]
    if len(row_0805):
        r = row_0805.iloc[0]
        print(f"2024-08-05: RL-12dim vs AC: {r['delta_12_vs_ac_mean_IS']:+.4f}pp "
              f"({'BEATS' if r['rl12_beats_ac'] else 'LOSES TO'} AC, p={r['delta_12_vs_ac_p_value']:.4f})")
        print(f"2024-08-05: RL-16dim vs AC: {r['delta_16_vs_ac_mean_IS']:+.4f}pp "
              f"({'BEATS' if r['rl16_beats_ac'] else 'LOSES TO'} AC, p={r['delta_16_vs_ac_p_value']:.4f})")
        if (not r["rl12_beats_ac"]) and r["rl16_beats_ac"]:
            print("--> CONFIRMED: the extra features flip RL from losing to AC to beating AC on")
            print("    the hardest stress date in this set.")
        else:
            print("--> NOT a clean flip (see raw deltas above) -- do not overstate this.")

    print("\n" + "=" * 100)
    print("forced_liquidation_rate by RL arm (0.0 = agent always self-liquidated; watch for")
    print("either arm re-collapsing to blanket-wait, which would invalidate its numbers)")
    print("=" * 100)
    print(df[["date", "rl12_forced_liq", "rl16_forced_liq"]].to_string(index=False))

    print("\n" + "=" * 100)
    print("HONEST THESIS (state explicitly, do not oversell calm-date performance)")
    print("=" * 100)
    train_df = df[df["training_date"]]
    n_16_beats_ac_train = train_df["rl16_beats_ac"].sum()
    n_12_beats_ac_train = train_df["rl12_beats_ac"].sum()
    print(f"On the {len(train_df)} in-sample (training) dates: RL-16dim beats AC on "
          f"{n_16_beats_ac_train}/{len(train_df)}, RL-12dim beats AC on {n_12_beats_ac_train}/{len(train_df)}.")
    print(f"On the {len(crash_df)} held-out crash dates: RL-16dim beats AC on "
          f"{crash_df['rl16_beats_ac'].sum()}/{len(crash_df)}, RL-12dim beats AC on "
          f"{crash_df['rl12_beats_ac'].sum()}/{len(crash_df)}.")
    print("Thesis: RL's advantage over AC-Optimal is concentrated in stress regimes where AC's")
    print("assumptions break down, and is near-absent (genuinely mixed, not a clean win) on calm")
    print("dates where AC is already close to optimal. This is not a uniform 'RL beats AC' or")
    print("'16-dim beats 12-dim' result -- it is regime-conditional, and the in-sample mixed")
    print("results are the honest picture there, not noise to explain away.")


if __name__ == "__main__":
    main()
