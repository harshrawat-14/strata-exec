"""
Phase 2R Task 4 — the actual ablation deliverable: 12-dim RL vs 16-dim RL vs
AC-Optimal vs Adaptive-AC vs TWAP, post-correction simulator, ALL clean dates.

This is the FIRST post-correction result (see rl/results/PROVENANCE.md):
produced against the counterfactual simulator AFTER Phase 2R's Task 2 (impact
calibration), Task 3 (drift restoration), the spread_ref scale-bug fix, AND
the classical-baseline action-fraction-table fix (Python no longer
hand-maintains a mirror of the real Rust ACTION_FRACTIONS -- see
get_action_fractions in evaluate.py). Do not compare these numbers against
any pre-Phase-2R report -- the simulator itself changed underneath them.

Both RL arms trained with an IDENTICAL config (matched-conditions discipline):
same 7 training dates, same entropy floor, same CPU device, same n_envs,
same lr_schedule, same full 3M-step schedule, same resume-safe/multi-date
reward-based best-model selection -- differing ONLY in state dims (16 vs 12).
They are REUSED here, not retrained: the fraction-table bug never touched RL
(it picks raw action indices from a trained policy, never routed through the
classical-baseline fraction tables), so only the baselines needed redoing.

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

ALL 11 CLEAN DATES ARE USED, not just the 9 from the first pass. Two dates
(2024-07-10, 2024-12-10) were sitting unused -- neither training dates nor
the 2 held-out crash dates -- and are genuinely held-out CALM/normal dates.
Including them enlarges the out-of-sample evidence beyond n=2 (crash-only)
and lets the "calm is a wash out-of-sample" half of the thesis be tested
directly instead of relying on in-sample calm evidence, which is weak by
construction. All 11 dates passed a basic data-quality check (consistent
BookDepth/AggTrades row counts, no truncation) before inclusion.

WHY AC, ADAPTIVE-AC, AND TWAP ARE ALL IN THIS COMPARISON: the 12-vs-16-dim
RL delta alone doesn't answer the question that matters -- whether either RL
representation actually beats the classical execution baselines, including
the STRONGER one. Standard AC is a static precomputed schedule; Adaptive-AC
recomputes as conditions change and is the harder, more honest target --
beating standard AC but losing to Adaptive-AC would be a materially weaker
result than beating both. TWAP/AC-Optimal/Adaptive-AC don't depend on the RL
model at all, so their numbers are identical whether computed alongside the
12-dim or 16-dim call (same seeds, same simulator physics) -- this script
verifies that consistency directly (now that the fraction-table bug is
fixed, it should hold) rather than assuming it.

Significance: paired bootstrap on (RL_is_samples - baseline_is_samples)
using the SAME seeds for both (evaluate_all_strategies_on_date runs every
strategy on seed=0..n_episodes-1 for a given date/call), more powerful and
more honest than an unpaired comparison against a scalar baseline --
especially important given the held-out crash set is only n=2 dates.

Usage: rl/.venv/bin/python rl/experiments/phase2r_ablation_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

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
    """
    Paired bootstrap CI + one-sided p-value for mean(a) - mean(b), resampling
    matched (a_i, b_i) pairs together (same seed index) rather than
    independently -- correct when a and b were evaluated on the same episode
    seeds, as RL vs any baseline always is here.
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
    p_value = float((resampled <= 0).mean())
    return {"point": point, "ci_lower": ci_lower, "ci_upper": ci_upper, "p_value": p_value}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 104)
    print("PHASE 2R TASK 4 ABLATION: RL-12dim vs RL-16dim vs AC-Optimal vs Adaptive-AC vs TWAP")
    print("POST-CORRECTION simulator, ALL 11 clean dates, fixed action-fraction table")
    print("=" * 104)
    print(
        "\n*** PROVENANCE ***\n"
        "First post-correction result: calibrated impact model, restored drift, fixed\n"
        "spread_ref, AND fixed classical-baseline action-fraction table (see\n"
        "rl/results/PROVENANCE.md). Do NOT compare against any pre-Phase-2R report.\n"
        "\n*** MATCHED CONDITIONS (RL arms, reused not retrained) ***\n"
        "Both RL arms: identical config (7 training dates, entropy floor, CPU device,\n"
        "n_envs=2, linear LR, full 3M-step schedule, resume-safe/multi-date reward-based\n"
        "best-model selection) -- differ ONLY in state dims (16 vs 12). Evaluated in\n"
        "COUNTERFACTUAL mode. best_model used for both. Reused from the prior run --\n"
        "the fraction-table bug never touched RL (raw action indices, no table lookup).\n"
        "\n*** ALL 11 CLEAN DATES ***\n"
        "7 training + 2 held-out CRASH (2024-03-05, 2024-08-05) + 2 held-out CALM\n"
        "(2024-07-10, 2024-12-10, previously unused) -- enlarging out-of-sample evidence\n"
        "beyond the crash-only n=2 and enabling a direct out-of-sample test of the\n"
        "'calm is a wash' half of the thesis.\n"
    )

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
            mode="counterfactual", btc_target=BTC_TARGET,
        )
        print("  [16-dim]")
        r16 = evaluate_all_strategies_on_date(
            rl_model=v16, date=date, n_episodes=N_EPISODES, n_state_dims=16,
            mode="counterfactual", btc_target=BTC_TARGET,
        )

        rl12, rl16 = r12["rl_agent"], r16["rl_agent"]

        baselines = {}
        for short, key in BASELINE_KEYS:
            b12, b16 = r12[key], r16[key]
            consistent = abs(b12["mean_IS"] - b16["mean_IS"]) < 1e-9
            if not consistent:
                any_inconsistency = True
                print(f"  *** WARNING: {BASELINE_LABELS[short]} differs between the 12-dim and "
                      f"16-dim calls on {date} (diff={b12['mean_IS']-b16['mean_IS']:.6f}) -- "
                      f"the fraction-table fix should have made this impossible, investigate.")
            baselines[short] = b12  # canonical copy, now expected to equal b16

        row = {
            "date": date, "regime": regime, "split": split,
            "rl12_mean_IS": rl12["mean_IS"], "rl12_cvar_95": rl12["cvar_95"],
            "rl12_forced_liq": rl12["forced_liquidation_rate"],
            "rl16_mean_IS": rl16["mean_IS"], "rl16_cvar_95": rl16["cvar_95"],
            "rl16_forced_liq": rl16["forced_liquidation_rate"],
            "delta_16_vs_12_mean_IS": rl16["mean_IS"] - rl12["mean_IS"],
            "delta_16_vs_12_cvar_95": rl16["cvar_95"] - rl12["cvar_95"],
        }
        for short, _ in BASELINE_KEYS:
            b = baselines[short]
            row[f"{short}_mean_IS"] = b["mean_IS"]
            row[f"{short}_cvar_95"] = b["cvar_95"]
            row[f"{short}_cvar_95_ci_lower"] = b["cvar_95_ci_lower"]
            row[f"{short}_cvar_95_ci_upper"] = b["cvar_95_ci_upper"]
            for arm_label, arm in [("16", rl16), ("12", rl12)]:
                boot = paired_bootstrap_delta(arm["is_samples"], b["is_samples"])
                row[f"delta_{arm_label}_vs_{short}_mean_IS"] = arm["mean_IS"] - b["mean_IS"]
                row[f"delta_{arm_label}_vs_{short}_ci_lower"] = boot["ci_lower"]
                row[f"delta_{arm_label}_vs_{short}_ci_upper"] = boot["ci_upper"]
                row[f"delta_{arm_label}_vs_{short}_p_value"] = boot["p_value"]
                row[f"rl{arm_label}_beats_{short}"] = arm["mean_IS"] > b["mean_IS"]
        rows.append(row)

        print(f"  RL-12dim: mean_IS={rl12['mean_IS']:+.4f}%  cvar_95={rl12['cvar_95']:+.4f}%  "
              f"forced_liq={rl12['forced_liquidation_rate']:.2f}")
        print(f"  RL-16dim: mean_IS={rl16['mean_IS']:+.4f}%  cvar_95={rl16['cvar_95']:+.4f}%  "
              f"forced_liq={rl16['forced_liquidation_rate']:.2f}")
        for short, _ in BASELINE_KEYS:
            b = baselines[short]
            print(f"  {BASELINE_LABELS[short]:12s}: mean_IS={b['mean_IS']:+.4f}%  "
                  f"cvar_95={b['cvar_95']:+.4f}% [{b['cvar_95_ci_lower']:.4f}, {b['cvar_95_ci_upper']:.4f}]")
            for arm_label in ("16", "12"):
                d = row[f"delta_{arm_label}_vs_{short}_mean_IS"]
                p = row[f"delta_{arm_label}_vs_{short}_p_value"]
                beats = row[f"rl{arm_label}_beats_{short}"]
                print(f"    RL-{arm_label}dim vs {BASELINE_LABELS[short]}: delta={d:+.4f}pp  "
                      f"p={p:.4f}  {'BEATS' if beats else 'loses to'}")

    if not any_inconsistency:
        print("\n*** Consistency check passed on every date: AC/Adaptive-AC/TWAP results are "
              "identical between the 12-dim and 16-dim calls, confirming the fraction-table fix. ***")

    df = pd.DataFrame(rows)
    csv_path = f"{OUTPUT_DIR}phase2r_ablation_postcorrection.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull ablation table saved: {csv_path}")

    print("\n" + "=" * 104)
    print("SUMMARY -- mean across all 11 dates (in-sample dominated by 7 training dates; see")
    print("split-out sections below for the fair generalization tests)")
    print("=" * 104)
    for label, col in [("RL-12dim", "rl12_mean_IS"), ("RL-16dim", "rl16_mean_IS"),
                       ("AC-Optimal", "ac_mean_IS"), ("Adaptive-AC", "aac_mean_IS"),
                       ("TWAP", "tw_mean_IS")]:
        print(f"{label:12s} mean_IS avg: {df[col].mean():+.4f}%")

    for split_name, split_key, note in [
        ("HELD-OUT CRASH (2024-03-05, 2024-08-05)", "held_out_crash",
         "n=2: directionally informative, NOT statistically conclusive on its own."),
        ("HELD-OUT CALM (2024-07-10, 2024-12-10)", "held_out_calm",
         "The out-of-sample test of the 'calm is a wash' half of the thesis."),
        ("IN-SAMPLE TRAINING (7 dates)", "training",
         "In-sample; the honest (mixed) picture, not evidence to explain away."),
    ]:
        sub = df[df["split"] == split_key]
        if len(sub) == 0:
            continue
        print("\n" + "=" * 104)
        print(f"SUMMARY -- {split_name}")
        print(note)
        print("=" * 104)
        print(sub[["date", "regime", "rl12_mean_IS", "rl16_mean_IS", "ac_mean_IS",
                   "aac_mean_IS", "tw_mean_IS"]].to_string(index=False))
        for short, _ in BASELINE_KEYS:
            n16 = sub[f"rl16_beats_{short}"].sum()
            n12 = sub[f"rl12_beats_{short}"].sum()
            print(f"  RL-16dim beats {BASELINE_LABELS[short]}: {n16}/{len(sub)}   "
                  f"RL-12dim beats {BASELINE_LABELS[short]}: {n12}/{len(sub)}")

    print("\n" + "=" * 104)
    print("THE SHARPEST SINGLE FINDING: does 16-dim flip 12-dim's loss-to-AC into a win, on the")
    print("hardest stress date (2024-08-05)? (vs standard AC-Optimal specifically)")
    print("=" * 104)
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

    print("\n" + "=" * 104)
    print("forced_liquidation_rate by RL arm (0.0 = agent always self-liquidated; watch for")
    print("either arm re-collapsing to blanket-wait, which would invalidate its numbers)")
    print("=" * 104)
    print(df[["date", "split", "rl12_forced_liq", "rl16_forced_liq"]].to_string(index=False))

    print("\n" + "=" * 104)
    print("HONEST THESIS")
    print("=" * 104)
    crash_df = df[df["split"] == "held_out_crash"]
    calm_df = df[df["split"] == "held_out_calm"]
    train_df = df[df["split"] == "training"]
    print(f"In-sample (7 training dates): RL-16dim beats AC on {train_df['rl16_beats_ac'].sum()}/7, "
          f"RL-12dim beats AC on {train_df['rl12_beats_ac'].sum()}/7.")
    print(f"Held-out CRASH (2 dates): RL-16dim beats AC on {crash_df['rl16_beats_ac'].sum()}/2, "
          f"RL-12dim beats AC on {crash_df['rl12_beats_ac'].sum()}/2.")
    print(f"Held-out CALM (2 dates): RL-16dim beats AC on {calm_df['rl16_beats_ac'].sum()}/2, "
          f"RL-12dim beats AC on {calm_df['rl12_beats_ac'].sum()}/2.")
    print()
    print("AC-Optimal winning or tying on calm dates is EXPECTED and CORRECT, not a problem: AC")
    print("is provably optimal under its assumptions (roughly-linear impact, roughly-normal price")
    print("dynamics), and calm days are exactly when those assumptions hold -- there is little room")
    print("for RL to beat a baseline that is already near-optimal in its own designed regime. If RL")
    print("dramatically beat AC on calm days, THAT would be suspicious (a bug or a mis-implemented")
    print("baseline), not a win. Watch specifically for RL genuinely LOSING (not tying) on calm --")
    print("that would indicate RL mis-executing even in easy conditions, a real concern:")
    for short, _ in BASELINE_KEYS:
        calm_delta_16 = calm_df[f"delta_16_vs_{short}_mean_IS"].mean() if len(calm_df) else float("nan")
        calm_delta_12 = calm_df[f"delta_12_vs_{short}_mean_IS"].mean() if len(calm_df) else float("nan")
        print(f"  calm-date mean delta vs {BASELINE_LABELS[short]}: RL-16dim={calm_delta_16:+.4f}pp  "
              f"RL-12dim={calm_delta_12:+.4f}pp")
    print()
    print("Thesis: RL's advantage over the classical baselines is concentrated in stress regimes")
    print("where their assumptions break down (crash dates), and is a wash -- not a clean win, but")
    print("also not a loss -- on calm dates where AC/Adaptive-AC are already close to optimal. This")
    print("is regime-conditional, not a uniform 'RL beats everything' or '16-dim beats 12-dim'")
    print("result, and the held-out calm dates now test the 'wash' half out-of-sample rather than")
    print("relying on the weaker in-sample calm evidence.")


if __name__ == "__main__":
    main()
