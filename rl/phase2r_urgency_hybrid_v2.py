"""
Phase 2R close-out gate, follow-up correction to the Task A hybrid experiment.

The first hybrid (rl/phase2r_urgency_diagnosis.py) capped every catch-up step at
RL's own median chunk size -- a constraint AC itself is never subject to -- and
that cap was found to be the binding constraint on 2024-08-05 (the hybrid fell
short of tuned AC there). That means the first experiment's own construction
could have produced its shortfall; the numbers reported were a lower bound on
what "AC's pacing + fine slicing" can do, not a verdict. This reruns the same
hybrid construction with two more generous caps:
  (a) cap = AC's own maximum single-step chunk actually used in its real
      trajectory (excluding the terminal forced-liquidation dump), i.e. "as
      fast as tuned AC itself ever moves, just sliced finer when it doesn't
      need to move that fast"
  (b) uncapped (cap = 1.0, i.e. no artificial ceiling at all)

Also corrects the earlier framing: report the full ranking (RL / AC / hybrid),
not just "beats AC" or "loses to AC" framed against a single comparator.

Usage: rl/.venv/bin/python rl/phase2r_urgency_hybrid_v2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from environment import ACTION_FRACTIONS
from evaluate import load_model, ACOptimalPolicy, nearest_action
from phase2r_urgency_diagnosis import (
    MODEL_16DIM, BEST_AC_LAMBDA, REGIME, _mk_env, _attach_forced_liq, decompose_is,
    collect_rl_trajectory_with_state, collect_ac_trajectory,
)


def collect_hybrid_trajectory_v2(date, cap_frac, label):
    env = _mk_env(date)
    obs, info = env.reset(seed=0)
    ac_schedule = ACOptimalPolicy(n_state_dims=16, lam=BEST_AC_LAMBDA[date])._schedule
    total_steps = len(ac_schedule)
    rows = []
    done = False
    step = 0
    while not done:
        q_frac = float(obs[0])
        t_frac = float(obs[1])
        if t_frac < 0.001 or q_frac < 0.001:
            action_idx = nearest_action(1.0, ACTION_FRACTIONS)
        else:
            step_idx = min(int((1.0 - t_frac) * total_steps), total_steps - 1)
            target_cum_orig = ac_schedule[:step_idx + 1].sum()
            executed_orig = 1.0 - q_frac
            behind = max(target_cum_orig - executed_orig, 0.0)
            behind_of_remaining = min(1.0, behind / q_frac) if q_frac > 1e-6 else 1.0
            target = min(behind_of_remaining, cap_frac)
            action_idx = nearest_action(target, ACTION_FRACTIONS)
        obs, reward, done, _, info = env.step(action_idx)
        rows.append({"step": step, "action_idx": action_idx, "action_frac": ACTION_FRACTIONS[action_idx],
                     "fill_qty": info.get("fill_qty", 0.0), "fill_price": info.get("fill_price", 0.0),
                     "book_mid": info.get("book_mid", np.nan)})
        step += 1
    _attach_forced_liq(rows, info)
    df = pd.DataFrame(rows)
    decomp = decompose_is(df[df["fill_qty"] > 0], 100.0)
    return df, decomp


def main():
    print("Loading RL-16dim model...")
    model = load_model(MODEL_16DIM)

    summary = []
    for date in ["2024-08-05", "2024-03-05"]:
        rl_df, _, rl_info = collect_rl_trajectory_with_state(model, date)
        ac_df, ac_info = collect_ac_trajectory(date)
        rl_decomp = decompose_is(rl_df[rl_df["fill_qty"] > 0], 100.0)
        ac_decomp = decompose_is(ac_df[ac_df["fill_qty"] > 0], 100.0)

        # AC's own max single-step chunk, excluding the terminal forced-liq row
        # (action_idx == -1 marks that synthetic row; exclude it here).
        ac_voluntary = ac_df[(ac_df["fill_qty"] > 0) & (ac_df["action_idx"] != -1)]
        ac_max_chunk = float(ac_voluntary["action_frac"].max()) if len(ac_voluntary) else 1.0

        rl_nonzero = rl_df[(rl_df["fill_qty"] > 0) & (rl_df["action_frac"].notna())]
        rl_median_cap = float(rl_nonzero["action_frac"].median())

        variants = {
            "hybrid_rl_median_cap (original)": rl_median_cap,
            "hybrid_ac_max_chunk_cap": ac_max_chunk,
            "hybrid_uncapped": 1.0,
        }

        print(f"\n{'=' * 100}\n{date} ({REGIME[date]})  AC's own max voluntary chunk={ac_max_chunk:.3f}  "
              f"RL median chunk={rl_median_cap:.4f}\n{'=' * 100}")

        results = {"RL16 (pure)": rl_decomp, "AC (pure, best λ)": ac_decomp}
        for label, cap in variants.items():
            _, decomp = collect_hybrid_trajectory_v2(date, cap, label)
            results[label] = decomp

        ranked = sorted(results.items(), key=lambda kv: kv[1]["total_pct"], reverse=True)
        print(f"  {'policy':>34s} {'drift_pct':>10s} {'exec_pct':>10s} {'dust_pct':>10s} {'total':>10s}")
        for name, d in results.items():
            print(f"  {name:>34s} {d['drift_pct']:>+10.4f} {d['execution_pct']:>+10.4f} "
                  f"{d['dust_pct']:>+10.4f} {d['total_pct']:>+10.4f}")
        print(f"\n  RANKING (best to worst): " + " > ".join(f"{name} ({d['total_pct']:+.4f})" for name, d in ranked))

        for name, d in results.items():
            summary.append({"date": date, "regime": REGIME[date], "policy": name, **d})

    out = pd.DataFrame(summary)
    out.to_csv("rl/results/phase2r_urgency_hybrid_v2.csv", index=False)
    print(f"\nSaved rl/results/phase2r_urgency_hybrid_v2.csv")


if __name__ == "__main__":
    main()
