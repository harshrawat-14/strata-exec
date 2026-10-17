"""
Phase 2R close-out gate: two analyses requested before an item 5/6/residual-phase
recommendation.

Task A: cheaply test the implication of "RL learned execution but not urgency" --
does AC's pacing combined with RL's execution style beat tuned AC, WITHOUT any
training? Two versions: (1) a naive analytical estimate (AC's drift_pct + RL's
execution_pct, from the already-collected decomposition), and (2) an actual
simulated hybrid policy -- AC's cumulative schedule sets how much SHOULD be sold
by each step, but the catch-up is sliced into RL-sized chunks (not AC's own,
occasionally much larger, per-step blocks) and run through the real simulator.

Task B: is regime information present and discriminative in the state EARLY in
the episode (when urgency must be decided), or does it arrive late/attenuated?
Checks dims 2 (vol ratio), 10-11 (vol/liq regime), 13 (intraday vol ratio) at the
first several dozen decision points, crash dates vs calm dates.

Usage: rl/.venv/bin/python rl/experiments/phase2r_urgency_diagnosis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np
import pandas as pd

from environment import StrataExecEnv, ACTION_FRACTIONS
from evaluate import load_model, ACOptimalPolicy, nearest_action

MODEL_16DIM = "rl/models/ppo_ablation_16dim_cf_best/best_model"
BTC_TARGET = 50_000.0
SEED = 0
TOTAL_INVENTORY = 1_000_000.0

BEST_AC_LAMBDA = {"2024-03-05": 1e-4, "2024-08-05": 1e-3, "2024-07-10": 1e-6, "2024-12-10": 1e-6}
REGIME = {"2024-03-05": "crash (milder)", "2024-08-05": "crash (severe)",
          "2024-07-10": "calm", "2024-12-10": "calm"}


def _mk_env(date):
    return StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=BTC_TARGET,
                          n_state_dims=16, fixed_steps=True, fixed_size=True, no_time_features=False)


def decompose_is(fills: pd.DataFrame, arrival_price: float, total_inventory=TOTAL_INVENTORY):
    if fills.empty:
        return {"drift_pct": 0.0, "execution_pct": 0.0, "dust_pct": -100.0, "total_pct": -100.0, "n_fills": 0}
    ideal = arrival_price * total_inventory
    drift_dollar = (fills["fill_qty"] * (fills["book_mid"] - arrival_price)).sum()
    exec_dollar = (fills["fill_qty"] * (fills["fill_price"] - fills["book_mid"])).sum()
    executed_qty = fills["fill_qty"].sum()
    dust_dollar = -max(total_inventory - executed_qty, 0.0) * arrival_price
    return {
        "drift_pct": drift_dollar / ideal * 100, "execution_pct": exec_dollar / ideal * 100,
        "dust_pct": dust_dollar / ideal * 100,
        "total_pct": (drift_dollar + exec_dollar + dust_dollar) / ideal * 100,
        "n_fills": len(fills),
    }


# ---------------------------------------------------------------------------
# Task A: RL trajectory collection (also records the full state vector per
# step, needed for Task B) + the AC-paced/RL-sliced hybrid policy.
# ---------------------------------------------------------------------------

def collect_rl_trajectory_with_state(model, date):
    env = _mk_env(date)
    obs, info = env.reset(seed=SEED)
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    rows, states = [], []
    done = False
    step = 0
    while not done:
        states.append({"step": step, **{f"s{i}": float(obs[i]) for i in range(len(obs))}})
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start,
                                             deterministic=True)
        episode_start = np.zeros((1,), dtype=bool)
        action_idx = int(action)
        obs, reward, done, _, info = env.step(action_idx)
        rows.append({"step": step, "action_idx": action_idx, "action_frac": ACTION_FRACTIONS[action_idx],
                     "fill_qty": info.get("fill_qty", 0.0), "fill_price": info.get("fill_price", 0.0),
                     "book_mid": info.get("book_mid", np.nan)})
        step += 1
    _attach_forced_liq(rows, info)
    return pd.DataFrame(rows), pd.DataFrame(states), info


def _attach_forced_liq(rows, info):
    fl_qty = info.get("forced_liquidation_qty", 0.0)
    if fl_qty and fl_qty > 0:
        total_notional = info.get("forced_liq_filled_notional", 0.0) + info.get("forced_liq_punitive_notional", 0.0)
        eff_price = total_notional / fl_qty if fl_qty > 1e-12 else info.get("book_mid", 100.0)
        rows.append({"step": rows[-1]["step"] + 1 if rows else 0, "action_idx": -1, "action_frac": None,
                     "fill_qty": fl_qty, "fill_price": eff_price, "book_mid": info.get("book_mid", 100.0)})


def collect_ac_trajectory(date):
    env = _mk_env(date)
    obs, info = env.reset(seed=SEED)
    policy = ACOptimalPolicy(n_state_dims=16, lam=BEST_AC_LAMBDA[date])
    rows = []
    done = False
    step = 0
    while not done:
        action, _ = policy.predict(obs)
        action_idx = int(action[0])
        obs, reward, done, _, info = env.step(action_idx)
        rows.append({"step": step, "action_idx": action_idx, "action_frac": ACTION_FRACTIONS[action_idx],
                     "fill_qty": info.get("fill_qty", 0.0), "fill_price": info.get("fill_price", 0.0),
                     "book_mid": info.get("book_mid", np.nan)})
        step += 1
    _attach_forced_liq(rows, info)
    return pd.DataFrame(rows), info


def collect_hybrid_trajectory(date, rl_step_cap):
    """
    AC's own cumulative schedule sets how much SHOULD have been sold by each
    step (its urgency/pacing); the catch-up is capped at rl_step_cap (RL's
    typical per-step chunk size on THIS date) and sliced across as many steps
    as needed, rather than executed in whatever single (possibly much larger)
    bucket AC-Optimal itself would pick. If AC is behind its own schedule by
    more than the cap, the policy keeps requesting rl_step_cap-sized chunks
    every step until caught up -- same overall urgency curve, RL-sized slices.
    """
    env = _mk_env(date)
    obs, info = env.reset(seed=SEED)
    ac_schedule = ACOptimalPolicy(n_state_dims=16, lam=BEST_AC_LAMBDA[date])._schedule
    total_steps = len(ac_schedule)
    cap_idx = nearest_action(rl_step_cap, ACTION_FRACTIONS)
    cap_frac = ACTION_FRACTIONS[cap_idx]
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
            target_cum_orig = ac_schedule[:step_idx + 1].sum()  # AC's cumulative target, frac of ORIGINAL inv
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
    return pd.DataFrame(rows), info, cap_frac


def task_a(model):
    print(f"\n{'#' * 90}\nTASK A: does AC-pacing + RL-style slicing beat tuned AC, without training?\n{'#' * 90}")
    summary = []
    for date in ["2024-08-05", "2024-03-05"]:
        arrival = 100.0
        rl_df, rl_states, rl_info = collect_rl_trajectory_with_state(model, date)
        ac_df, ac_info = collect_ac_trajectory(date)
        rl_decomp = decompose_is(rl_df[rl_df["fill_qty"] > 0], arrival)
        ac_decomp = decompose_is(ac_df[ac_df["fill_qty"] > 0], arrival)

        # RL's typical per-step chunk size (median of its nonzero actions) on this date.
        rl_nonzero = rl_df[(rl_df["fill_qty"] > 0) & (rl_df["action_frac"].notna())]
        rl_step_cap = float(rl_nonzero["action_frac"].median())

        hybrid_df, hybrid_info, cap_frac = collect_hybrid_trajectory(date, rl_step_cap)
        hybrid_decomp = decompose_is(hybrid_df[hybrid_df["fill_qty"] > 0], arrival)

        naive_estimate = ac_decomp["drift_pct"] + rl_decomp["execution_pct"]

        print(f"\n{date} ({REGIME[date]})  RL median nonzero step size={rl_step_cap:.4f} -> "
              f"hybrid cap bucket={cap_frac:.3f}")
        print(f"  {'policy':>14s} {'drift_pct':>10s} {'exec_pct':>10s} {'dust_pct':>10s} {'total':>10s}")
        print(f"  {'RL16 (pure)':>14s} {rl_decomp['drift_pct']:>+10.4f} {rl_decomp['execution_pct']:>+10.4f} "
              f"{rl_decomp['dust_pct']:>+10.4f} {rl_decomp['total_pct']:>+10.4f}")
        print(f"  {'AC (pure)':>14s} {ac_decomp['drift_pct']:>+10.4f} {ac_decomp['execution_pct']:>+10.4f} "
              f"{ac_decomp['dust_pct']:>+10.4f} {ac_decomp['total_pct']:>+10.4f}")
        print(f"  {'HYBRID (sim)':>14s} {hybrid_decomp['drift_pct']:>+10.4f} {hybrid_decomp['execution_pct']:>+10.4f} "
              f"{hybrid_decomp['dust_pct']:>+10.4f} {hybrid_decomp['total_pct']:>+10.4f}")
        print(f"  {'naive estimate (AC drift + RL exec)':>14s} = {naive_estimate:>+10.4f}")
        beats_ac_sim = hybrid_decomp["total_pct"] > ac_decomp["total_pct"]
        beats_ac_naive = naive_estimate > ac_decomp["total_pct"]
        print(f"  => simulated hybrid {'BEATS' if beats_ac_sim else 'DOES NOT BEAT'} tuned AC "
              f"(delta={hybrid_decomp['total_pct']-ac_decomp['total_pct']:+.4f}pp); "
              f"naive estimate {'BEATS' if beats_ac_naive else 'DOES NOT BEAT'} tuned AC "
              f"(delta={naive_estimate-ac_decomp['total_pct']:+.4f}pp)")

        summary.append({"date": date, "regime": REGIME[date], "rl_step_cap": rl_step_cap,
                         **{f"rl_{k}": v for k, v in rl_decomp.items()},
                         **{f"ac_{k}": v for k, v in ac_decomp.items()},
                         **{f"hybrid_{k}": v for k, v in hybrid_decomp.items()},
                         "naive_estimate_total": naive_estimate})
    out = pd.DataFrame(summary)
    out.to_csv("rl/results/phase2r_urgency_hybrid.csv", index=False)
    print(f"\nSaved rl/results/phase2r_urgency_hybrid.csv")
    return {row["date"]: row for row in summary}, model


# ---------------------------------------------------------------------------
# Task B: is regime information present & discriminative early in the episode?
# ---------------------------------------------------------------------------

def task_b(model):
    print(f"\n{'#' * 90}\nTASK B: is regime info present & discriminative early in the episode?\n{'#' * 90}")
    dims = {"s2": "vol_ratio tanh(sigma/sigma_ref)", "s10": "vol_regime (discrete 0/.5/1)",
            "s11": "liq_regime (discrete 0/.5/1)", "s13": "intraday_vol_ratio tanh(sigma/sigma_ref - 1)"}
    all_states = {}
    for date in ["2024-08-05", "2024-03-05", "2024-07-10", "2024-12-10"]:
        _, states, _ = collect_rl_trajectory_with_state(model, date)
        all_states[date] = states

    windows = [(0, 1, "step 0 (very first decision)"), (0, 10, "steps 0-9"),
               (0, 50, "steps 0-49"), (0, 100, "steps 0-99")]
    for lo, hi, label in windows:
        print(f"\n--- {label} ---")
        print(f"  {'date':>12s} {'regime':>15s}  " + "  ".join(f"{d:>8s}" for d in dims))
        for date in all_states:
            w = all_states[date][(all_states[date]["step"] >= lo) & (all_states[date]["step"] < hi)]
            vals = "  ".join(f"{w[d].mean():>8.4f}" for d in dims)
            print(f"  {date:>12s} {REGIME[date]:>15s}  {vals}")

    print(f"\n--- separation check: severe-crash (08-05) vs calm-mean, per window ---")
    for lo, hi, label in windows:
        row = {}
        for date in all_states:
            w = all_states[date][(all_states[date]["step"] >= lo) & (all_states[date]["step"] < hi)]
            row[date] = {d: w[d].mean() for d in dims}
        calm_mean = {d: np.mean([row["2024-07-10"][d], row["2024-12-10"][d]]) for d in dims}
        crash_val = row["2024-08-05"]
        diffs = {d: crash_val[d] - calm_mean[d] for d in dims}
        print(f"  {label:>28s}: " + "  ".join(f"{d}_diff={diffs[d]:+.4f}" for d in dims))

    all_df = pd.concat([df.assign(date=date, regime=REGIME[date]) for date, df in all_states.items()])
    all_df.to_csv("rl/results/phase2r_state_discriminability.csv", index=False)
    print(f"\nSaved rl/results/phase2r_state_discriminability.csv ({len(all_df)} rows)")


def main():
    print("Loading RL-16dim model...")
    model = load_model(MODEL_16DIM)
    task_a(model)
    task_b(model)


if __name__ == "__main__":
    main()
