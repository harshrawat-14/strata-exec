"""
Phase 2R close-out gate follow-up (greenlit, corrected scope): granularity is back on
the table after item 1's own extended grid retracted Gate A's "second-order" verdict.

Investigates the specific chattering mechanism: AC's continuous intent survives
20-bucket rounding because per-step rounding errors partly average out over 2880
steps; RL, learning natively in bucket space with no schedule prior, may not find a
smooth trajectory and instead chatter between the zero bucket and an overshoot
bucket -- especially where the smallest usable step size is large relative to what a
patient schedule wants (thin/crash-day books). Also runs the crash-date IS
decomposition (drift/"adverse holding" vs execution/"spread+impact") the same
message asked for.

Usage: rl/.venv/bin/python rl/phase2r_granularity_investigation.py
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import StrataExecEnv, ACTION_FRACTIONS
from evaluate import load_model, ACOptimalPolicy, nearest_action, get_action_fractions

MODEL_16DIM = "rl/models/ppo_ablation_16dim_cf_best/best_model"
BTC_TARGET = 50_000.0
SEED = 0  # single representative trajectory per date/policy, fixed for apples-to-apples comparison

# Best AC lambda per date, from the greenlit item-1 full grid (rl/results/phase2r_lambda_grid_full.csv).
BEST_AC_LAMBDA = {
    "2024-03-05": 1e-4,   # crash (flash cascade) -- RL's one win
    "2024-08-05": 1e-3,   # crash (Yen unwind) -- RL's worst loss
    "2024-07-10": 1e-6,   # calm
    "2024-12-10": 1e-6,   # calm
}
REGIME = {
    "2024-03-05": "crash",
    "2024-08-05": "crash",
    "2024-07-10": "calm",
    "2024-12-10": "calm",
}

TRIVIAL_ACTION_IDX = ACTION_FRACTIONS.index(0.100)  # 10%/step trivial policy, action index 11


def _mk_env(date, n_state_dims=16):
    return StrataExecEnv(
        mode="counterfactual", lob_date=date, agg_date=date, btc_target=BTC_TARGET,
        n_state_dims=n_state_dims, fixed_steps=True, fixed_size=True,
        no_time_features=(n_state_dims < 16),
    )


def collect_rl_trajectory(model, date):
    env = _mk_env(date, n_state_dims=16)
    obs, info = env.reset(seed=SEED)
    arrival_price = 100.0
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    rows = []
    done = False
    step = 0
    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start, deterministic=True,
        )
        episode_start = np.zeros((1,), dtype=bool)
        action_idx = int(action)
        obs, reward, done, _, info = env.step(action_idx)
        rows.append({
            "step": step, "action_idx": action_idx,
            "action_frac": ACTION_FRACTIONS[action_idx],
            "fill_qty": info.get("fill_qty", 0.0), "fill_price": info.get("fill_price", 0.0),
            "book_mid": info.get("book_mid", np.nan), "remaining": info.get("remaining", np.nan),
        })
        step += 1
    _attach_forced_liq(rows, info, arrival_price)
    return pd.DataFrame(rows), info, arrival_price


def collect_policy_trajectory(policy, date, n_state_dims=16):
    env = _mk_env(date, n_state_dims=n_state_dims)
    obs, info = env.reset(seed=SEED)
    arrival_price = 100.0
    rows = []
    done = False
    step = 0
    while not done:
        action, _ = policy.predict(obs)
        action_idx = int(action[0]) if hasattr(action, "__len__") else int(action)
        obs, reward, done, _, info = env.step(action_idx)
        rows.append({
            "step": step, "action_idx": action_idx,
            "action_frac": ACTION_FRACTIONS[action_idx],
            "fill_qty": info.get("fill_qty", 0.0), "fill_price": info.get("fill_price", 0.0),
            "book_mid": info.get("book_mid", np.nan), "remaining": info.get("remaining", np.nan),
        })
        step += 1
    _attach_forced_liq(rows, info, arrival_price)
    return pd.DataFrame(rows), info, arrival_price


def collect_trivial_trajectory(date, n_state_dims=16):
    env = _mk_env(date, n_state_dims=n_state_dims)
    obs, info = env.reset(seed=SEED)
    arrival_price = 100.0
    rows = []
    done = False
    step = 0
    while not done:
        obs, reward, done, _, info = env.step(TRIVIAL_ACTION_IDX)
        rows.append({
            "step": step, "action_idx": TRIVIAL_ACTION_IDX,
            "action_frac": ACTION_FRACTIONS[TRIVIAL_ACTION_IDX],
            "fill_qty": info.get("fill_qty", 0.0), "fill_price": info.get("fill_price", 0.0),
            "book_mid": info.get("book_mid", np.nan), "remaining": info.get("remaining", np.nan),
        })
        step += 1
    _attach_forced_liq(rows, info, arrival_price)
    return pd.DataFrame(rows), info, arrival_price


def _attach_forced_liq(rows, info, arrival_price):
    """If the episode ended in forced liquidation, add one synthetic 'fill' row for it
    so the IS decomposition below accounts for the whole executed notional, not just
    voluntary fills."""
    fl_qty = info.get("forced_liquidation_qty", 0.0)
    if fl_qty and fl_qty > 0:
        filled_notional = info.get("forced_liq_filled_notional", 0.0)
        punitive_notional = info.get("forced_liq_punitive_notional", 0.0)
        total_notional = filled_notional + punitive_notional
        eff_price = total_notional / fl_qty if fl_qty > 1e-12 else info.get("book_mid", arrival_price)
        rows.append({
            "step": rows[-1]["step"] + 1 if rows else 0, "action_idx": -1, "action_frac": None,
            "fill_qty": fl_qty, "fill_price": eff_price,
            "book_mid": info.get("book_mid", arrival_price), "remaining": 0.0,
            "forced_liquidation": True,
        })


def decompose_is(df: pd.DataFrame, arrival_price: float, total_inventory: float = 1_000_000.0):
    """
    Split the reported IS (received-vs-ideal, positive=good, matching mean_IS's sign
    convention used throughout this phase's tables) into:
      drift_pct      = price movement between arrival and each fill's book_mid ("adverse
                        holding" -- cost/benefit purely from when you happened to trade)
      execution_pct  = fill_price vs book_mid at fill time ("spread + impact" -- cost of
                        how you crossed the book given the price at that moment)
      dust_pct       = inventory left unsold when the episode ends via the voluntary
                        `inv_done` path (remaining <= 0.1% of total, no forced
                        liquidation triggered) -- silently excluded from `received`
                        entirely, so it costs the full arrival price on that sliver.
                        Zero whenever the episode instead ends via forced liquidation
                        (that path is already captured as an ordinary fill row).
    The three terms reconstruct the environment's own reported total_IS_pct exactly
    (verified against `info["total_IS_pct"]` at each call site, not assumed).
    """
    fills = df[df["fill_qty"] > 0].copy()
    ideal = arrival_price * total_inventory
    if fills.empty:
        return {"drift_pct": 0.0, "execution_pct": 0.0, "dust_pct": -100.0, "total_pct": -100.0, "n_fills": 0}
    drift_dollar = (fills["fill_qty"] * (fills["book_mid"] - arrival_price)).sum()
    exec_dollar = (fills["fill_qty"] * (fills["fill_price"] - fills["book_mid"])).sum()
    executed_qty = fills["fill_qty"].sum()
    dust_qty = max(total_inventory - executed_qty, 0.0)
    dust_dollar = -dust_qty * arrival_price  # priced at zero received -- full cost
    return {
        "drift_pct": drift_dollar / ideal * 100,
        "execution_pct": exec_dollar / ideal * 100,
        "dust_pct": dust_dollar / ideal * 100,
        "total_pct": (drift_dollar + exec_dollar + dust_dollar) / ideal * 100,
        "n_fills": len(fills),
    }


def chattering_metrics(df: pd.DataFrame):
    """Quantify how 'jagged' a trajectory is: trade frequency, isolated-burst rate
    (0 -> trade -> 0 pattern), and dispersion of the per-fill fraction traded."""
    acts = df["action_idx"].values
    fills = df[df["fill_qty"] > 0]
    n_steps = len(df)
    n_trades = len(fills)
    # isolated burst: a nonzero action immediately preceded AND followed by zero action
    isolated = 0
    for i in range(1, len(acts) - 1):
        if acts[i] > 0 and acts[i - 1] == 0 and acts[i + 1] == 0:
            isolated += 1
    frac_isolated = isolated / max(n_trades, 1)
    per_fill_slippage_bps = (
        (fills["book_mid"] - fills["fill_price"]) / fills["book_mid"] * 10_000
    ) if n_trades else pd.Series([], dtype=float)
    return {
        "n_steps": n_steps, "n_trades": n_trades,
        "trade_frequency": n_trades / n_steps if n_steps else 0.0,
        "frac_isolated_bursts": frac_isolated,
        "mean_slippage_bps": per_fill_slippage_bps.mean() if n_trades else 0.0,
        "median_slippage_bps": per_fill_slippage_bps.median() if n_trades else 0.0,
        "max_slippage_bps": per_fill_slippage_bps.max() if n_trades else 0.0,
        "std_action_frac_nonzero": fills["action_frac"].std() if n_trades > 1 else 0.0,
    }


def plot_trajectories(date, rl_df, ac_df, tw_df, outpath, total_inventory=1_000_000.0):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for df, label, color in [(rl_df, "RL-16dim", "tab:red"),
                              (ac_df, f"AC (best λ)", "tab:blue"),
                              (tw_df, "TWAP 10%/step", "tab:gray")]:
        axes[0].plot(df["step"], 1.0 - df["remaining"] / total_inventory, label=label, color=color, lw=1.3)
        axes[1].plot(df["step"], df["action_frac"].fillna(0.0), label=label, color=color,
                     lw=0.8, alpha=0.8)
    axes[0].set_ylabel("cumulative fraction executed")
    axes[0].set_title(f"{date} -- execution trajectory (seed={SEED})")
    axes[0].legend()
    axes[1].set_ylabel("per-step action fraction")
    axes[1].set_xlabel("step")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)


def main():
    print("ACTION_FRACTIONS (single source of truth, from Rust):")
    print(json.dumps(ACTION_FRACTIONS))
    print(f"Smallest nonzero bucket: {ACTION_FRACTIONS[1]}  |  "
          f"largest gap: {max(b-a for a,b in zip(ACTION_FRACTIONS, ACTION_FRACTIONS[1:])):.3f} "
          f"(between {[ (a,b) for a,b in zip(ACTION_FRACTIONS, ACTION_FRACTIONS[1:]) if b-a > 0.15 ]})")

    print("\nLoading RL-16dim model...")
    model = load_model(MODEL_16DIM)

    summary_rows = []
    import os
    os.makedirs("rl/results/plots", exist_ok=True)

    for date in ["2024-03-05", "2024-08-05", "2024-07-10", "2024-12-10"]:
        print(f"\n{'=' * 90}\n{date} ({REGIME[date]})\n{'=' * 90}")
        rl_df, rl_info, arrival = collect_rl_trajectory(model, date)
        ac_policy = ACOptimalPolicy(n_state_dims=16, lam=BEST_AC_LAMBDA[date])
        ac_df, ac_info, _ = collect_policy_trajectory(ac_policy, date, n_state_dims=16)
        tw_df, tw_info, _ = collect_trivial_trajectory(date, n_state_dims=16)

        plot_path = f"rl/results/plots/granularity_{date}.png"
        plot_trajectories(date, rl_df, ac_df, tw_df, plot_path)
        print(f"Saved trajectory plot: {plot_path}")

        for label, df, info in [("RL16", rl_df, rl_info), ("AC_best", ac_df, ac_info),
                                 ("TWAP10pct", tw_df, tw_info)]:
            chat = chattering_metrics(df)
            decomp = decompose_is(df, arrival)
            reported_is = -info.get("total_IS_pct", -decomp["total_pct"])
            row = {"date": date, "regime": REGIME[date], "policy": label,
                   "reported_mean_IS": reported_is, **chat, **decomp}
            summary_rows.append(row)
            print(f"  {label:>10s}: trades={chat['n_trades']:4d} "
                  f"trade_freq={chat['trade_frequency']:.4f} "
                  f"isolated_burst_frac={chat['frac_isolated_bursts']:.3f} "
                  f"mean_slip_bps={chat['mean_slippage_bps']:7.2f} "
                  f"max_slip_bps={chat['max_slippage_bps']:8.2f} | "
                  f"drift_pct={decomp['drift_pct']:+.4f} exec_pct={decomp['execution_pct']:+.4f} "
                  f"dust_pct={decomp['dust_pct']:+.4f} "
                  f"decomp_total={decomp['total_pct']:+.4f} vs reported_IS={reported_is:+.4f}")

    out = pd.DataFrame(summary_rows)
    out.to_csv("rl/results/phase2r_granularity_investigation.csv", index=False)
    print(f"\nSaved rl/results/phase2r_granularity_investigation.csv ({len(out)} rows)")

    print(f"\n{'=' * 90}\nCALM vs CRASH comparison (RL16 only, tests the severity prediction)\n{'=' * 90}")
    rl_only = out[out["policy"] == "RL16"]
    for regime in ["calm", "crash"]:
        sub = rl_only[rl_only["regime"] == regime]
        print(f"  {regime:>6s}: mean trade_freq={sub['trade_frequency'].mean():.4f}  "
              f"mean isolated_burst_frac={sub['frac_isolated_bursts'].mean():.3f}  "
              f"mean slip_bps={sub['mean_slippage_bps'].mean():.2f}  "
              f"max slip_bps={sub['max_slippage_bps'].max():.2f}")


if __name__ == "__main__":
    main()
