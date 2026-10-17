"""
Phase 2R objective-bias analysis: the reproduction script behind
docs/experiments/02-reward-objective-audit.md.

Produces the three tables in that report:
  1. trained-policy action distribution / forced-liq / IS per date
  2. H1: best achievable execution vs pure-wait on the two crash dates
  3. H2: expected terminal IS across the 7 training dates, blanket-wait vs
     best-constant-execute (the decisive "the agent is correctly optimizing a
     directionally-biased objective" table)

Run: rl/.venv/bin/python rl/phase2r_objective_analysis.py
"""
from __future__ import annotations

import numpy as np
from collections import Counter

from evaluate import load_model
from environment import StrataExecEnv

MODEL_PATH = "rl/models/ppo_lstm_v6r_time_features_best/best_model"
TRAINING_DATES = ["2024-01-15", "2024-02-20", "2024-04-15", "2024-06-10",
                  "2024-09-15", "2024-10-15", "2024-11-06"]
HELD_OUT_CRASH = ["2024-08-05", "2024-03-05"]
# constant fraction-of-remaining action indices spanning the action space
EXEC_ACTIONS = [7, 9, 10, 11, 12, 14, 16]


def _mk_env(date):
    return StrataExecEnv(mode="historical", adversarial=False, lob_date=date,
                         agg_date=date, n_state_dims=16, fixed_steps=True,
                         fixed_size=True, no_time_features=False)


def run_const(date, action, n_seeds=10):
    env = _mk_env(date)
    vals, forced = [], 0
    for seed in range(n_seeds):
        _, info = env.reset(seed=seed)
        done = False
        obs = None
        while not done:
            obs, _, done, _, info = env.step(action)
        if info.get("forced_liquidation_qty", 0) > 0:
            forced += 1
        vals.append(-info.get("total_IS_pct"))
    env.close()
    return float(np.mean(vals)), forced


def run_model(model, date, n_seeds=20):
    env = _mk_env(date)
    forced, actions, is_vals = 0, [], []
    for seed in range(n_seeds):
        obs, info = env.reset(seed=seed)
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        done = False
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            episode_start = np.zeros((1,), dtype=bool)
            actions.append(int(action))
            obs, _, done, _, info = env.step(int(action))
        if info.get("forced_liquidation_qty", 0) > 0:
            forced += 1
        is_vals.append(-info.get("total_IS_pct"))
    env.close()
    c = Counter(actions)
    return c.get(0, 0) / sum(c.values()), forced, float(np.mean(is_vals)), n_seeds


def main():
    model = load_model(MODEL_PATH)

    print("=" * 78)
    print("TABLE 1 -- trained policy (ppo_lstm_v6r) per date")
    print("=" * 78)
    for date in TRAINING_DATES + HELD_OUT_CRASH:
        z, f, mis, n = run_model(model, date)
        tag = "TRAIN" if date in TRAINING_DATES else "HELD-OUT"
        print(f"  {date} [{tag:8s}]: wait_frac={z:.3f}  forced={f}/{n}  IS={mis:+.4f}%")

    print("\n" + "=" * 78)
    print("TABLE 2 -- H1: best executable vs pure-wait on crash dates")
    print("=" * 78)
    for date in HELD_OUT_CRASH:
        wait, _ = run_const(date, 0)
        execs = {a: run_const(date, a)[0] for a in EXEC_ACTIONS}
        best_a = max(execs, key=execs.get)
        print(f"  {date}: best_exec(a{best_a})={execs[best_a]:+.4f}%  "
              f"wait={wait:+.4f}%  advantage={execs[best_a]-wait:+.4f}pp")

    print("\n" + "=" * 78)
    print("TABLE 3 -- H2: expected IS across training mix, wait vs execute")
    print("=" * 78)
    wait_all, exec_all = [], []
    for date in TRAINING_DATES:
        wait, _ = run_const(date, 0)
        execs = {a: run_const(date, a)[0] for a in EXEC_ACTIONS}
        best_a = max(execs, key=execs.get)
        wait_all.append(wait)
        exec_all.append(execs[best_a])
        winner = "WAIT" if wait > execs[best_a] else "EXEC"
        print(f"  {date}: wait={wait:+.4f}%  best_exec(a{best_a})={execs[best_a]:+.4f}%  -> {winner}")
    print(f"\n  EXPECTED (equal-weight): always-WAIT={np.mean(wait_all):+.4f}%  "
          f"always-EXECUTE={np.mean(exec_all):+.4f}%")
    print(f"  --> {'WAIT' if np.mean(wait_all) > np.mean(exec_all) else 'EXECUTE'} "
          f"maximizes expected reward across the training distribution")


if __name__ == "__main__":
    main()
