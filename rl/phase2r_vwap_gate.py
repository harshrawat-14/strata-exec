"""
Phase 2R VWAP-benchmark analytic gate.

Question: does replacing the static arrival-price benchmark with a DRIFT-NEUTRAL
interval benchmark remove the directional-speculation incentive that makes
blanket-wait beat blanket-execute (docs/phase2r_objective_bias_finding.md, Table 3)?

Method (cheap analytic gate, no training):
  - benchmark = interval TWAP of the UNDISTURBED market mid over the execution
    window, captured from a wait episode (action 0 -> no agent impact, so the
    book_mid path is the clean market path). TWAP is used as the drift-neutral
    benchmark proxy: like VWAP it averages over the window, so it rises with a
    rally and falls with a crash -- the property that matters here. (If a result
    is borderline, refine to true volume-weighting via aggTrades.)
  - avg_exec_price for any policy = 100 - raw_total_IS_pct  (sim units, arrival
    fixed at 100 by the env; raw total_IS_pct is (ideal-received)/ideal*100).
  - Two IS conventions reported per policy, negative = underperformed benchmark:
      IS_arrival = -(raw_total_IS_pct)            # the current objective
      IS_twap    = (avg_exec - twap)/twap * 100   # drift-neutral benchmark

Run: rl/.venv/bin/python rl/phase2r_vwap_gate.py
"""
from __future__ import annotations

import numpy as np

from environment import StrataExecEnv, ACTION_FRACTIONS

TRAINING_DATES = ["2024-01-15", "2024-02-20", "2024-04-15", "2024-06-10",
                  "2024-09-15", "2024-10-15", "2024-11-06"]
CRASH_DATES = ["2024-08-05", "2024-03-05"]
EXEC_ACTIONS = [7, 9, 10, 11, 12, 14, 16]
N_SEEDS = 10


def _mk_env(date):
    return StrataExecEnv(mode="historical", adversarial=False, lob_date=date,
                         agg_date=date, n_state_dims=16, fixed_steps=True,
                         fixed_size=True, no_time_features=False)


def wait_pass(env, seed):
    """
    Wait episode (action 0 -> no agent impact). Returns:
      twap        = interval TWAP of the undisturbed market mid over the window
      horizon     = number of steps in the window (for scheduling TWAP execution)
      wait_avg_ex = avg execution price of the wait policy (all inventory dumped
                    at the horizon via forced liquidation)
    """
    obs, info = env.reset(seed=seed)
    mids = []
    done = False
    steps = 0
    while not done:
        obs, _, done, _, info = env.step(0)
        m = info.get("book_mid")
        if m is not None and m > 0:
            mids.append(m)
        steps += 1
    twap = float(np.mean(mids)) if mids else 100.0
    wait_avg_ex = 100.0 - info.get("total_IS_pct")
    return twap, steps, wait_avg_ex


def _nearest_action(target_frac):
    return int(np.argmin([abs(f - target_frac) for f in ACTION_FRACTIONS]))


def twap_exec_scheduled(env, seed, horizon):
    """Equal-quantity-per-step TWAP execution given the known episode horizon.
    Achieves avg_exec ~ interval TWAP by construction (up to spread/impact)."""
    obs, info = env.reset(seed=seed)
    done = False
    step = 0
    while not done:
        steps_left = max(horizon - step, 1)
        target = 1.0 / steps_left  # fraction of remaining -> ~equal qty/step
        obs, _, done, _, info = env.step(_nearest_action(target))
        step += 1
    return 100.0 - info.get("total_IS_pct")


def score_date(date):
    """Return per-date means: (wait_arr, exec_arr, wait_twap, exec_twap)."""
    env = _mk_env(date)
    wait_arr, exec_arr, wait_tw, exec_tw = [], [], [], []
    for seed in range(N_SEEDS):
        twap, horizon, wait_avg = wait_pass(env, seed)
        exec_avg = twap_exec_scheduled(env, seed, horizon)
        wait_arr.append((wait_avg - 100.0))
        exec_arr.append((exec_avg - 100.0))
        wait_tw.append((wait_avg - twap) / twap * 100.0)
        exec_tw.append((exec_avg - twap) / twap * 100.0)
    env.close()
    return (float(np.mean(wait_arr)), float(np.mean(exec_arr)),
            float(np.mean(wait_tw)), float(np.mean(exec_tw)))


def main():
    print("=" * 96)
    print("PHASE 2R VWAP GATE -- blanket-WAIT vs genuine TWAP-execution (equal qty/step)")
    print("benchmarks: static-arrival (current objective) vs interval-TWAP (drift-neutral, proposed)")
    print("IS convention: signed vs benchmark; <0 = underperformed benchmark (a cost)")
    print("=" * 96)
    print(f"{'date':12s} {'arr:WAIT':>9s} {'arr:EXEC':>9s} {'|':>2s} "
          f"{'twap:WAIT':>10s} {'twap:EXEC':>10s}  winner(twap)")

    rows = {}
    for date in TRAINING_DATES + CRASH_DATES:
        rows[date] = score_date(date)

    wA, eA, wT, eT = [], [], [], []
    for date in TRAINING_DATES:
        w_arr, e_arr, w_tw, e_tw = rows[date]
        wA.append(w_arr); eA.append(e_arr); wT.append(w_tw); eT.append(e_tw)
        win = "EXEC" if e_tw > w_tw else "WAIT"
        print(f"{date:12s} {w_arr:+9.4f} {e_arr:+9.4f} {'|':>2s} {w_tw:+10.4f} {e_tw:+10.4f}  {win}")

    print("-" * 96)
    print(f"{'E[training]':12s} {np.mean(wA):+9.4f} {np.mean(eA):+9.4f} {'|':>2s} "
          f"{np.mean(wT):+10.4f} {np.mean(eT):+10.4f}")
    print()
    print("KEY QUESTION 1 -- does WAIT have POSITIVE expected value (i.e. is speculation rewarded)?")
    print(f"  under STATIC-ARRIVAL: E[WAIT]={np.mean(wA):+.4f}%   "
          f"({'POSITIVE -> speculation pays, objective BROKEN' if np.mean(wA) > 0 else 'negative'})")
    print(f"  under DRIFT-NEUTRAL:  E[WAIT]={np.mean(wT):+.4f}%   "
          f"({'still positive -> NOT fixed' if np.mean(wT) > 0 else 'NEGATIVE -> speculation no longer pays'})")
    print()
    print("KEY QUESTION 2 -- does competent EXECUTION beat WAIT in expectation?")
    print(f"  under STATIC-ARRIVAL: E[EXEC]-E[WAIT]={np.mean(eA)-np.mean(wA):+.4f}pp   "
          f"({'EXEC wins' if np.mean(eA) > np.mean(wA) else 'WAIT wins'})")
    print(f"  under DRIFT-NEUTRAL:  E[EXEC]-E[WAIT]={np.mean(eT)-np.mean(wT):+.4f}pp   "
          f"({'EXEC wins' if np.mean(eT) > np.mean(wT) else 'WAIT wins'})")

    print("\n--- crash dates (execution should clearly beat waiting under the drift-neutral benchmark) ---")
    for date in CRASH_DATES:
        w_arr, e_arr, w_tw, e_tw = rows[date]
        print(f"{date}: arr WAIT={w_arr:+.4f} EXEC={e_arr:+.4f} | twap WAIT={w_tw:+.4f} EXEC={e_tw:+.4f}")


if __name__ == "__main__":
    main()
