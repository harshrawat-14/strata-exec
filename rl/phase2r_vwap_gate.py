"""
Phase 2R VWAP-benchmark analytic gate.

Question: does replacing the static arrival-price benchmark with a DRIFT-NEUTRAL
interval benchmark remove the directional-speculation incentive that makes
blanket-wait beat blanket-execute (docs/experiments/02-reward-objective-audit.md, Table 3)?

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
      twap        = interval TWAP (equal time weight) of the undisturbed market mid
      vwap        = interval VWAP (real aggTrades volume weight) of the same mid path
      horizon     = number of steps in the window (for scheduling TWAP execution)
      wait_avg_ex = avg execution price of the wait policy (all inventory dumped
                    at the horizon via forced liquidation)
    """
    obs, info = env.reset(seed=seed)
    mids, vols = [], []
    done = False
    steps = 0
    while not done:
        obs, _, done, _, info = env.step(0)
        m = info.get("book_mid")
        v = info.get("interval_volume", 0.0)
        if m is not None and m > 0:
            mids.append(m)
            vols.append(max(v, 0.0))
        steps += 1
    mids = np.array(mids)
    vols = np.array(vols)
    twap = float(mids.mean()) if len(mids) else 100.0
    vwap = float((mids * vols).sum() / vols.sum()) if vols.sum() > 0 else twap
    wait_avg_ex = 100.0 - info.get("total_IS_pct")
    return twap, vwap, steps, wait_avg_ex


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
    """Per-date means vs three benchmarks: arrival, TWAP, VWAP; for WAIT and EXEC."""
    env = _mk_env(date)
    acc = {k: [] for k in ("wA", "eA", "wT", "eT", "wV", "eV")}
    for seed in range(N_SEEDS):
        twap, vwap, horizon, wait_avg = wait_pass(env, seed)
        exec_avg = twap_exec_scheduled(env, seed, horizon)
        acc["wA"].append(wait_avg - 100.0)
        acc["eA"].append(exec_avg - 100.0)
        acc["wT"].append((wait_avg - twap) / twap * 100.0)
        acc["eT"].append((exec_avg - twap) / twap * 100.0)
        acc["wV"].append((wait_avg - vwap) / vwap * 100.0)
        acc["eV"].append((exec_avg - vwap) / vwap * 100.0)
    env.close()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    print("=" * 104)
    print("PHASE 2R VWAP GATE -- blanket-WAIT vs genuine TWAP-execution (equal qty/step)")
    print("benchmarks: static-ARRIVAL (current) | interval-TWAP | interval-VWAP (real aggTrades volume)")
    print("IS convention: signed vs benchmark; <0 = underperformed benchmark (a cost)")
    print("=" * 104)
    print(f"{'date':12s} {'arrWAIT':>8s} {'arrEXEC':>8s} | {'twapWAIT':>9s} {'twapEXEC':>9s} | "
          f"{'vwapWAIT':>9s} {'vwapEXEC':>9s}  win(vwap)")

    rows = {d: score_date(d) for d in TRAINING_DATES + CRASH_DATES}

    agg = {k: [] for k in ("wA", "eA", "wT", "eT", "wV", "eV")}
    for date in TRAINING_DATES:
        r = rows[date]
        for k in agg:
            agg[k].append(r[k])
        win = "EXEC" if r["eV"] > r["wV"] else "WAIT"
        print(f"{date:12s} {r['wA']:+8.3f} {r['eA']:+8.3f} | {r['wT']:+9.3f} {r['eT']:+9.3f} | "
              f"{r['wV']:+9.3f} {r['eV']:+9.3f}  {win}")

    m = {k: float(np.mean(v)) for k, v in agg.items()}
    print("-" * 104)
    print(f"{'E[training]':12s} {m['wA']:+8.3f} {m['eA']:+8.3f} | {m['wT']:+9.3f} {m['eT']:+9.3f} | "
          f"{m['wV']:+9.3f} {m['eV']:+9.3f}")
    print()
    print("Q1 -- does WAIT still have POSITIVE expected value (speculation rewarded)?")
    for label, wk in (("ARRIVAL", "wA"), ("TWAP", "wT"), ("VWAP", "wV")):
        verdict = "POSITIVE -> speculation pays" if m[wk] > 0 else "NEGATIVE -> speculation no longer pays"
        print(f"  {label:8s}: E[WAIT]={m[wk]:+.4f}%   {verdict}")
    print()
    print("Q2 -- does competent EXECUTION beat WAIT in expectation, and by how much?")
    for label, wk, ek in (("ARRIVAL", "wA", "eA"), ("TWAP", "wT", "eT"), ("VWAP", "wV", "eV")):
        d = m[ek] - m[wk]
        print(f"  {label:8s}: E[EXEC]-E[WAIT]={d:+.4f}pp   ({'EXEC wins' if d > 0 else 'WAIT wins'})")

    print("\n--- crash dates under each benchmark ---")
    for date in CRASH_DATES:
        r = rows[date]
        print(f"{date}: arr W={r['wA']:+.3f} E={r['eA']:+.3f} | twap W={r['wT']:+.3f} E={r['eT']:+.3f} | "
              f"vwap W={r['wV']:+.3f} E={r['eV']:+.3f}")

    # Per-date EXEC-vs-WAIT verdict under VWAP, for the conditional-learnability analysis.
    print("\n--- per-date VWAP verdict (feeds the conditional-signal analysis) ---")
    for date in TRAINING_DATES + CRASH_DATES:
        r = rows[date]
        print(f"  {date}: {'EXEC' if r['eV'] > r['wV'] else 'WAIT'}  (margin {r['eV']-r['wV']:+.3f}pp)")


if __name__ == "__main__":
    main()
