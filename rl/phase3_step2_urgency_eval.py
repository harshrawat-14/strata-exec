"""
Phase 3 Step 2 evaluation: does RL's execution pace now vary with regime?

Step 0 repaired the volatility signal (s10 went from bit-identical across
crash and calm to cleanly severity-ordered). Step 2 retrains on that repaired
state. The question this script answers is the one Step 2 exists for:

    RL16 paced at ~the same rate regardless of regime, while tuned AC's pace
    swung sharply by regime ("RL learned execution but not urgency"). With a
    live regime signal, does the retrained policy's pace now respond?

PACE METRIC -- defined here explicitly rather than inherited, because the
earlier diagnosis reported pace informally and this needs to be reproducible:

    t50 / t90 = the fraction of the execution horizon elapsed at the moment
    cumulative sold inventory first reaches 50% / 90%.

    Lower  = more urgent (front-loaded).
    Higher = more patient (back-loaded).

A regime-responsive policy has a MEANINGFULLY LOWER t50 on crash dates than on
calm dates. A regime-blind one has roughly the same t50 everywhere -- which is
exactly what RL16 does and what this is testing for.

BINARY SELECTION IS LOAD-BEARING. Each model must be evaluated against the
binary whose state encoding it was trained on:
    RL16 (pre-Step-0)  -> STRATAEXEC_RL_ENV_BIN=<pre-Step-0 build>
    Step 2 (post-Step-0) -> default ./target/release/rl-env
Comparing implementation shortfall across the two is valid because nothing in
the counterfactual fill/impact/price path reads sigma_ref -- verified and
recorded in docs/proposals/research-roadmap.md, Step 2. Comparing PACE across them is even
safer, since pace is a function of actions alone.

Usage (run once per model, with the matching binary):
    rl/.venv/bin/python rl/phase3_step2_urgency_eval.py \
        --model rl/models/phase3_step2_16dim_v2_best/best_model --label step2
    STRATAEXEC_RL_ENV_BIN=/tmp/rl-env-prestep0 \
      rl/.venv/bin/python rl/phase3_step2_urgency_eval.py \
        --model rl/models/ppo_ablation_16dim_cf_best/best_model --label rl16
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from environment import StrataExecEnv, RL_ENV_BIN, REGIME_DIMS_ARE_CONTINUOUS
from evaluate import (
    load_model,
    ACOptimalPolicy,
    TWAPPolicy,
    AC_DEFAULT_LAMBDA,
    SEVERITY_RANK_V2,
    HELD_OUT_DATES_V2,
)

HONEST_LAMBDA = 1e-6
TOTAL_STEPS = 2880
BTC_TARGET = 50_000.0


def pace_metrics(sold_fracs: list[float], horizon: int) -> dict:
    """t50/t90 from a cumulative-sold-fraction trace, plus completion."""
    cum = np.asarray(sold_fracs, dtype=float)
    out = {}
    for pct, key in ((0.50, "t50"), (0.90, "t90")):
        hits = np.nonzero(cum >= pct)[0]
        # NaN (not 1.0) when the level is never reached: a policy that never
        # sells 90% has no t90, and coding that as "the very end" would make a
        # failure look like maximal patience.
        out[key] = float(hits[0] / horizon) if len(hits) else float("nan")
    out["final_sold_frac"] = float(cum[-1]) if len(cum) else 0.0
    return out


def roll(policy, date: str, seed: int, is_rl: bool) -> dict:
    env = StrataExecEnv(
        mode="counterfactual", lob_date=date, agg_date=date,
        btc_target=BTC_TARGET, n_state_dims=16, total_steps=TOTAL_STEPS,
        fixed_steps=True, fixed_size=True, no_time_features=False,
    )
    try:
        obs, info = env.reset(seed=seed)
        total_inv = info.get("total_inventory", 1_000_000.0)
        lstm_state, ep_start = None, np.ones((1,), dtype=bool)
        cum, sold, done, steps = [], 0.0, False, 0
        while not done:
            if is_rl:
                action, lstm_state = policy.predict(
                    obs, state=lstm_state, episode_start=ep_start, deterministic=True
                )
                ep_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = policy.predict(obs)
            obs, _r, done, _t, info = env.step(int(action))
            sold += float(info.get("fill_qty", 0.0))
            cum.append(min(sold / max(total_inv, 1e-9), 1.0))
            steps += 1
        m = pace_metrics(cum, steps)
        m["total_IS_pct"] = float(info.get("total_IS_pct", float("nan")))
        return m
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dates", type=str, default=None,
                    help="comma-separated; default = V2 held-out dates")
    args = ap.parse_args()

    dates = ([d.strip() for d in args.dates.split(",")] if args.dates
             else list(HELD_OUT_DATES_V2))

    print(f"binary   : {RL_ENV_BIN}")
    print(f"continuous regime dims: {REGIME_DIMS_ARE_CONTINUOUS}")
    print(f"model    : {args.model}  (label={args.label})")
    print(f"dates    : {len(dates)} V2 held-out\n")

    rl = load_model(args.model)
    rows = []
    for date in dates:
        regime = SEVERITY_RANK_V2.get(date, (0, "unknown", ""))[1]
        cohort = {
            f"rl_{args.label}": (rl, True),
            "ac_optimal": (ACOptimalPolicy(n_state_dims=16, lam=HONEST_LAMBDA,
                                           total_steps=TOTAL_STEPS), False),
            "twap": (TWAPPolicy(n_state_dims=16), False),
        }
        for name, (pol, is_rl) in cohort.items():
            for seed in range(args.seeds):
                m = roll(pol, date, seed, is_rl)
                rows.append({"date": date, "regime": regime,
                             "policy": name, "seed": seed, **m})
        print(f"  done {date} ({regime})")

    df = pd.DataFrame(rows)
    out = f"rl/results/phase3_step2_urgency_{args.label}.csv"
    df.to_csv(out, index=False)

    print(f"\n=== pace by regime (mean t50 = fraction of horizon to 50% sold) ===")
    piv = df.groupby(["policy", "regime"])["t50"].mean().unstack()
    print(piv.round(4).to_string())
    if {"crash", "calm"}.issubset(piv.columns):
        print("\n=== regime responsiveness: t50(calm) - t50(crash) ===")
        print("    positive = front-loads on crashes (responsive); ~0 = regime-blind")
        for pol in piv.index:
            print(f"    {pol:20} {piv.loc[pol,'calm'] - piv.loc[pol,'crash']:+.4f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
