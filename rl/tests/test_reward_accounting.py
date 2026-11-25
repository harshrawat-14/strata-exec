"""
Doc 37 audit: the counterfactual reward's accounting, on real episodes.

Every sale splits exactly into market move (arrival - mid without our impact), our impact
(that mid - the impacted mid) and spread/book walk (impacted mid - fill price), so
  market + impact + spread == IS                     (all as % of the order's arrival value)
and with --reward-baseline mid the fill rewards are exactly the impact + spread part:
  100 * sum(fill rewards) == impact + spread
Pace 3 finishes inside the episode with no forced dump or dust, so both hold to rounding.
Needs the Bybit data and the rl-env binary; skipped without them.
"""
import os

import numpy as np
import pytest

from environment import RL_ENV_BIN, StrataExecEnv
from evaluation.rate import RATES, RateAgent
from market_data import has_date

DATE = "2024-04-13"
pytestmark = pytest.mark.skipif(not (has_date(DATE) and os.path.exists(RL_ENV_BIN)),
                                reason="needs Bybit data for 2024-04-13 and the rl-env binary")


class _Fixed:
    def __init__(self, pace):
        self.a = RATES.index(pace)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.array([self.a]), state


def _episode(pace, baseline=None):
    kw = dict(reward_baseline=baseline) if baseline else {}
    env = StrataExecEnv(mode="counterfactual", lob_date=DATE, agg_date=DATE, order_share_adv=0.05,
                        n_state_dims=16, fixed_steps=True, fixed_size=True, episode_steps=600, **kw)
    try:
        obs, info = env.reset(seed=0, options={"start_frac": 0.25})
        agent = RateAgent(_Fixed(pace))
        agent.set_horizon(info["episode_horizon"])
        arrival, total = info["arrival_price"], info["total_inventory"]
        done, state, steps, fill_reward = False, None, [], 0.0
        while not done:
            a, state = agent.predict(obs, state=state, deterministic=True)
            obs, r, done, _, info = env.step(int(np.asarray(a).item()))
            steps.append(info)
            if info.get("fill_qty", 0.0) > 0:
                fill_reward += r
    finally:
        env.close()
    q = np.array([s.get("fill_qty", 0.0) for s in steps])
    p = np.array([s.get("fill_price", 0.0) for s in steps])
    mid = np.array([s["book_mid"] for s in steps])
    ref = np.array([s["reward_reference_mid"] for s in steps])
    k = 100.0 / (arrival * total)
    return dict(IS=-steps[-1]["total_IS_pct"], market=-k * np.sum(q * (arrival - ref)),
                impact=-k * np.sum(q * (ref - mid)), spread=-k * np.sum(q * (mid - p)),
                fill_reward=100 * fill_reward, sold=q.sum() / total, dust=steps[-1].get("dust_liquidated_qty", 0.0),
                forced=steps[-1].get("forced_liquidation_qty", 0.0))


def test_cost_parts_sum_to_is_and_reward_is_impact_plus_spread():
    e = _episode(3.0)
    assert e["forced"] == 0 and e["dust"] < 1e-6 and e["sold"] == pytest.approx(1.0)  # dust: float residue only
    assert e["market"] + e["impact"] + e["spread"] == pytest.approx(e["IS"], abs=1e-9)
    assert e["fill_reward"] == pytest.approx(e["impact"] + e["spread"], abs=1e-9)
    assert e["impact"] < 0 and e["spread"] < 0


def test_twap_baseline_reward_ranks_schedules_exactly_by_is():
    """Under --reward-baseline twap, fills are charged against the arrival price and TWAP's
    market gain on the same path is subtracted once (doc 25). So reward - IS is the same
    constant for every schedule on one price path: the reward orders schedules exactly as IS."""
    a, b = _episode(3.0, baseline="twap"), _episode(5.0, baseline="twap")
    assert a["IS"] != pytest.approx(b["IS"], abs=1e-6)  # the schedules really differ
    assert a["fill_reward"] - a["IS"] == pytest.approx(b["fill_reward"] - b["IS"], abs=1e-9)
