"""
Rate-based action space (M4): a chosen pace is delivered despite the
simulator's coarse action table, no action can dump a large share of the
order, and training and evaluation share one mapping.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from environment import RL_ENV_BIN, StrataExecEnv
from evaluation.protocol import EPISODE_STEPS
from evaluation.rate import RATES, RateAgent

DATE = "2024-07-05"
pytestmark = pytest.mark.skipif(
    not (os.path.exists(RL_ENV_BIN)
         and os.path.exists(f"TradeData/BookDepth/BTCUSDT-bookDepth-{DATE}.csv")),
    reason="requires the rl-env binary and TradeData/",
)


class _Fixed:
    def __init__(self, rate):
        self.idx = RATES.index(rate)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.array([self.idx]), state


def _env():
    return StrataExecEnv(mode="counterfactual", lob_date=DATE, agg_date=DATE,
                         btc_target=50_000.0, n_state_dims=16, fixed_steps=True,
                         fixed_size=True, episode_steps=EPISODE_STEPS)


def _run(agent, env, frac=0.5):
    obs, info = env.reset(seed=2, options={"start_frac": frac})
    agent.set_horizon(info["episode_horizon"])
    total = info["total_inventory"]
    done, step, q_mid, biggest = False, 0, None, 0.0
    while not done:
        a, _ = agent.predict(obs)
        obs, _, done, _, info = env.step(int(a[0]))
        step += 1
        biggest = max(biggest, info.get("fill_qty", 0.0) / total)
        if step == EPISODE_STEPS // 2:
            q_mid = float(obs[0])
    return q_mid, biggest, info.get("forced_liquidation_qty", 0.0) / total, step


def test_steady_pace_sells_evenly_and_finishes():
    env = _env()
    try:
        q_mid, biggest, forced, steps = _run(RateAgent(_Fixed(1.0)), env)
    finally:
        env.close()
    assert abs(q_mid - 0.5) < 0.01, q_mid
    assert forced < 0.002
    assert biggest < 0.003


def test_fastest_pace_is_bounded():
    env = _env()
    try:
        _, biggest, _, steps = _run(RateAgent(_Fixed(20.0)), env)
    finally:
        env.close()
    assert biggest < 0.02, biggest           # never a large share of the order at once
    assert steps < EPISODE_STEPS // 10       # but it does finish fast


def test_pause_holds_inventory():
    env = _env()
    try:
        obs, info = env.reset(seed=2, options={"start_frac": 0.5})
        agent = RateAgent(_Fixed(0.0))
        agent.set_horizon(info["episode_horizon"])
        for _ in range(100):
            a, _ = agent.predict(obs)
            obs, *_ = env.step(int(a[0]))
    finally:
        env.close()
    assert float(obs[0]) == pytest.approx(1.0)


def test_training_wrapper_matches_evaluation_agent():
    from training.rate_env import RateActionWrapper

    wrapped, plain = RateActionWrapper(_env()), _env()
    try:
        assert wrapped.action_space.n == len(RATES)
        agent = RateAgent(_Fixed(1.5))
        obs_w, _ = wrapped.reset(seed=4, options={"start_frac": 0.2})
        obs_p, info_p = plain.reset(seed=4, options={"start_frac": 0.2})
        agent.set_horizon(info_p["episode_horizon"])
        done = False
        while not done:
            raw, _ = agent.predict(obs_p)
            obs_p, _, done, _, info_p = plain.step(int(raw[0]))
            obs_w, _, done_w, _, info_w = wrapped.step(RATES.index(1.5))
            assert done_w == done and info_w["rate"] == 1.5
        assert info_w["total_IS_pct"] == info_p["total_IS_pct"]
    finally:
        wrapped.close()
        plain.close()
