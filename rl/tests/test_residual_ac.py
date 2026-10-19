"""
Residual-around-AC action space (M2): offset 0 must BE the AC baseline, and the
training wrapper and evaluation agent must share one mapping.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from environment import RL_ENV_BIN, StrataExecEnv
from evaluation.episode import _run_episode
from evaluation.policies.almgren_chriss import ACOptimalPolicy
from evaluation.protocol import AC_LAMBDA, EPISODE_STEPS
from evaluation.residual import DELTAS, ResidualACAgent

DATE = "2024-07-05"
pytestmark = pytest.mark.skipif(
    not (os.path.exists(RL_ENV_BIN)
         and os.path.exists(f"TradeData/BookDepth/BTCUSDT-bookDepth-{DATE}.csv")),
    reason="requires the rl-env binary and TradeData/",
)


class _Fixed:
    """Decoder stub that always picks one offset."""

    def __init__(self, delta):
        self.idx = DELTAS.index(delta)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.array([self.idx]), state


def _env():
    return StrataExecEnv(mode="counterfactual", lob_date=DATE, agg_date=DATE,
                         btc_target=50_000.0, n_state_dims=16, fixed_steps=True,
                         fixed_size=True, episode_steps=EPISODE_STEPS)


def _actions(policy, env, frac):
    obs, info = env.reset(seed=3, options={"start_frac": frac})
    policy.set_horizon(info["episode_horizon"])
    acts, done, mid_q = [], False, None
    while not done:
        a, _ = policy.predict(obs)
        a = int(np.asarray(a).item())
        acts.append(a)
        obs, _, done, _, info = env.step(a)
        if len(acts) == EPISODE_STEPS // 2:
            mid_q = float(obs[0])
    return acts, -info["total_IS_pct"], mid_q


def test_zero_offset_reproduces_ac_action_for_action():
    env = _env()
    try:
        for frac in (0.0, 0.6):
            ac = _actions(ACOptimalPolicy(16, lam=AC_LAMBDA, track_schedule=True), env, frac)
            res = _actions(ResidualACAgent(_Fixed(0.0)), env, frac)
            assert res[0] == ac[0]
            assert res[1] == ac[1]
    finally:
        env.close()


def test_positive_offset_sells_earlier_negative_later():
    env = _env()
    try:
        mid = {d: _actions(ResidualACAgent(_Fixed(d)), env, 0.3)[2] for d in (-0.1, 0.0, 0.2)}
    finally:
        env.close()
    assert mid[0.2] < mid[0.0] < mid[-0.1]


def test_training_wrapper_matches_evaluation_agent():
    from training.residual_env import ResidualACActionWrapper

    wrapped = ResidualACActionWrapper(_env())
    plain = _env()
    try:
        assert wrapped.action_space.n == len(DELTAS)
        agent = ResidualACAgent(_Fixed(0.1))
        obs_w, info_w = wrapped.reset(seed=5, options={"start_frac": 0.4})
        obs_p, info_p = plain.reset(seed=5, options={"start_frac": 0.4})
        agent.set_horizon(info_p["episode_horizon"])
        done = False
        while not done:
            raw, _ = agent.predict(obs_p)
            obs_p, _, done, _, info_p = plain.step(int(raw[0]))
            obs_w, _, done_w, _, info_w = wrapped.step(DELTAS.index(0.1))
            assert done_w == done
            assert info_w["residual_delta"] == 0.1
        assert info_w["total_IS_pct"] == info_p["total_IS_pct"]
    finally:
        wrapped.close()
        plain.close()
