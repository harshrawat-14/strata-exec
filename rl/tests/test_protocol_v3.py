"""
Evaluation protocol v3 (rl/evaluation/protocol.py) and the training-seed fix.

The seed regression: StrataExecEnv.reset(seed=None) used to reuse the
constructor seed, and SB3 calls reset() without a seed after every worker's
first episode -- so every training episode replayed one fixed start, horizon
and order size per date (docs/history/timeline.md, 2026-10-18).
"""
from __future__ import annotations

import os

import pytest

from environment import RL_ENV_BIN, StrataExecEnv
from evaluation.dates import (
    HELD_OUT_DATES_V2, SEVERITY_RANK_V2, TRAINING_DATES_V2, TRAINING_DATES_V3,
    VALIDATION_DATES_V3,
)
from evaluation.protocol import EPISODE_STEPS, eval_start_fracs, fit_start_fracs

DATE = "2024-01-15"
needs_env = pytest.mark.skipif(
    not (os.path.exists(RL_ENV_BIN)
         and os.path.exists(f"TradeData/BookDepth/BTCUSDT-bookDepth-{DATE}.csv")),
    reason="requires the rl-env binary and TradeData/",
)


def _env(**kw):
    return StrataExecEnv(mode="counterfactual", lob_date=DATE, agg_date=DATE,
                         btc_target=50_000.0, n_state_dims=16, **kw)


@needs_env
def test_seedless_resets_draw_new_episodes():
    env = _env(seed=42)  # legacy training config: random horizon and size
    try:
        env.reset(seed=42)  # what SB3's first, seeded reset does
        episodes = {(i["episode_horizon"], round(i["size_fraction"], 6))
                    for i in (env.reset()[1] for _ in range(6))}
    finally:
        env.close()
    assert len(episodes) == 6, f"seedless resets repeated an episode: {episodes}"


@needs_env
def test_seeded_reset_restarts_the_same_stream():
    env = _env(seed=1)
    try:
        env.reset(seed=7)
        first = [env.reset()[1]["size_fraction"] for _ in range(3)]
        env.reset(seed=7)
        again = [env.reset()[1]["size_fraction"] for _ in range(3)]
    finally:
        env.close()
    assert first == again


@needs_env
def test_v3_training_episodes_vary_start_with_fixed_horizon():
    env = _env(seed=42, episode_steps=EPISODE_STEPS)
    try:
        env.reset(seed=42)
        infos = [env.reset()[1] for _ in range(6)]
    finally:
        env.close()
    assert {i["episode_horizon"] for i in infos} == {EPISODE_STEPS}
    assert len({i["start_offset"] for i in infos}) == 6
    assert all(0 <= i["start_offset"] <= i["max_start"] for i in infos)


@needs_env
def test_start_cycle_repeats_the_same_spread():
    env = _env(seed=9999, episode_steps=EPISODE_STEPS, fixed_size=True, start_cycle=3)
    try:
        starts = [env.reset()[1]["start_offset"] for _ in range(6)]
        max_start = env._last_info["max_start"]
    finally:
        env.close()
    assert starts == [0, round(max_start / 2), max_start] * 2


@needs_env
def test_v3_baselines_adopt_env_horizon_and_never_dump_on_empty_book():
    from evaluation.harness import evaluate_all_strategies_on_date
    from evaluation.policies.baselines import TWAPPolicy

    res = evaluate_all_strategies_on_date(
        rl_model=TWAPPolicy(16), date=DATE, n_episodes=2, n_state_dims=16,
        mode="counterfactual", ac_lambda=1e-6, n_bootstrap=50,
        episode_steps=EPISODE_STEPS,
    )
    # AC-Optimal at honest lambda finishes well before the horizon; under a
    # wrong (2880-step) horizon assumption it would under-sell and dump.
    assert res["ac_optimal"]["forced_liquidation_rate"] == 0.0


def test_protocol_start_positions():
    assert eval_start_fracs(5) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert eval_start_fracs(1) == [0.0]
    fit = fit_start_fracs(3)
    assert fit == [1 / 6, 0.5, 5 / 6]
    assert not set(fit) & set(eval_start_fracs(20))


def test_v3_split_is_disjoint_and_covers_regimes():
    assert set(TRAINING_DATES_V3) | set(VALIDATION_DATES_V3) == set(TRAINING_DATES_V2)
    assert not set(TRAINING_DATES_V3) & set(VALIDATION_DATES_V3)
    assert not (set(TRAINING_DATES_V3) | set(VALIDATION_DATES_V3)) & set(HELD_OUT_DATES_V2)
    for dates in (TRAINING_DATES_V3, VALIDATION_DATES_V3):
        regimes = {SEVERITY_RANK_V2[d][1] for d in dates}
        assert regimes == {"crash", "normal", "calm"}, (dates, regimes)


def test_train_refuses_held_out_selection_dates(monkeypatch):
    import train as train_module

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(train_module.mlflow, "start_run", lambda **k: FakeRun())
    monkeypatch.setattr(train_module.mlflow, "log_params", lambda p: None)
    monkeypatch.setattr(train_module, "SubprocVecEnv", lambda fns: object())
    with pytest.raises(ValueError, match="held-out"):
        train_module.train(mode="counterfactual", multi_dates=list(TRAINING_DATES_V3),
                           eval_dates=["2024-08-05"], total_timesteps=1, n_envs=1,
                           force_device="cpu")
