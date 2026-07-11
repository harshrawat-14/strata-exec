"""
Regression check for the SB3 VecEnv adapter (Phase S review follow-up #2):
Phase 0's harness (evaluate_all_strategies_on_date) only ever exercises the
single-instance JSON path via StrataExecEnv, never the batched env or the
VecEnv wrapper Phase 2's retrain will actually train through. A clean Phase 0
harness result says nothing about whether BatchedStrataExecVecEnv's batch
ordering, reset semantics, or done-flag handling are correct -- this is
exactly where those would silently break.

This test drives BatchedStrataExecVecEnv with fixed_steps=False (real
training config: randomised per-episode horizons, so different instances
finish episodes at different global step counts -- the scenario that
actually exercises the auto-reset-on-done / reset_indices path, unlike
Phase 0's fixed-horizon eval config where every instance finishes together
and the whole distinction is invisible). It then uses the VecEnv's
instrumented reset_log to reconstruct exactly which seed backed each
per-slot episode segment, and replays those exact segments through the
single-instance JSON path (StrataExecEnv / rl/evaluate.py's own transport),
asserting per-step observation arrays match to float32 precision.
"""
from __future__ import annotations

import numpy as np
import pytest

from batched_vec_env import BatchedStrataExecVecEnv
from environment import StrataExecEnv

DATE = "2024-01-15"
BATCH_SIZE = 4
N_STEPS = 400
SEED_BASE = 5000


def _action_at(global_step: int) -> int:
    # Deterministic but varied action sequence, indexed by GLOBAL step count
    # so a per-slot segment starting at global step t0 can be replayed
    # exactly: local step k within that segment used action_at(t0 + k).
    return global_step % 20


@pytest.mark.skipif(
    not __import__("os").path.exists(f"TradeData/BookDepth/BTCUSDT-bookDepth-{DATE}.csv"),
    reason="requires TradeData/ fixtures",
)
def test_batched_vec_env_matches_single_instance_json_path():
    vec_env = BatchedStrataExecVecEnv(
        batch_size=BATCH_SIZE,
        mode="historical",
        lob_date=DATE,
        agg_date=DATE,
        total_inventory=1_000_000.0,
        btc_target=50_000.0,
        fixed_steps=False,   # real training config: randomised horizons per
        fixed_size=False,    # episode, so instances finish at different times
        no_time_features=True,
        seed_base=SEED_BASE,
    )

    try:
        vec_env.seed(SEED_BASE)
        obs = vec_env.reset()
        assert obs.shape == (BATCH_SIZE, vec_env.batched_env.n_state_dims)

        # Record every slot's per-step observation, indexed by global step.
        recorded_obs = {i: [] for i in range(BATCH_SIZE)}  # slot -> list of (global_step, obs)
        # obs at global_step=0 (post-reset, pre-first-step) per slot:
        for i in range(BATCH_SIZE):
            recorded_obs[i].append((0, obs[i].copy()))

        for step_count in range(1, N_STEPS + 1):
            actions = np.array([_action_at(step_count - 1)] * BATCH_SIZE)
            vec_env.step_async(actions)
            obs, rewards, dones, infos = vec_env.step_wait()
            for i in range(BATCH_SIZE):
                recorded_obs[i].append((step_count, obs[i].copy()))

        reset_log = list(vec_env.reset_log)
    finally:
        vec_env.close()

    assert len(reset_log) >= BATCH_SIZE + 1, (
        "expected at least the initial reset plus some auto-resets over "
        f"{N_STEPS} steps of randomised-horizon episodes -- got {len(reset_log)} "
        "reset_log entries; the test may not be exercising the auto-reset path "
        "at all (fixed_steps might be accidentally True)"
    )

    # Group reset_log entries by slot, sorted by step, to get each slot's
    # sequence of (segment_start_step, seed) pairs.
    by_slot: dict[int, list[tuple[int, int]]] = {i: [] for i in range(BATCH_SIZE)}
    for step, slot, seed in reset_log:
        by_slot[slot].append((step, seed))
    for slot in by_slot:
        by_slot[slot].sort()

    n_segments_checked = 0
    n_points_checked = 0

    for slot, segments in by_slot.items():
        for seg_idx, (t0, seed) in enumerate(segments):
            t_end = segments[seg_idx + 1][0] if seg_idx + 1 < len(segments) else N_STEPS

            env = StrataExecEnv(
                mode="historical",
                lob_date=DATE,
                agg_date=DATE,
                btc_target=50_000.0,
                n_state_dims=12,
                fixed_steps=False,
                fixed_size=False,
                no_time_features=True,
            )
            try:
                single_obs, _info = env.reset(seed=seed)
                batched_obs_at_t0 = dict(recorded_obs[slot])[t0]
                assert np.array_equal(single_obs, batched_obs_at_t0), (
                    f"slot={slot} segment starting at global step {t0} (seed={seed}): "
                    f"reset() observation mismatch between batched VecEnv and "
                    f"single-instance JSON path"
                )
                n_points_checked += 1

                local_step = 0
                g = t0
                while g < t_end:
                    action = _action_at(g)
                    single_obs, _reward, done, _trunc, _info = env.step(action)
                    g += 1
                    local_step += 1

                    batched_points = dict(recorded_obs[slot])
                    if g in batched_points:
                        # Only compare while this segment is still active
                        # in the batched run (it stops being this segment's
                        # data once t_end is reached / the next reset fires).
                        if g <= t_end:
                            expected = batched_points[g]
                            # After the segment's own terminal step, the
                            # batched run's obs[i] at t_end has ALREADY been
                            # overwritten with the next episode's reset
                            # observation (auto-reset happens within the same
                            # step_wait() call) -- so only compare up to and
                            # including t_end - 1's post-step observation,
                            # i.e. skip the exact boundary step here (it's
                            # covered as the NEXT segment's t0 check above).
                            if g < t_end:
                                assert np.array_equal(single_obs, expected), (
                                    f"slot={slot} segment seed={seed} local_step={local_step} "
                                    f"(global step {g}): step() observation mismatch"
                                )
                                n_points_checked += 1
                    if done:
                        break
                n_segments_checked += 1
            finally:
                env.close()

    assert n_segments_checked >= BATCH_SIZE, "should have checked at least one segment per slot"
    print(
        f"batched-VecEnv vs single-instance JSON: {n_segments_checked} episode "
        f"segments, {n_points_checked} observation points, all matched exactly"
    )
