"""
Environment factories for training and evaluation VecEnvs.
"""

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import StrataExecEnv


def make_multi_date_eval_env(mode, adversarial, eval_dates, btc_target,
                             n_state_dims, no_time_features):
    """
    Phase 2R fix (2): a MULTI-date eval env for best-model selection.

    The original eval env was a single date (multi_dates[0] = 2024-01-15, a
    trending-up date where waiting scores acceptably), so best-model selection --
    even though it selects on reward -- was blind to crash-day behavior and could
    not distinguish a wait-collapsed policy from a good one on the dimension that
    matters most. This builds one sub-env per eval date (all training dates PLUS
    the held-out crash dates), so EvalCallback's mean reward is averaged across
    all regimes and a policy that collapses on crash days is correctly penalized.

    CAVEAT (documented, for the writeup): including the held-out crash dates in
    best-model SELECTION means their eval numbers are no longer a clean held-out
    generalization measure for the selected model. Per explicit instruction, crash
    behavior must count in selection (or the single-trending-date pathology
    returns); if a pure generalization claim on the crash dates is later needed, a
    separate untouched test date must be reserved.
    """
    def _one(date):
        return lambda: Monitor(StrataExecEnv(
            mode=mode, adversarial=adversarial, lob_date=date, agg_date=date,
            btc_target=btc_target, n_state_dims=n_state_dims, seed=9999,
            fixed_steps=True, fixed_size=True,
            no_time_features=no_time_features,
        ))
    return DummyVecEnv([_one(d) for d in eval_dates])


def make_env(
    mode="synthetic",
    adversarial=False,
    lob_date=None,
    agg_date=None,
    btc_target=50000.0,
    multi_dates=None,
    n_state_dims=8,
    seed=42,
    fixed_steps=False,
    fixed_size=False,
    no_time_features=False,
    worker_date=None,
):
    import random

    def _init():
        # `_init` runs ONCE per SubprocVecEnv worker. It used to do
        # `date = random.choice(multi_dates)` here, which pinned each worker to
        # a single RANDOM date for the whole run -- so an N-date training set
        # was really trained on at most n_envs dates, chosen by luck. The Step 2
        # run (16 dates, n_envs=4) hit 3 distinct dates and ZERO crash dates,
        # defeating the entire purpose of the V2 split.
        #
        # Two correct strategies, picked by the caller in train():
        #   worker_date  -- this worker owns one date, assigned round-robin so
        #                   the workers jointly cover every date. Used when
        #                   n_envs >= len(multi_dates). No respawns, and every
        #                   PPO batch contains all dates rather than a random
        #                   subset, so regime coverage is guaranteed per update.
        #   lob_dates    -- redraw per episode (respawns the subprocess).
        #                   Correct at any n_envs but ~15x slower on short
        #                   episodes, so it is the fallback when there are more
        #                   dates than workers.
        env = StrataExecEnv(
            mode=mode,
            adversarial=adversarial,
            lob_date=worker_date if worker_date else (lob_date if not multi_dates else None),
            agg_date=worker_date if worker_date else (agg_date if not multi_dates else None),
            lob_dates=None if worker_date else multi_dates,
            btc_target=btc_target,
            n_state_dims=n_state_dims,
            seed=seed,
            fixed_steps=fixed_steps,
            fixed_size=fixed_size,
            no_time_features=no_time_features,
        )
        return Monitor(env)
    return _init
