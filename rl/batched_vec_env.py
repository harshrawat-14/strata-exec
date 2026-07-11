"""
SB3-compatible VecEnv adapter around BatchedStrataExecEnv (Phase S, ADR-006).

This is the actual training-time wrapper Phase 2's retrain will drive
RecurrentPPO through -- not the standalone Phase 0 harness
(evaluate_all_strategies_on_date), which only ever exercises the
single-instance JSON path via StrataExecEnv. The batch ordering, reset
semantics, and done-flag handling implemented here are exactly what a
"byte-for-byte identical" regression claim needs to cover and the
standalone harness cannot: SB3's VecEnv contract requires that when an
instance's episode ends, ONLY that instance auto-resets (with the new
episode's observation returned in its slot and the true final observation
stashed in info["terminal_observation"]) while every other in-progress
instance continues untouched. Getting this wrong silently truncates
episodes during training without raising any error.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from batched_environment import BatchedStrataExecEnv


class BatchedStrataExecVecEnv(VecEnv):
    """N StrataExec environment instances, stepped together via one rl-env
    batch process, exposed through SB3's VecEnv interface."""

    def __init__(
        self,
        batch_size: int,
        mode: str = "counterfactual",
        lob_date: Optional[str] = None,
        agg_date: Optional[str] = None,
        total_inventory: float = 1_000_000.0,
        total_steps: int = 2880,
        btc_target: float = 50_000.0,
        fixed_steps: bool = False,
        fixed_size: bool = False,
        no_time_features: bool = True,
        seed_base: int = 0,
    ):
        self.batched_env = BatchedStrataExecEnv(
            batch_size=batch_size,
            mode=mode,
            lob_date=lob_date,
            agg_date=agg_date,
            total_inventory=total_inventory,
            total_steps=total_steps,
            btc_target=btc_target,
            fixed_steps=fixed_steps,
            fixed_size=fixed_size,
            no_time_features=no_time_features,
        )
        n_state_dims = self.batched_env.n_state_dims
        observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_state_dims,), dtype=np.float32
        )
        action_space = gym.spaces.Discrete(20)
        super().__init__(batch_size, observation_space, action_space)

        self._next_seed = seed_base
        self._actions: Optional[np.ndarray] = None
        self._seeds_pending: Optional[list[int]] = None  # set by .seed(), consumed by reset()
        self._step_count = 0

        # Instrumentation for regression testing: (step_count, instance_index,
        # seed) for every reset (initial or auto-reset-on-done), so a test can
        # reconstruct exactly which seed backed each per-slot episode segment
        # and replay it through the single-instance JSON path for comparison.
        # Not used by training; harmless to leave populated.
        self.reset_log: list[tuple[int, int, int]] = []

    def _draw_seeds(self, n: int) -> list[int]:
        seeds = list(range(self._next_seed, self._next_seed + n))
        self._next_seed += n
        return seeds

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self._seeds_pending = [seed + i for i in range(self.num_envs)]
        return self._seeds_pending

    def reset(self) -> np.ndarray:
        if self._seeds_pending is not None:
            seeds = self._seeds_pending
            self._seeds_pending = None
        else:
            seeds = self._draw_seeds(self.num_envs)
        obs = self.batched_env.reset_batch(seeds)
        for i, s in enumerate(seeds):
            self.reset_log.append((self._step_count, i, s))
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        assert self._actions is not None, "step_async must be called before step_wait"
        actions = [int(a) for a in self._actions]
        obs, rewards, dones = self.batched_env.step_batch(actions)
        self._step_count += 1
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        # SB3 VecEnv auto-reset-on-done contract: for every instance whose
        # episode just ended, stash the true final observation in
        # info["terminal_observation"], then reset ONLY that instance (via
        # reset_indices, not a full reset_batch) and splice the new episode's
        # initial observation into obs[i] before returning. Every other
        # instance's in-progress episode is untouched.
        done_indices = [i for i, d in enumerate(dones) if d]
        if done_indices:
            for i in done_indices:
                infos[i]["terminal_observation"] = obs[i].copy()
            reset_seeds = self._draw_seeds(len(done_indices))
            new_obs = self.batched_env.reset_indices(done_indices, reset_seeds)
            for k, i in enumerate(done_indices):
                obs[i] = new_obs[k]
                self.reset_log.append((self._step_count, i, reset_seeds[k]))

        return obs, rewards, dones, infos

    def close(self) -> None:
        self.batched_env.close()

    # -- Minimal VecEnv plumbing methods -------------------------------------
    # These have no real per-instance Python object to introspect (the N
    # environments live inside the Rust batch process, not as N separate
    # gym.Env objects), so they're honestly limited rather than faked.

    # Attributes SB3's own VecEnv base class (and common wrappers/callbacks)
    # query via get_attr with a well-defined, sensible default -- these are
    # answered honestly (there is genuinely no render mode, no gym.Env
    # instances to report a spec for) rather than raising, since raising here
    # breaks VecEnv.__init__ itself (it calls get_attr("render_mode")).
    _KNOWN_ATTRS = {
        "render_mode": None,
        "spec": None,
    }

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        indices = self._get_indices(indices)
        if attr_name in self._KNOWN_ATTRS:
            return [self._KNOWN_ATTRS[attr_name] for _ in indices]
        raise NotImplementedError(
            f"BatchedStrataExecVecEnv has no per-instance Python objects to "
            f"introspect (all N instances live inside one Rust batch process); "
            f"get_attr('{attr_name}') is not supported. If SB3 code you're "
            f"using needs this, it likely isn't compatible with this batched "
            f"transport."
        )

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        raise NotImplementedError(
            "BatchedStrataExecVecEnv has no per-instance Python objects to "
            "set attributes on; set_attr is not supported."
        )

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        raise NotImplementedError(
            "BatchedStrataExecVecEnv has no per-instance Python objects to "
            "call methods on; env_method is not supported."
        )

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        # No gym.Wrapper is applied anywhere in this batched path.
        indices = self._get_indices(indices)
        return [False for _ in indices]
