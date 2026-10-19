"""
Training-side wrapper for the residual-around-AC action space (M2).

The agent's action indexes evaluation.residual.DELTAS (a schedule offset);
the wrapper turns it into the environment's raw action with the same
ResidualACMapper the evaluation agent uses, so training and evaluation share
one mapping.
"""
import gymnasium as gym

from evaluation.residual import DELTAS, ResidualACMapper


class ResidualACActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(DELTAS))
        self.mapper = ResidualACMapper(n_state_dims=env.observation_space.shape[0])
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.mapper.set_horizon(info["episode_horizon"])
        self._obs = obs
        return obs, info

    def step(self, action):
        delta = DELTAS[int(action)]
        obs, reward, terminated, truncated, info = self.env.step(
            self.mapper.raw_action(delta, self._obs))
        self._obs = obs
        info = dict(info, residual_delta=delta)
        return obs, reward, terminated, truncated, info
