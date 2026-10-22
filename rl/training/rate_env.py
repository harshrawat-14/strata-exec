"""
Training-side wrapper for the rate-based action space (M4): the agent's
action indexes evaluation.rate.RATES; the same RateTracker the evaluation
agent uses turns it into the environment's raw action.
"""
import gymnasium as gym

from evaluation.rate import RATES, RateTracker


class RateActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(RATES))
        self.tracker = RateTracker(n_state_dims=env.observation_space.shape[0])
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.tracker.start_episode(info["episode_horizon"])
        self._obs = obs
        return obs, info

    def step(self, action):
        rate = RATES[int(action)]
        obs, reward, terminated, truncated, info = self.env.step(
            self.tracker.raw_action(rate, self._obs))
        self._obs = obs
        return obs, reward, terminated, truncated, dict(info, rate=rate)
