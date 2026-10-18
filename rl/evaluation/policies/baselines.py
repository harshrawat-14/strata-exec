"""
Rule-based baselines: TWAP and the heuristic policy.
"""

import numpy as np

from ..actions import get_action_fractions, nearest_action


class TWAPPolicy:
    """
    Time-Weighted Average Price execution policy.
    Divides total inventory uniformly across all remaining steps.
    """

    def __init__(self, n_state_dims: int = 12):
        self.fractions = get_action_fractions(n_state_dims)
        self.total_steps = 2880  # default evaluation horizon

    def set_horizon(self, total_steps: int) -> None:
        """Match the episode length the environment actually runs."""
        self.total_steps = int(total_steps)

    def predict(self, obs, state=None, episode_start=None,
                deterministic=True):
        q_frac = float(obs[0])   # remaining inventory fraction
        t_frac = float(obs[1])   # time remaining fraction

        if t_frac < 0.001 or q_frac < 0.001:
            action = nearest_action(1.0, self.fractions)
        else:
            remaining_steps = max(1.0, t_frac * self.total_steps)
            target = min(1.0, 1.0 / remaining_steps)
            action = nearest_action(target, self.fractions)

        return np.array([action]), state


class HeuristicPolicy:
    """
    Volatility and liquidity-adjusted TWAP.
    Sells faster when volatility is high or the book is deep,
    slower when volatility is low or the book is thin.
    """

    def __init__(self, n_state_dims: int = 12,
                 vol_weight: float = 0.30,
                 depth_weight: float = 0.20):
        self.fractions = get_action_fractions(n_state_dims)
        self.total_steps = 2880
        self.vol_weight = vol_weight
        self.depth_weight = depth_weight

    def set_horizon(self, total_steps: int) -> None:
        """Match the episode length the environment actually runs."""
        self.total_steps = int(total_steps)

    def predict(self, obs, state=None, episode_start=None,
                deterministic=True):
        q_frac    = float(obs[0])
        t_frac    = float(obs[1])
        vol_ratio = float(obs[2])    # already tanh-normalised
        depth_ratio = float(obs[4])  # already tanh-normalised

        if t_frac < 0.001 or q_frac < 0.001:
            action = nearest_action(1.0, self.fractions)
        else:
            remaining_steps = max(1.0, t_frac * self.total_steps)
            base = min(1.0, 1.0 / remaining_steps)

            vol_scalar   = 1.0 + self.vol_weight * vol_ratio
            depth_scalar = 1.0 + self.depth_weight * depth_ratio

            target = min(1.0, base * vol_scalar * depth_scalar)
            target = max(0.0, target)

            action = nearest_action(target, self.fractions)

        return np.array([action]), state
