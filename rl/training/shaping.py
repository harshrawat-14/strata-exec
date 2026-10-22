"""
Potential-based reward shaping for self-inflicted impact (M4; docs/experiments/17-*).

A trade's main cost is not paid on the step it happens: it thins the book for
hours (temporary impact, ~400-step half-life) or for good (permanent impact),
so every later fill is worse. PPO links costs to only the last ~17 decisions,
so it cannot see that. The potential below estimates what the REMAINING
inventory will pay because of the impact already in the book:

    Phi(s) = -k * q * (P + T * D(n))

q = remaining inventory fraction, P / T = permanent / temporary depth
reduction, D(n) = the average remaining strength of temporary impact over the
n steps left, (1 - e^{-a n}) / (a n) with a = rho * dt from the simulator's own
calibration. The shaped reward adds F = gamma * Phi(s') - Phi(s), so a trade
that raises impact is charged its estimated future cost immediately.

Potential-based shaping leaves the optimal policy unchanged (Ng, Harada &
Russell, 1999); Phi = 0 at the start (no impact) and at the end (no inventory),
so the shaped return equals the original return plus -Phi(s0) = 0. Only the
speed of learning changes. k is fitted on training dates only
(rl/experiments/m4_fit_impact_potential.py).
"""
import math

import gymnasium as gym


def decay_weight(n_remaining: float, rho_dt: float) -> float:
    """Average remaining strength of a temporary impact over the next n steps."""
    x = rho_dt * max(n_remaining, 0.0)
    return 1.0 if x < 1e-9 else (1.0 - math.exp(-x)) / x


def impact_exposure(q_frac: float, info: dict, n_remaining: float, rho_dt: float) -> float:
    """q * (P + T * D(n)) from one step's info (impact fields are in percent)."""
    total = info.get("depth_reduction_pct", 0.0) / 100.0
    permanent = info.get("permanent_impact_pct", 0.0) / 100.0
    temporary = max(total - permanent, 0.0)
    return q_frac * (permanent + temporary * decay_weight(n_remaining, rho_dt))


class ImpactPotentialShaping(gym.Wrapper):
    def __init__(self, env, k: float, gamma: float, rho_dt: float):
        super().__init__(env)
        self.k, self.gamma, self.rho_dt = k, gamma, rho_dt
        self._phi = 0.0
        self._horizon = None

    def _potential(self, obs, info, done: bool) -> float:
        if done:
            return 0.0  # terminal: no inventory left
        q_frac, t_frac = float(obs[0]), float(obs[1])
        n_remaining = t_frac * self._horizon
        return -self.k * impact_exposure(q_frac, info, n_remaining, self.rho_dt)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._horizon = info["episode_horizon"]
        self._phi = 0.0  # no impact at the start of an episode
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = self._potential(obs, info, terminated or truncated)
        shaping = self.gamma * phi_next - self._phi
        self._phi = phi_next
        info = dict(info, shaping_reward=shaping, env_reward=reward)
        return obs, reward + shaping, terminated, truncated, info
