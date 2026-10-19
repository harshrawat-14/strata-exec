"""
Residual action space around AC-Optimal (M2; docs/experiments/15-*).

Instead of choosing a raw sell fraction, the agent chooses each step how far
ahead of (+) or behind (-) AC's schedule to be, as a fraction of the episode:
the target inventory after this step is AC's scheduled inventory at
(elapsed + delta * horizon). The trade that reaches it is executed exactly as
the closed-loop AC baseline executes (schedule tracking + nearest action), so
delta = 0 reproduces AC action for action. The agent inherits AC's execution
quality and only has to learn timing: when to run ahead of AC (e.g. before a
falling price) and when to hang back.

One mapping, used by both the training wrapper and the evaluation agent.
"""
import numpy as np

from .actions import get_action_fractions, nearest_action
from .policies.almgren_chriss import ACOptimalPolicy
from .protocol import AC_LAMBDA, EPISODE_STEPS

# Schedule offsets, fraction of the episode. Ordered, so the agent's
# distribution can be decoded by its expected value (MeanValueDecoder).
DELTAS = [-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


class ResidualACMapper:
    """Translate a schedule offset into the environment's raw action index."""

    def __init__(self, lam: float = AC_LAMBDA, total_steps: int = EPISODE_STEPS,
                 n_state_dims: int = 16):
        self.fractions = get_action_fractions(n_state_dims)
        self._ac = ACOptimalPolicy(n_state_dims, lam=lam, total_steps=total_steps,
                                   track_schedule=True)

    @property
    def total_steps(self) -> int:
        return self._ac.total_steps

    def set_horizon(self, total_steps: int) -> None:
        self._ac.set_horizon(total_steps)

    def raw_action(self, delta: float, obs) -> int:
        q_frac, t_frac = float(obs[0]), float(obs[1])
        if t_frac < 0.001 or q_frac < 0.001:
            return nearest_action(1.0, self.fractions)
        schedule = self._ac._schedule
        T = self.total_steps
        # Same index arithmetic as actions.schedule_target(track=True), shifted.
        k = int(round((1.0 - t_frac) * T)) + int(round(delta * T))
        if k < 0:
            target_after = 1.0
        else:
            k = min(k, len(schedule) - 1)
            target_after = max(0.0, 1.0 - float(np.sum(schedule[:k + 1])))
        target = max(0.0, (q_frac - target_after) / q_frac) if q_frac > 1e-6 else 1.0
        return nearest_action(target, self.fractions)


class ResidualACAgent:
    """Evaluation-side agent: a decoder over DELTAS feeding ResidualACMapper.

    `decoder` is any object with predict(obs, state, episode_start) returning
    an index into DELTAS (normally MeanValueDecoder(model, DELTAS)).
    """

    def __init__(self, decoder, lam: float = AC_LAMBDA, n_state_dims: int = 16):
        self.decoder = decoder
        self.mapper = ResidualACMapper(lam=lam, n_state_dims=n_state_dims)
        self.last_delta = None

    def set_horizon(self, total_steps: int) -> None:
        self.mapper.set_horizon(total_steps)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        idx, state = self.decoder.predict(obs, state=state, episode_start=episode_start,
                                          deterministic=deterministic)
        self.last_delta = DELTAS[int(np.asarray(idx).reshape(-1)[0])]
        return np.array([self.mapper.raw_action(self.last_delta, obs)]), state
