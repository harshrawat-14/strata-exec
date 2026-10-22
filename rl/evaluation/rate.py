"""
Rate-based action space (M4; docs/experiments/19-*).

The agent chooses a selling PACE each step: k times the steady rate
(order / horizon). k = 1 every step is a steady (TWAP-like) sale; 0 pauses;
the largest choice, 20x, sells about 1% of the order in a step, so no
action can dump a large share of it at once.

The simulator only accepts its fixed "fraction of remaining" actions, so a
requested pace usually cannot be hit exactly. RateTracker keeps the running
"should have sold by now" inventory implied by the agent's own chosen paces
and each step sells toward it, so rounding is corrected on the next step.
The fine alternation between tiny buckets that AC needs (doc 18: 837 switches
per episode) is done here, mechanically; the agent only picks the pace. There
is no external schedule to chase, so the M2 lag-and-burst pattern cannot arise.

One tracker serves the training wrapper and the evaluation agent.
"""
import numpy as np

from .actions import get_action_fractions, nearest_action

# Multiples of the steady rate. Ordered, so a policy over them can be decoded
# by its expected value (MeanValueDecoder).
RATES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]


class RateTracker:
    """Turn a chosen pace into the environment's raw action index."""

    def __init__(self, n_state_dims: int = 16):
        self.fractions = get_action_fractions(n_state_dims)
        self.total_steps = None
        self.desired_remaining = 1.0

    def start_episode(self, total_steps: int) -> None:
        self.total_steps = int(total_steps)
        self.desired_remaining = 1.0

    def raw_action(self, rate: float, obs) -> int:
        q_frac, t_frac = float(obs[0]), float(obs[1])
        if t_frac < 0.001 or q_frac < 0.001:
            return nearest_action(1.0, self.fractions)
        self.desired_remaining = max(0.0, self.desired_remaining - rate / self.total_steps)
        target = max(0.0, (q_frac - self.desired_remaining) / q_frac)
        return nearest_action(target, self.fractions)


class RateAgent:
    """Evaluation-side agent: a decoder over RATES feeding RateTracker.

    Under protocol v3, _run_episode calls set_horizon at the start of every
    episode; that is where the tracker is reset.
    """

    def __init__(self, decoder, n_state_dims: int = 16):
        self.decoder = decoder
        self.tracker = RateTracker(n_state_dims)
        self.last_rate = None

    def set_horizon(self, total_steps: int) -> None:
        self.tracker.start_episode(total_steps)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        idx, state = self.decoder.predict(obs, state=state, episode_start=episode_start,
                                          deterministic=deterministic)
        self.last_rate = RATES[int(np.asarray(idx).reshape(-1)[0])]
        return np.array([self.tracker.raw_action(self.last_rate, obs)]), state
