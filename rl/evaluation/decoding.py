"""
Deterministic action decoding for the RL agent's discrete sell-fraction policy.

The 20 actions are ordered sell fractions (0, 0.1%, ..., 100% of remaining).
SB3's `deterministic=True` takes the single most likely action. For a policy
that spreads probability across neighbouring fractions, that mode can be
"wait" (action 0, ~30%) while the sell actions together carry ~70%: the
argmax agent then waits most steps and dumps the order at the end, behaving
nothing like the policy it was trained as (found evaluating M0, 2026-10-18;
docs/experiments/13-*).

MeanFractionDecoder instead plays the action nearest to the policy's
expected sell fraction -- deterministic and reproducible, and faithful to
what the trained (sampling) policy does on average.
"""
import numpy as np
import torch as th

from .actions import get_action_fractions, nearest_action


class MeanFractionDecoder:
    """Wrap a RecurrentPPO model; predict() plays the nearest action to E[fraction]."""

    def __init__(self, model, n_state_dims: int = 16):
        self.model = model
        self.fractions = np.asarray(get_action_fractions(n_state_dims), dtype=np.float64)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        policy = self.model.policy
        policy.set_training_mode(False)
        obs_t, _ = policy.obs_to_tensor(obs)
        n_envs = obs_t.shape[0]
        if state is None:
            zeros = np.concatenate([np.zeros(policy.lstm_hidden_state_shape)
                                    for _ in range(n_envs)], axis=1)
            state = (zeros, zeros)
        if episode_start is None:
            episode_start = np.zeros(n_envs, dtype=bool)
        with th.no_grad():
            lstm_states = (th.tensor(state[0], dtype=th.float32, device=policy.device),
                           th.tensor(state[1], dtype=th.float32, device=policy.device))
            starts = th.tensor(episode_start, dtype=th.float32, device=policy.device)
            dist, new_states = policy.get_distribution(obs_t, lstm_states, starts)
            probs = dist.distribution.probs.cpu().numpy()
        expected = probs @ self.fractions
        actions = np.array([nearest_action(float(f), list(self.fractions)) for f in expected])
        return actions, (new_states[0].cpu().numpy(), new_states[1].cpu().numpy())
