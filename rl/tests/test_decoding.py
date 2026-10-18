"""MeanFractionDecoder plays the action nearest the policy's expected sell fraction."""
from __future__ import annotations

import numpy as np
import torch as th

from evaluation.actions import get_action_fractions, nearest_action
from evaluation.decoding import MeanFractionDecoder


class _Probs:
    def __init__(self, probs):
        self.probs = th.tensor(probs, dtype=th.float32)


class _Dist:
    def __init__(self, probs):
        self.distribution = _Probs(probs)


class _Policy:
    """Stands in for RecurrentActorCriticPolicy with a fixed action distribution."""
    device = "cpu"
    lstm_hidden_state_shape = (1, 1, 4)

    def __init__(self, probs):
        self._probs = probs

    def set_training_mode(self, mode):
        pass

    def obs_to_tensor(self, obs):
        return th.tensor(np.atleast_2d(obs), dtype=th.float32), False

    def get_distribution(self, obs, lstm_states, starts):
        return _Dist([self._probs] * obs.shape[0]), lstm_states


class _Model:
    def __init__(self, probs):
        self.policy = _Policy(probs)


def test_wait_mode_with_sell_mass_decodes_to_a_sell():
    fr = get_action_fractions(16)
    probs = np.zeros(len(fr))
    probs[0] = 0.30                 # "wait" is the single most likely action...
    probs[3:7] = 0.175              # ...but 70% of the mass sells 0.5%-1.5% of remaining
    assert int(probs.argmax()) == 0  # so argmax decoding would wait
    action, _ = MeanFractionDecoder(_Model(probs)).predict(np.zeros(16))
    expected = float(probs @ np.asarray(fr))
    assert action[0] == nearest_action(expected, fr) and action[0] != 0


def test_point_mass_decodes_to_that_action():
    fr = get_action_fractions(16)
    for k in (0, 7, 19):
        probs = np.zeros(len(fr)); probs[k] = 1.0
        action, state = MeanFractionDecoder(_Model(probs)).predict(np.zeros(16))
        assert action[0] == k
        assert state[0].shape == (1, 1, 4)
