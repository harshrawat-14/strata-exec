"""
Action-index helpers over the canonical ACTION_FRACTIONS table.
"""

import numpy as np

from environment import ACTION_FRACTIONS


# ACTION_FRACTIONS is imported from environment.py above -- that is the one
# canonical Python-side query point (which itself queries the real Rust
# environment rather than hand-copying). Phase 2R finding: this project has
# had TWO separate hand-maintained mirrors of this table silently diverge
# (V5_FRACTIONS/V6_FRACTIONS here, and a third copy formerly in
# rl/environment.py) -- never hand-copy this table again.


def get_action_fractions(n_state_dims: int) -> list:
    """
    Return the action fraction list. n_state_dims is accepted (not used) only
    to keep every existing call site (TWAPPolicy(n_state_dims), etc.) working
    unmodified -- the action space does not depend on the state
    representation, so there is only ever one answer now.
    """
    return ACTION_FRACTIONS


def nearest_action(target_frac: float, fractions: list) -> int:
    """Map a desired sell fraction to the nearest discrete action."""
    target_frac = max(0.0, min(1.0, target_frac))
    diffs = [abs(f - target_frac) for f in fractions]
    return int(np.argmin(diffs))
