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


def schedule_target(schedule, q_frac: float, t_frac: float, total_steps: int,
                    track: bool = False) -> float:
    """
    Sell fraction of REMAINING inventory for a per-step schedule given as
    fractions of the ORIGINAL order (entries sum to 1).

    Open loop (legacy protocol): sell this step's scheduled share of the
    original order. With a coarse action table the rounding error is never
    corrected -- at honest lambda the AC baseline's real pace was the 0.1%
    bucket, not its schedule, and it finished only because the legacy horizon
    was long enough.

    Tracking (protocol v3): sell whatever returns inventory to the scheduled
    path after this step, so each step corrects the previous rounding.
    """
    if q_frac <= 1e-6:
        return 1.0
    if track:
        k = min(int(round((1.0 - t_frac) * total_steps)), len(schedule) - 1)
        target_after = max(0.0, 1.0 - float(np.sum(schedule[:k + 1])))
        return max(0.0, (q_frac - target_after) / q_frac)
    step_idx = min(int((1.0 - t_frac) * total_steps), len(schedule) - 1)
    return min(1.0, schedule[step_idx] / q_frac)
