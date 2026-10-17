"""
Learning-rate schedules for RecurrentPPO.
"""

import math


def cosine_with_restarts(progress_remaining: float) -> float:
    """
    Cosine annealing with 3 cycles over full training.
    progress_remaining: 1.0 at step 0, 0.0 at final step.

    Cycle boundaries (progress elapsed):
      Cycle 0: 0%  -> 33%  peak multiplier = 1.000
      Cycle 1: 33% -> 66%  peak multiplier = 0.500
      Cycle 2: 66% -> 100% peak multiplier = 0.250

    Each cycle: cosine decay from peak to ~0.
    Halving peaks prevents late-training instability while
    still allowing meaningful exploration after each restart.
    """
    progress = 1.0 - progress_remaining
    cycle_length = 1.0 / 3.0
    cycle_num = min(int(progress / cycle_length), 2)
    cycle_progress = (progress - cycle_num * cycle_length) / cycle_length
    peak = 0.5 ** cycle_num
    return peak * 0.5 * (1.0 + math.cos(math.pi * cycle_progress))



def get_lr_schedule(lr_schedule: str, base_lr: float, total_timesteps: int):
    """
    Map the --lr-schedule CLI choice to what RecurrentPPO's learning_rate
    argument expects: either a constant float, or a callable
    (progress_remaining: float) -> float, where progress_remaining goes
    1.0 -> 0.0 over training regardless of total_timesteps (SB3's own
    convention -- total_timesteps is accepted here for interface symmetry
    with the other two branches and isn't otherwise needed).

    Regression note (Phase 2): this function was called but never defined
    anywhere in this file -- train() raised NameError on the first line
    that needed a learning rate, meaning training had never been run to
    completion in this state. cosine_restarts here reuses
    cosine_with_restarts (already implemented, already exercised at module
    import time by the assertions above) scaled by base_lr.
    """
    if lr_schedule == "constant":
        return base_lr
    if lr_schedule == "linear":
        return lambda progress_remaining: base_lr * progress_remaining
    if lr_schedule == "cosine_restarts":
        return lambda progress_remaining: base_lr * cosine_with_restarts(progress_remaining)
    raise ValueError(
        f"unknown lr_schedule {lr_schedule!r}; expected one of "
        f"'constant', 'linear', 'cosine_restarts'"
    )
