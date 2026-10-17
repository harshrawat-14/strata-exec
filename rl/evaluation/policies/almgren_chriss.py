"""
Almgren-Chriss optimal schedule (static and live-volatility adaptive).
"""

import numpy as np

from environment import CALIBRATED_SIGMA_REF, REGIME_DIMS_ARE_CONTINUOUS
from ..actions import get_action_fractions, nearest_action


# AC risk-aversion default used by the Python AC-family policies below.
# NOTE: the Rust `research-sim` AlmgrenChriss/Scheduler default is 1e-4
# (SimParams.lambda, src/bin/research_sim.rs), an order of magnitude
# different from this Python default. The two implementations are
# independent (see STRATEGIES audit finding) and were never reconciled.
# Both ACOptimalPolicy and AdaptiveACPolicy now accept `lam` as an override
# so callers (e.g. a Phase 1 lambda-grid sweep) aren't stuck with either
# hardcoded value -- but the *default* here is left at the pre-existing
# 0.001 rather than silently changed to match Rust, since changing a
# baseline's default risk-aversion would change every AC number in this
# file without an explicit decision to do so.
AC_DEFAULT_LAMBDA = 0.001


def _stable_sinh_ratio(kappa: float, T: int, t: np.ndarray) -> np.ndarray:
    """
    sinh(kappa*(T-t)) / sinh(kappa*T), computed without overflow for large
    kappa*T.

    Found via Phase 1's AC lambda-grid sweep test: at high lambda combined
    with AdaptiveACPolicy's live-volatility clamp (current_sigma up to 2.0,
    predict()), kappa*T can reach into the thousands -- sinh() overflows to
    inf, and inf/inf produces NaN, silently corrupting the AC schedule (a
    RuntimeWarning that a naive test could pass right past, exactly the
    "config that hides the interesting failure" pattern -- here the
    trajectory-level assertions still happened to hold on average, which is
    what makes it easy to miss).

    Numerically stable rewrite: sinh(x) = (e^x - e^-x)/2, so for x = kappa*t
    and y = kappa*T both large and positive,
        sinh(x)/sinh(y) = exp(x-y) * (1 - exp(-2x)) / (1 - exp(-2y))
    which stays in [0, 1] and never evaluates e^(anything > 0) for the
    dominant term. t ranges over [0, T), so x = kappa*(T-t) > 0 always.
    """
    x = kappa * (T - t)
    y = kappa * T
    if y > 20.0:
        # sinh(y) itself would already be ~e^20/2 here; the direct formula
        # is safe up to y~700 in float64, but this rewrite is exact in the
        # limit and avoids ever computing sinh() of a value that could grow
        # unboundedly across future callers.
        return np.exp(x - y) * (1.0 - np.exp(-2.0 * x)) / (1.0 - np.exp(-2.0 * y))
    denom = np.sinh(y) + 1e-10
    return np.sinh(x) / denom


class ACOptimalPolicy:
    """
    Almgren-Chriss closed-form optimal execution trajectory.
    Pre-computes the full AC schedule at initialisation using:
      x(t) = X0 * sinh(kappa*(T-t)) / sinh(kappa*T)
    """

    def __init__(self, n_state_dims: int = 12,
                 sigma: float = 0.20,
                 lam: float = AC_DEFAULT_LAMBDA,
                 eta: float = 0.10,
                 total_steps: int = 2880):
        self.fractions = get_action_fractions(n_state_dims)
        self.total_steps = total_steps
        self.kappa = np.sqrt(lam * sigma**2 / (eta + 1e-10))
        self._schedule = self._build_schedule()

    def _build_schedule(self) -> np.ndarray:
        """
        Precompute AC optimal sell fraction at each step.
        Returns array of shape (total_steps,) where entry t
        is the fraction of ORIGINAL inventory to sell at step t.
        Entries sum to 1.0.
        """
        T = self.total_steps
        k = self.kappa
        steps = np.arange(T)

        remaining_at_t = _stable_sinh_ratio(k, T, steps)
        remaining_at_t_plus_1 = np.concatenate([
            _stable_sinh_ratio(k, T, steps[1:]),
            [0.0]
        ])

        sell_frac = remaining_at_t - remaining_at_t_plus_1
        sell_frac = np.maximum(sell_frac, 0.0)

        total = sell_frac.sum()
        if total > 1e-10:
            sell_frac = sell_frac / total

        return sell_frac

    def predict(self, obs, state=None, episode_start=None,
                deterministic=True):
        q_frac = float(obs[0])
        t_frac = float(obs[1])

        if t_frac < 0.001 or q_frac < 0.001:
            action = nearest_action(1.0, self.fractions)
        else:
            step_idx = int((1.0 - t_frac) * self.total_steps)
            step_idx = min(step_idx, len(self._schedule) - 1)

            ac_original_frac = self._schedule[step_idx]

            if q_frac > 1e-6:
                target = min(1.0, ac_original_frac / q_frac)
            else:
                target = 1.0

            action = nearest_action(target, self.fractions)

        return np.array([action]), state


class AdaptiveACPolicy:
    """
    Receding-horizon Almgren-Chriss with live volatility re-estimation.
    Recomputes the AC schedule each step using the current
    volatility (from state[2]) rather than a fixed initial sigma.

    CORRECTION (Phase 3, 2026-08-13) -- every Adaptive-AC number produced
    before this date was generated by a policy reading the wrong volatility.
    Three compounding defects in the state->sigma reconstruction, all fixed:

    1. Wrong inversion formula. `state[2]` is `tanh(current_sigma/sigma_ref)`,
       but the code inverted it as `sigma_ref * (1.0 + arctanh(s2))` -- a
       spurious `1.0 +` apparently carried over from `s13`, which IS
       `tanh(ratio - 1.0)`. Pre-Step-0 this inflated the volatility read by
       ~8.7x (true ratio ~0.13 read as ~1.13).
    2. Hardcoded `sigma_ref = 0.20` (bug-class instance #5) that no call site
       overrode, so after Step 0 recalibrated the environment's sigma_ref to
       CALIBRATED_SIGMA_REF the reconstruction used a reference the
       environment had stopped using. Now queried from the Rust binary.
    3. Guardrails mis-scaled for the corrected ratios: the arctanh clip
       saturated at ratio 3.80 and the sigma clamp at 2.0, both of which bind
       precisely on the severe-crash range (2024-08-05 reaches ratio ~4.1,
       sigma ~2.05) -- capping this policy's regime response exactly where
       the project measures it.

    Consequence for the record: Adaptive-AC is a baseline in every comparison
    table in this project. Its numbers are not invalidated by construction --
    it was a consistently-parameterized policy, just not the one described --
    but no Adaptive-AC result should be cited as "AC with live volatility
    re-estimation" until re-measured. See docs/proposals/research-roadmap.md.
    """

    def __init__(self, n_state_dims: int = 12,
                 sigma_ref: float = None,
                 lam: float = AC_DEFAULT_LAMBDA,
                 eta: float = 0.10,
                 total_steps: int = 2880):
        # Defaults to the value queried from the Rust binary at import time,
        # never a hardcoded literal (bug-class instance #5 -- this default was
        # `0.20` and no call site overrode it, so it silently diverged from the
        # environment when Step 0 recalibrated sigma_ref). `None` means "use
        # whatever the binary in use actually runs"; passing an explicit value
        # is still supported for sweeps.
        if sigma_ref is None:
            sigma_ref = CALIBRATED_SIGMA_REF
        self.fractions = get_action_fractions(n_state_dims)
        self.total_steps = total_steps
        self.sigma_ref = sigma_ref
        self.lam = lam
        self.eta = eta
        self._ac_cache = {}   # cache (sigma_rounded) -> schedule

    def _get_schedule(self, sigma: float) -> np.ndarray:
        """Get or compute AC schedule for given sigma."""
        sigma_key = round(sigma, 2)
        if sigma_key not in self._ac_cache:
            kappa = np.sqrt(
                self.lam * sigma_key**2 / (self.eta + 1e-10)
            )
            T = self.total_steps
            steps = np.arange(T)
            remaining = _stable_sinh_ratio(kappa, T, steps)
            next_rem = np.concatenate([
                _stable_sinh_ratio(kappa, T, steps[1:]),
                [0.0]
            ])
            sell_frac = np.maximum(remaining - next_rem, 0.0)
            total = sell_frac.sum()
            if total > 1e-10:
                sell_frac /= total
            self._ac_cache[sigma_key] = sell_frac
        return self._ac_cache[sigma_key]

    # Numerically safe clip for arctanh. |x| this close to 1 still inverts
    # cleanly in float64; the value bounds amplification without truncating
    # real signal (see _recover_sigma).
    _ARCTANH_CLIP = 0.9999999

    def _recover_sigma(self, obs) -> float:
        """
        Reconstruct the environment's `current_sigma` from the state vector.

        Prefers s10 = tanh(ln(sigma/sigma_ref)) over s2 = tanh(sigma/sigma_ref)
        because s2 saturates where it matters. Own-impact-driven volatility on
        2024-08-05 reaches sigma ~6.57 (ratio ~13), and tanh(13) is 1.0 to
        within 6e-12 -- s2 cannot represent it, so inverting s2 caps out near
        ratio 8.4 (sigma ~4.2) and understates the true value by ~36% on the
        single most severe date in the dataset. s10 applies the log FIRST, so
        the same ratio encodes as tanh(2.58) = 0.988 and inverts exactly;
        after clipping it recovers ratios into the thousands.

        s10 is only the continuous encoding on post-Step-0 binaries. Against a
        pre-Step-0 binary -- required for evaluating the RL16/RL12 checkpoints,
        see rl/results/PROVENANCE.md -- s10 is the old dead 3-way bucket, so
        this falls back to s2. That path carries the saturation ceiling above;
        it is the best available from that binary's state vector, not a
        preference.
        """
        if REGIME_DIMS_ARE_CONTINUOUS:
            ratio = float(np.exp(np.arctanh(
                np.clip(float(obs[10]), -self._ARCTANH_CLIP, self._ARCTANH_CLIP)
            )))
        else:
            # NOTE: s2 is tanh(ratio), NOT tanh(ratio - 1.0) -- that is s13, a
            # different dim. This inversion previously carried a spurious
            # `1.0 +` borrowed from s13's definition (see class docstring).
            ratio = float(np.arctanh(
                np.clip(float(obs[2]), -self._ARCTANH_CLIP, self._ARCTANH_CLIP)
            ))
        # Backstop only, deliberately wide: real sigma is observed above 6.5 on
        # 2024-08-05 once own impact feeds back into realized volatility. The
        # previous 2.0 clamp bound hard on exactly that date, capping the crash
        # signal this policy exists to react to.
        return max(0.01, min(self.sigma_ref * ratio, 20.0))

    def predict(self, obs, state=None, episode_start=None,
                deterministic=True):
        q_frac    = float(obs[0])
        t_frac    = float(obs[1])
        if t_frac < 0.001 or q_frac < 0.001:
            action = nearest_action(1.0, self.fractions)
        else:
            current_sigma = self._recover_sigma(obs)

            schedule = self._get_schedule(current_sigma)
            step_idx = int((1.0 - t_frac) * self.total_steps)
            step_idx = min(step_idx, len(schedule) - 1)

            ac_original_frac = schedule[step_idx]
            if q_frac > 1e-6:
                target = min(1.0, ac_original_frac / q_frac)
            else:
                target = 1.0

            action = nearest_action(target, self.fractions)

        return np.array([action]), state
