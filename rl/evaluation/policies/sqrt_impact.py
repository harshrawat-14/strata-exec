"""
Numerically-optimized sqrt-impact baseline, fit against the real simulator.
"""

import json
import subprocess

from scipy.optimize import minimize as scipy_minimize
from scipy.special import betainc
import numpy as np

from environment import RL_ENV_BIN, StrataExecEnv
from ..actions import schedule_target, get_action_fractions, nearest_action
from ..episode import _run_episode
from ..protocol import fit_start_fracs


def query_calibration(date: str, agg_date: str = None, btc_target: float = 50_000.0) -> dict:
    """
    Read back the per-date calibrated impact_coefficient (Y), sigma, adv_btc,
    and decay/split constants that `StrataExecEnv`'s counterfactual mode
    actually uses for `date` -- via `rl-env --print-calibration`, i.e. the
    same `CounterfactualLobConfig::calibrated` call reset_counterfactual()
    makes, not a second hand-copied estimate (see "canonical values" rule,
    docs/proposals/research-roadmap.md). Used to record what SqrtImpactOptimalPolicy was
    actually fit against, for reporting -- not required for the fit itself,
    which runs the real environment end-to-end and never needs these numbers
    directly in Python.
    """
    agg_date = agg_date or date
    result = subprocess.run(
        [
            RL_ENV_BIN, "--rl-mode", "counterfactual",
            "--lob-data", f"TradeData/BookDepth/BTCUSDT-bookDepth-{date}.csv",
            "--agg-trades", f"TradeData/AggTrades/BTCUSDT-aggTrades-{agg_date}.csv",
            "--btc-target", str(btc_target),
            "--print-calibration",
        ],
        capture_output=True, text=True, check=True,
        stdin=subprocess.DEVNULL, timeout=120,
    )
    return json.loads(result.stdout.strip())


class SqrtImpactOptimalPolicy:
    """
    Numerically-optimized execution schedule under the TRUE calibrated
    square-root / transient-decay impact model this project's counterfactual
    environment actually runs (`src/market/counterfactual_lob.rs`), instead
    of AC-Optimal's closed-form linear-impact assumption.

    There is no closed form for the optimal trajectory here: instantaneous
    impact is concave (sqrt of participation, not linear), decay is
    exponential (OW-style), and fills walk a real, per-date order book --
    reimplementing that mechanism analytically in Python would be exactly
    this project's recurring "hand-copied canonical value" bug class (see
    "Standing rules", docs/proposals/research-roadmap.md), with no way to catch a silent
    mismatch against the simulator RL is actually scored in.

    So this fits directly against the real simulator instead: it searches a
    2-parameter Beta-CDF selling-curve family (a=b=1 reduces to TWAP; a<b
    front-loads, a>b back-loads -- a strict superset of AC-Optimal's
    monotone-decay shape) by running real episodes through `StrataExecEnv`
    and minimizing realized mean IS with Nelder-Mead, then freezes the
    resulting schedule exactly like `ACOptimalPolicy` freezes its closed-form
    one. This is the item-5 "numerically-optimized sqrt-impact baseline"
    from docs/proposals/research-roadmap.md Step 3.

    FIT-SEED PLACEMENT IS LOAD-BEARING -- found empirically while building
    this (2026-08-21/22), not a theoretical concern. `reset_counterfactual`
    picks `start_offset = seed % max_start` within the first third of the
    day, and on 2024-08-05 this project's most severe date, mean_IS at a
    FIXED policy varies by more than an order of magnitude depending on
    which few episodes happen to get sampled: TWAP alone measured -0.66%
    (seeds 900000+, the first attempt), -2.7% (seeds 500+), -4.0% (seeds
    5000+), -8.1% to -9.5% (seeds 20+ and seeds 0-19, the actual comparison
    table's convention) on the SAME date with the SAME schedule. This is a
    much sharper, quantified version of the audit's finding 5 ("eval
    episodes are near-duplicates, crash-date statistics are not credible",
    docs/methodology/platform-audit.md) -- the seed=0..n_episodes-1 convention
    every strategy in this file's comparison table is scored against
    already samples one arbitrary, narrow, near-duplicate slice of a day
    whose true outcome distribution is far wider. A DISTANT fit-seed range
    (the original design here used seed 900_000+) fits against a different,
    unrelated slice of that distribution and then gets scored against a
    slice it was never optimized for -- not overfitting in the usual sense,
    just optimizing for the wrong target. Fitting on seeds immediately
    ADJACENT to the eval range (empirically the same near-duplicate regime,
    confirmed on 2024-08-05) is the safest available mitigation without
    redesigning the environment's start-offset sampling -- hence
    `n_eval_episodes`, not a fixed constant, controls where the fit range
    starts.
    """

    def __init__(self, date: str, n_eval_episodes: int,
                 n_state_dims: int = 16,
                 mode: str = "counterfactual",
                 btc_target: float = 50_000.0,
                 fixed_steps: bool = True,
                 fixed_size: bool = True,
                 total_steps: int = 2880,
                 n_fit_seeds: int = 3,
                 maxiter: int = 25,
                 verbose: bool = False,
                 episode_steps: int = None):
        self.fractions = get_action_fractions(n_state_dims)
        # Protocol v3: the schedule spans the fixed episode length, and fit
        # episodes start between the eval starts (see evaluation/protocol.py).
        self.episode_steps = episode_steps
        # v3 baselines follow their schedule closed-loop (actions.schedule_target).
        self.track_schedule = bool(episode_steps)
        self.total_steps = episode_steps if episode_steps else total_steps
        self.date = date
        # Immediately after the eval range (0..n_eval_episodes-1) -- disjoint
        # (no leakage) and, empirically, the same near-duplicate intraday
        # regime the eval range itself samples (see class docstring).
        self._fit_seed_base = n_eval_episodes
        self._shape = (1.0, 1.0)
        self._schedule = self._build_schedule(1.0, 1.0)  # TWAP until fit() runs
        self.a, self.b, self.fit_mean_is = self._fit(
            date=date, n_state_dims=n_state_dims, mode=mode,
            btc_target=btc_target, fixed_steps=fixed_steps,
            fixed_size=fixed_size, n_fit_seeds=n_fit_seeds,
            maxiter=maxiter, verbose=verbose,
        )
        self._shape = (self.a, self.b)
        self._schedule = self._build_schedule(self.a, self.b)

    def set_horizon(self, total_steps: int) -> None:
        """Rebuild the current shape for the episode length the environment runs."""
        if int(total_steps) != self.total_steps:
            self.total_steps = int(total_steps)
            self._schedule = self._build_schedule(*self._shape)

    def _build_schedule(self, a: float, b: float) -> np.ndarray:
        """
        Per-step ORIGINAL-inventory sell fractions from the Beta(a, b) CDF,
        same array shape/semantics as ACOptimalPolicy._schedule (entries sum
        to 1.0, indexed by elapsed step).
        """
        T = self.total_steps
        t = np.arange(T + 1) / T
        cdf = betainc(a, b, t)
        sell_frac = np.diff(cdf)
        sell_frac = np.maximum(sell_frac, 0.0)
        total = sell_frac.sum()
        if total > 1e-10:
            sell_frac = sell_frac / total
        return sell_frac

    def _fit(self, date, n_state_dims, mode, btc_target, fixed_steps,
              fixed_size, n_fit_seeds, maxiter, verbose):
        env_kwargs = dict(
            mode=mode, adversarial=False, lob_date=date, agg_date=date,
            n_state_dims=n_state_dims, fixed_steps=fixed_steps,
            fixed_size=fixed_size, no_time_features=n_state_dims < 16,
            episode_steps=self.episode_steps,
        )
        if mode == "counterfactual":
            env_kwargs["btc_target"] = btc_target
        env = StrataExecEnv(**env_kwargs)
        fit_fracs = fit_start_fracs(n_fit_seeds) if self.episode_steps else [None] * n_fit_seeds

        def objective(log_params) -> float:
            a = float(np.clip(np.exp(log_params[0]), 0.05, 30.0))
            b = float(np.clip(np.exp(log_params[1]), 0.05, 30.0))
            self._shape = (a, b)
            self._schedule = self._build_schedule(a, b)

            is_vals = []
            for i in range(n_fit_seeds):
                is_pct, _, _, _ = _run_episode(
                    self, env, seed=self._fit_seed_base + i, start_frac=fit_fracs[i]
                )
                if is_pct is not None:
                    is_vals.append(is_pct)
            mean_is = float(np.mean(is_vals)) if is_vals else -1e6
            if verbose:
                print(f"    [SqrtImpactFit {date}] a={a:.3f} b={b:.3f} "
                      f"mean_IS={mean_is:.4f}")
            return -mean_is  # Nelder-Mead minimizes; higher IS is better.

        try:
            # x0=(0,0) -> a=b=1 (TWAP): an uninformative, unbiased starting
            # shape rather than one that presupposes front- or back-loading.
            # Explicit initial simplex: scipy's default perturbation for a
            # zero coordinate is a fixed 0.00025 (see `_minimize_neldermead`),
            # far too small a log-space step to move any of the 2880
            # discretized per-step schedule fractions -- the simplex
            # immediately reads as converged (three vertices with identical
            # realized IS) before it has explored anything. +-0.5 in
            # log-space (~0.6x-1.6x in a/b) is large enough to visibly change
            # the schedule shape between simplex vertices.
            x0 = np.array([0.0, 0.0])
            initial_simplex = np.array([
                x0,
                x0 + np.array([0.5, 0.0]),
                x0 + np.array([0.0, 0.5]),
            ])
            result = scipy_minimize(
                objective, x0=x0, method="Nelder-Mead",
                options={
                    "maxiter": maxiter, "xatol": 0.05, "fatol": 1e-3,
                    "initial_simplex": initial_simplex,
                },
            )
            a = float(np.clip(np.exp(result.x[0]), 0.05, 30.0))
            b = float(np.clip(np.exp(result.x[1]), 0.05, 30.0))
            mean_is = float(-result.fun)
        finally:
            env.close()

        return a, b, mean_is

    def predict(self, obs, state=None, episode_start=None,
                deterministic=True):
        q_frac = float(obs[0])
        t_frac = float(obs[1])

        if t_frac < 0.001 or q_frac < 0.001:
            action = nearest_action(1.0, self.fractions)
        else:
            target = schedule_target(self._schedule, q_frac, t_frac,
                                     self.total_steps, self.track_schedule)
            action = nearest_action(target, self.fractions)

        return np.array([action]), state
