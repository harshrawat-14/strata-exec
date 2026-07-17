"""
THE HONEST FAILURE ANALYSIS.
This is the core research deliverable of Phase 4.
"""

from sb3_contrib import RecurrentPPO
from environment import StrataExecEnv
from collections import Counter
from scipy import stats
import numpy as np
import pandas as pd
import argparse
import os
import io
import json
import subprocess
import zipfile
import tempfile
import torch
import mlflow

TEST_DATES = {
    "2024-04-15": "BTC correction",
    "2024-07-10": "mid-summer sideways",
    "2024-12-10": "December bull run",
}

DATES = {
    "2024-01-15": "calm bull",
    # Was labeled "BTC breakout" -- see HELD_OUT_CRASH_DATES below: this is
    # actually the cleanest liquidation-cascade signature in the dataset.
    "2024-03-05": "crash - flash liquidation cascade",
    "2024-06-10": "quiet consolidation",
    "2024-08-05": "crash - Yen unwind",
    "2024-11-06": "post-election surge",
}

# ── Phase 0 canonical train/eval date split (Global Rule 9) ───────────────
# Declared here so Phase 2's retrain has one source of truth; enforced by
# an executable guard (rl/tests/test_date_split.py) rather than left as
# prose. Do NOT retrain in Phase 0 -- this only declares the split.
#
# Stress-day survey: of the 11 dates with clean BookDepth+AggTrades data,
# only two show a genuine liquidation-cascade signature -- a large adverse
# intraday move AND book depth thinning coincident (within ~1 min) with the
# price drop, not just high volume or volatility:
#   2024-08-05  -15.9% intraday, well-documented Yen-unwind carry-trade
#               crash (mandatory holdout).
#   2024-03-05  -13.5% intraday, the cleanest signature of any date: the
#               bid-depth minimum and the price minimum are 47 seconds
#               apart, and depth drops to 22% of its daily median right at
#               the crash. Previously mislabeled "BTC breakout" above.
# The README's previously-documented --multi-date training invocation
# included BOTH of these dates, meaning any existing "RL beats AC on the
# crash day" claim is unverified until a model is retrained on this split
# (Global Rule 9) -- see README for the corresponding flag on that claim.
#
# 2024-04-15 (-5.1% intraday, moderate but real coincident depth thinning,
# ratio 0.63) is the next most-stressed date and is the only remaining
# candidate that shows genuine stress rather than plain volatility (checked
# and rejected: 2024-02-20, 2024-09-15, 2024-10-15, 2024-12-10 all show
# depth-thinning ratios >= 0.9 at their own price minimum -- volatile but
# not a liquidity event). It is moved from TEST_DATES into TRAINING_DATES
# below (approved 2026-07-11) so the training set retains at least one
# genuine crash-like day without touching either holdout.
HELD_OUT_CRASH_DATES = ["2024-08-05", "2024-03-05"]

TRAINING_DATES = [
    "2024-01-15", "2024-02-20", "2024-04-15", "2024-06-10",
    "2024-09-15", "2024-10-15", "2024-11-06",
]

assert not (set(TRAINING_DATES) & set(HELD_OUT_CRASH_DATES)), (
    "TRAINING_DATES / HELD_OUT_CRASH_DATES overlap -- a held-out crash "
    "date would be trained on, invalidating the crash-day comparison"
)

# ── DEPRECATED for cross-strategy comparison ──────────────────────────────
# These four dicts are point estimates from the Rust `research-sim` binary's
# Monte Carlo (GBM/GARCH) pipeline with a square-root impact model -- a
# DIFFERENT simulator than the counterfactual LOB replay
# (evaluate_all_strategies_on_date) that produces RL's numbers below. Putting
# these side-by-side with live RL/LOB numbers in one table is the exact
# mixed-simulator bug Phase 0 exists to fix: it is not a valid comparison.
#
# They are retained ONLY as historical reference values, gated behind
# `legacy_mixed_comparison=True` in run_full_analysis (off by default).
# For any real strategy comparison, use evaluate_all_strategies_on_date,
# which runs every strategy through the identical counterfactual LOB path.
STATIC_OPTIMAL_IS = {
    "2024-01-15": -0.9900,
    "2024-03-05": -1.2275,
    "2024-06-10": -0.8684,
    "2024-08-05": -1.0714,
    "2024-11-06": -1.2122,
}
ADAPTIVE_OPTIMAL_IS = {
    "2024-01-15": -0.4935,
    "2024-03-05": -0.5861,
    "2024-06-10": -0.4494,
    "2024-08-05": -0.5854,
    "2024-11-06": -0.5816,
}

TWAP_IS = {
    "2024-01-15": 1.9869,
    "2024-03-05": 2.0544,
    "2024-06-10": 1.9491,
    "2024-08-05": 1.9156,
    "2024-11-06": 1.9261,
}

HEURISTIC_IS = {
    "2024-01-15": 1.8797,
    "2024-03-05": 1.9531,
    "2024-06-10": 1.8417,
    "2024-08-05": 1.8078,
    "2024-11-06": 1.8189,
}

def _query_action_fractions() -> list:
    """
    Single source of truth for the action->fraction mapping: query the real
    Rust environment (`ACTION_FRACTIONS` in rl_env.rs) rather than maintaining
    a hand-copied mirror in Python.

    Phase 2R finding: this file used to hand-maintain TWO separate mirrors
    (V5_FRACTIONS/V6_FRACTIONS, keyed off n_state_dims) that silently
    diverged from the real table -- one matched at only 3/20 indices. The
    classical-baseline policies (TWAP/Heuristic/AC-Optimal/AdaptiveAC) were
    picking action indices under a fraction assumption the environment never
    honored, for every n_state_dims=12 evaluation. The action space is a
    property of the environment, not of the state representation being
    evaluated -- it should never have been keyed off n_state_dims at all.
    Second instance of a hand-maintained mirror of a canonical value causing
    a silent bug in this project (cf. the CVaR sign inversion) -- canonical
    values get read from their single source, never copied.
    """
    result = subprocess.run(
        ["./target/release/rl-env", "--print-action-fractions"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout.strip())


ACTION_FRACTIONS = _query_action_fractions()


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


def compute_cvar_95(arr: np.ndarray) -> float:
    """
    CVaR-95: mean of the worst 5% of IS values (the tail mean, not a single
    percentile). Sign convention: more negative IS = worse execution cost
    (see _run_episode). Previously several call sites used
    np.percentile(arr, 5) here, which is VaR-5 (a single point on the
    distribution), not CVaR-95 (a tail mean) -- directionally correct but
    statistically the wrong quantity. This is the one implementation all
    evaluation functions should share.
    """
    if len(arr) == 0:
        return float("nan")
    n_tail = max(1, len(arr) // 20)
    sorted_asc = np.sort(arr)
    return float(np.mean(sorted_asc[:n_tail]))


def bootstrap_cvar_95_ci(
    arr: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Bootstrap confidence interval on the CVaR-95 estimate itself (Global
    Rule 4), not just on the mean. CVaR-95 on N episodes is the mean of only
    N//20 tail observations -- e.g. ~2-3 points on 50 episodes -- a very
    high-variance estimator whose point estimate alone is easy to
    over-trust. Resample the episode array with replacement n_bootstrap
    times, recompute compute_cvar_95 on each resample, and report the
    percentile interval of that distribution.

    Uses a local np.random.Generator (seeded, reproducible per Global Rule 2)
    rather than the global numpy RNG, so this never interacts with any other
    code's random state (e.g. test_vs_baseline's bootstrap, which still uses
    the legacy global np.random API).
    """
    n = len(arr)
    if n == 0:
        return {"point": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_episodes": 0, "n_tail": 0, "width": float("nan")}

    point = compute_cvar_95(arr)
    rng = np.random.default_rng(seed)
    resample_cvars = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(arr, size=n, replace=True)
        resample_cvars[i] = compute_cvar_95(resample)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(resample_cvars, 100 * alpha / 2))
    ci_upper = float(np.percentile(resample_cvars, 100 * (1 - alpha / 2)))
    n_tail = max(1, n // 20)

    return {
        "point": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_episodes": n,
        "n_tail": n_tail,
        "width": ci_upper - ci_lower,
    }


def test_vs_baseline(rl_is_samples: list, baseline_is: float) -> dict:
    """
    Test whether RL IS is significantly better than the static AC baseline.
    Returns t-statistic, one-tailed p-value, and 95% bootstrap CI.
    """
    arr = np.array(rl_is_samples)
    n = len(arr)

    if n == 0 or np.std(arr) < 1e-10:
        mean_val = float(arr.mean()) if n > 0 else float("nan")
        return {
            "t_stat": None,
            "p_value": None,
            "ci_lower": mean_val,
            "ci_upper": mean_val,
            "significantly_better": None,
            "note": "zero variance - deterministic policy",
        }

    t_stat, p_value = stats.ttest_1samp(arr, baseline_is)
    # One-tailed (RL better = higher IS than baseline).
    p_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    bootstrap_means = [
        np.mean(np.random.choice(arr, size=n, replace=True))
        for _ in range(10000)
    ]
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_one_tailed),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significantly_better": bool(p_one_tailed < 0.05),
    }


class TWAPPolicy:
    """
    Time-Weighted Average Price execution policy.
    Divides total inventory uniformly across all remaining steps.
    """

    def __init__(self, n_state_dims: int = 12):
        self.fractions = get_action_fractions(n_state_dims)
        self.total_steps = 2880  # default evaluation horizon

    def predict(self, obs, state=None, episode_start=None,
                deterministic=True):
        q_frac = float(obs[0])   # remaining inventory fraction
        t_frac = float(obs[1])   # time remaining fraction

        if t_frac < 0.001 or q_frac < 0.001:
            action = nearest_action(1.0, self.fractions)
        else:
            remaining_steps = max(1.0, t_frac * self.total_steps)
            target = min(1.0, 1.0 / remaining_steps)
            action = nearest_action(target, self.fractions)

        return np.array([action]), state


class HeuristicPolicy:
    """
    Volatility and liquidity-adjusted TWAP.
    Sells faster when volatility is high or the book is deep,
    slower when volatility is low or the book is thin.
    """

    def __init__(self, n_state_dims: int = 12,
                 vol_weight: float = 0.30,
                 depth_weight: float = 0.20):
        self.fractions = get_action_fractions(n_state_dims)
        self.total_steps = 2880
        self.vol_weight = vol_weight
        self.depth_weight = depth_weight

    def predict(self, obs, state=None, episode_start=None,
                deterministic=True):
        q_frac    = float(obs[0])
        t_frac    = float(obs[1])
        vol_ratio = float(obs[2])    # already tanh-normalised
        depth_ratio = float(obs[4])  # already tanh-normalised

        if t_frac < 0.001 or q_frac < 0.001:
            action = nearest_action(1.0, self.fractions)
        else:
            remaining_steps = max(1.0, t_frac * self.total_steps)
            base = min(1.0, 1.0 / remaining_steps)

            vol_scalar   = 1.0 + self.vol_weight * vol_ratio
            depth_scalar = 1.0 + self.depth_weight * depth_ratio

            target = min(1.0, base * vol_scalar * depth_scalar)
            target = max(0.0, target)

            action = nearest_action(target, self.fractions)

        return np.array([action]), state


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
    """

    def __init__(self, n_state_dims: int = 12,
                 sigma_ref: float = 0.20,
                 lam: float = AC_DEFAULT_LAMBDA,
                 eta: float = 0.10,
                 total_steps: int = 2880):
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

    def predict(self, obs, state=None, episode_start=None,
                deterministic=True):
        q_frac    = float(obs[0])
        t_frac    = float(obs[1])
        vol_ratio_tanh = float(obs[2])  # tanh(current_vol/ref - 1)

        if t_frac < 0.001 or q_frac < 0.001:
            action = nearest_action(1.0, self.fractions)
        else:
            vol_ratio_raw = np.arctanh(
                np.clip(vol_ratio_tanh, -0.999, 0.999)
            )
            current_sigma = self.sigma_ref * (1.0 + vol_ratio_raw)
            current_sigma = max(0.01, min(current_sigma, 2.0))

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


def load_model(path: str, device: str = 'cpu', env=None):
    """
    Load a model. Supports both RecurrentPPO (with _CPULSTMWrapper patch)
    and standard PPO.

    device: defaults to 'cpu' (the original behavior, for eval call sites).
    train.py's --resume-from path passes the real training device (e.g.
    'mps') so a resumed run continues on the same device the interrupted
    run used, rather than being silently forced onto CPU.

    env: if given, attached at load time via RecurrentPPO.load(path, env=env)
    rather than a separate model.set_env(env) call afterward. This matters
    when resuming with a DIFFERENT n_envs than the checkpoint was trained
    with (e.g. reducing parallelism after a memory incident) -- SB3's
    set_env() hard-asserts the new env's num_envs matches the checkpoint's
    exactly and refuses to resize the rollout buffers; passing env directly
    to .load() is the path SB3 itself points to for changing n_envs on
    resume.
    """
    zip_path = path if path.endswith('.zip') else path + '.zip'

    # Check if this is a recurrent model (contains lstm layers in policy state dict)
    is_recurrent = False
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if 'policy.pth' in zf.namelist():
                sd = torch.load(io.BytesIO(zf.read('policy.pth')), map_location='cpu')
                if any('lstm' in k for k in sd.keys()):
                    is_recurrent = True
    except Exception:
        pass

    if is_recurrent:
        all_files = {}
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                all_files[name] = zf.read(name)

        # Target policy.pth explicitly — not pytorch_variables.pth
        sd = torch.load(
            io.BytesIO(all_files['policy.pth']), map_location='cpu')

        new_sd = {}
        for k, v in sd.items():
            nk = (k
                  .replace('lstm_actor.lstm.', 'lstm_actor.')
                  .replace('lstm_critic.lstm.', 'lstm_critic.'))
            new_sd[nk] = v

        buf = io.BytesIO()
        torch.save(new_sd, buf)
        all_files['policy.pth'] = buf.getvalue()

        with tempfile.TemporaryDirectory() as tmpdir:
            patched = os.path.join(tmpdir, 'model.zip')
            with zipfile.ZipFile(patched, 'w', zipfile.ZIP_DEFLATED) as zf_out:
                for name, data in all_files.items():
                    zf_out.writestr(name, data)
            return RecurrentPPO.load(patched, env=env, device=device)
    else:
        from stable_baselines3 import PPO
        return PPO.load(zip_path, env=env, device=device)


def _run_episode(model, env, seed: int):
    """
    Run one episode with correct RecurrentPPO LSTM state tracking.
    Returns (is_pct, episode_reward, forced_liquidation, actions_taken).
    """
    obs, info = env.reset(seed=seed)
    arrival_price = info.get("arrival_price", 100.0)

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    done = False
    ep_reward = 0.0
    fill_qtys = []
    fill_prices = []
    actions_taken = []

    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start = np.zeros((1,), dtype=bool)

        actions_taken.append(int(action))
        obs, reward, done, _, info = env.step(int(action))
        ep_reward += reward

        if info.get("fill_qty", 0) > 0:
            fill_qtys.append(info["fill_qty"])
            fill_prices.append(info["fill_price"])

    forced = info.get("forced_liquidation_qty", 0) > 0

    is_pct = info.get("total_IS_pct")
    if is_pct is not None:
        is_pct = -is_pct
    elif fill_qtys:
        total_qty = sum(fill_qtys)
        vwap = sum(q * p for q, p in zip(fill_qtys, fill_prices)) / total_qty
        is_pct = (vwap - arrival_price) / arrival_price * 100

    return is_pct, ep_reward, forced, actions_taken



def evaluate_model_on_date(
    model,
    date: str,
    adversarial: bool = False,
    n_episodes: int = 50,
    n_state_dims: int = 8,
) -> dict:

    env = StrataExecEnv(
        mode="historical",
        adversarial=adversarial,
        lob_date=date,
        agg_date=date,
        n_state_dims=n_state_dims,
        fixed_steps=True,
        fixed_size=True,
        no_time_features=n_state_dims < 16,
    )

    all_is = []
    all_rewards = []
    all_actions = []
    forced_liquidations = 0

    for ep in range(n_episodes):
        is_pct, ep_reward, forced, actions = _run_episode(
            model, env, seed=ep)
        if forced:
            forced_liquidations += 1
        if is_pct is not None:
            all_is.append(is_pct)
        all_rewards.append(ep_reward)
        all_actions.extend(actions)

    env.close()

    arr = np.array(all_is)

    action_counts = Counter(all_actions)
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        action_dist = {
            a: action_counts[a] / total_actions
            for a in sorted(action_counts.keys())
        }
        mean_action = sum(a * f for a, f in action_dist.items())
        action_entropy = -sum(
            f * np.log(f + 1e-10) for f in action_dist.values()
        )
    else:
        action_dist = {}
        mean_action = float("nan")
        action_entropy = 0.0

    return {
        "date": date,
        "mean_IS": float(np.mean(arr)),
        "std_IS": float(np.std(arr)),
        "var_IS": float(np.var(arr)),
        "cvar95": compute_cvar_95(arr),
        "mean_reward": float(np.mean(all_rewards)),
        "forced_liquidation_rate": forced_liquidations / n_episodes,
        "n_episodes": n_episodes,
        "is_samples": all_is,
        "action_distribution": action_dist,
        "mean_action": float(mean_action),
        "action_entropy": float(action_entropy),
    }


def evaluate_model_counterfactual(
    model,
    date: str,
    n_episodes: int = 50,
    btc_target: float = 50000.0,
    n_state_dims: int = 12,
) -> dict:
    env = StrataExecEnv(
        mode="counterfactual",
        lob_date=date,
        agg_date=date,
        btc_target=btc_target,
        n_state_dims=n_state_dims,
        fixed_steps=True,
        fixed_size=True,
        no_time_features=n_state_dims < 16,
    )

    all_is = []
    all_rewards = []
    forced_liquidations = 0

    for ep in range(n_episodes):
        is_pct, ep_reward, forced, _ = _run_episode(model, env, seed=ep)
        if forced:
            forced_liquidations += 1
        if is_pct is not None:
            all_is.append(is_pct)
        all_rewards.append(ep_reward)

    env.close()
    arr = np.array(all_is)
    return {
        "mean_IS": float(np.mean(arr)),
        "std_IS":  float(np.std(arr)),
        "var_IS":  float(np.var(arr)),
        "cvar95":  compute_cvar_95(arr),
        "forced_liquidation_rate": forced_liquidations / n_episodes,
    }


def evaluate_on_synthetic(model, n_episodes=200, n_state_dims=8) -> dict:
    env = StrataExecEnv(
        mode="synthetic",
        n_state_dims=n_state_dims,
        fixed_steps=True,
        fixed_size=True,
        no_time_features=n_state_dims < 16,
    )
    all_is = []
    all_rewards = []

    for ep in range(n_episodes):
        is_pct, ep_reward, _, _ = _run_episode(
            model, env, seed=ep + 10000)
        if is_pct is not None:
            all_is.append(is_pct)
        all_rewards.append(ep_reward)

    env.close()
    arr = np.array(all_is)
    return {
        "mean_IS": float(np.mean(arr)),
        "var_IS": float(np.var(arr)),
        "cvar95": compute_cvar_95(arr),
    }


def evaluate_all_strategies_on_date(
    rl_model,
    date: str,
    n_episodes: int = 50,
    n_state_dims: int = 12,
    adversarial: bool = False,
    fixed_steps: bool = True,
    fixed_size: bool = True,
    ac_lambda: float = AC_DEFAULT_LAMBDA,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
    mode: str = "historical",
    btc_target: float = 50_000.0,
) -> dict:
    """
    Evaluate all 5 strategies on the same LOB environment using the same
    transient-impact model (OW-style permanent/temporary decay structure,
    square-root instantaneous magnitude -- see README "Market Impact" for
    the Phase 2R naming correction), same real Binance price path, and same
    arrival price.

    Replaces the mixed-framework comparison where TWAP/Heuristic
    used research-sim (Monte Carlo + square-root impact) and RL
    used evaluate.py (real LOB + this transient-impact model).

    mode: "historical" (default, preserves every existing caller's behavior
    -- pure replay, no causal own-impact feedback, matches Phase 0/1's
    established harness) or "counterfactual" (real LOB + accumulated own
    impact -- what RL training actually uses). Phase 2R Task 4's 16-dim vs
    12-dim ablation uses "counterfactual" specifically: historical mode has
    no cf_lob, so state dims 8/9 (own-impact fractions) are structurally
    dead there, which would test the 16-dim arm with two of its own
    distinguishing features disabled -- not a fair test of the
    representation. See docs/phase2r_objective_bias_finding.md Gate 3 and
    rl/results/PROVENANCE.md for why this default stays "historical" (all
    pre-Phase-2R results/comparability are historical-mode) while Task 4
    explicitly opts into "counterfactual".

    ac_lambda: risk-aversion parameter shared by both AC-family policies
    (AC Optimal and Adaptive AC) in this comparison -- exposed here rather
    than left as a hardcoded default so the AC baseline isn't a strawman
    (lambda too low) or an unbeatable wall (lambda too high) chosen
    invisibly. See AC_DEFAULT_LAMBDA for the Rust/Python default mismatch
    this parameter exists to let a caller work around.

    n_bootstrap/bootstrap_seed: passed to bootstrap_cvar_95_ci (Global Rule
    4) -- CVaR-95 on n_episodes episodes is a high-variance estimator (the
    mean of only n_episodes//20 tail points), so every cvar_95 reported here
    carries its own bootstrap CI rather than being presented as a clean
    point estimate.
    """
    policies = {
        "twap": TWAPPolicy(n_state_dims),
        "heuristic": HeuristicPolicy(n_state_dims),
        "ac_optimal": ACOptimalPolicy(n_state_dims, lam=ac_lambda),
        "adaptive_ac": AdaptiveACPolicy(n_state_dims, lam=ac_lambda),
        "rl_agent": rl_model,
    }

    strategy_labels = {
        "twap":        "TWAP",
        "heuristic":   "Heuristic",
        "ac_optimal":  "AC Optimal",
        "adaptive_ac": "Adaptive AC",
        "rl_agent":    "RL Agent",
    }

    results = {}

    for strategy_key, policy in policies.items():
        print(f"    [{strategy_labels[strategy_key]}]", end=" ", flush=True)

        env_kwargs = dict(
            mode=mode,
            adversarial=adversarial,
            lob_date=date,
            agg_date=date,
            n_state_dims=n_state_dims,
            fixed_steps=fixed_steps,
            fixed_size=fixed_size,
            no_time_features=n_state_dims < 16,
        )
        if mode == "counterfactual":
            env_kwargs["btc_target"] = btc_target
        env = StrataExecEnv(**env_kwargs)

        all_is = []
        all_rewards = []
        forced_liq = 0
        all_actions = []

        for ep in range(n_episodes):
            is_pct, ep_reward, forced, actions = _run_episode(
                policy, env, seed=ep
            )
            if forced:
                forced_liq += 1
            if is_pct is not None:
                all_is.append(is_pct)
            all_rewards.append(ep_reward)
            all_actions.extend(actions)

        env.close()

        arr = np.array(all_is) if all_is else np.array([0.0])

        action_counts = Counter(all_actions)
        total_actions = sum(action_counts.values()) or 1
        action_dist = {
            a: action_counts[a] / total_actions
            for a in sorted(action_counts.keys())
        }

        cvar_95 = compute_cvar_95(arr)
        cvar_ci = bootstrap_cvar_95_ci(arr, n_bootstrap=n_bootstrap, seed=bootstrap_seed)

        results[strategy_key] = {
            "strategy": strategy_labels[strategy_key],
            "date": date,
            "mean_IS": float(np.mean(arr)),
            "std_IS": float(np.std(arr)),
            "var_IS": float(np.var(arr)),
            "cvar_95": cvar_95,
            "cvar_95_ci_lower": cvar_ci["ci_lower"],
            "cvar_95_ci_upper": cvar_ci["ci_upper"],
            "cvar_95_ci_width": cvar_ci["width"],
            "cvar_95_n_tail": cvar_ci["n_tail"],
            "is_99th_pct": float(np.percentile(arr, 1)),
            "is_sharpe": float(np.mean(arr) / (np.std(arr) + 1e-10)),
            "mean_reward": float(np.mean(all_rewards)),
            "forced_liquidation_rate": forced_liq / n_episodes,
            "n_episodes": n_episodes,
            "is_samples": all_is,
            "action_distribution": action_dist,
            "mean_action": float(np.mean(all_actions)) if all_actions else 0.0,
        }

        print(f"IS={results[strategy_key]['mean_IS']:.4f}%  "
              f"CVaR95={cvar_95:.4f}% [{cvar_ci['ci_lower']:.4f}%, {cvar_ci['ci_upper']:.4f}%] "
              f"(tail n={cvar_ci['n_tail']})")

    return results


# Default risk-aversion grid for the AC baseline sweep (Phase 1 task 3a):
# spans patient (low lambda, trades slowly, more market risk) to urgent
# (high lambda, front-loads execution, more impact cost). Deliberately wider
# than a token 2-point grid so the reader can see AC_DEFAULT_LAMBDA's actual
# position on the curve, not just two endpoints.
AC_LAMBDA_GRID = [1e-5, 1e-4, AC_DEFAULT_LAMBDA, 1e-2, 1e-1]


def evaluate_ac_lambda_grid(
    date: str,
    lambdas: list = None,
    n_episodes: int = 50,
    n_state_dims: int = 12,
    fixed_steps: bool = True,
    fixed_size: bool = True,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
) -> dict:
    """
    Phase 1 task 3a: the AC baseline is not unique -- it depends on a
    risk-aversion parameter (see AC_DEFAULT_LAMBDA's Rust/Python mismatch
    note), and every p-value/comparison in this project is measured against
    it. Sweep AC Optimal and Adaptive AC across a lambda grid on the same
    counterfactual LOB episodes (same seeds) so a reader can see whether
    AC_DEFAULT_LAMBDA sits on a strawman (too patient), an unbeatable wall
    (too urgent), or somewhere reasonable in between -- rather than trusting
    a single hidden value.

    Only runs the two lambda-dependent policies (TWAP/Heuristic/RL don't
    depend on lambda and are already covered once by
    evaluate_all_strategies_on_date -- no need to re-run them per grid point).

    Returns {lambda: {"ac_optimal": {...}, "adaptive_ac": {...}}}, same
    per-strategy metrics shape as evaluate_all_strategies_on_date.
    """
    if lambdas is None:
        lambdas = AC_LAMBDA_GRID

    grid_results = {}

    for lam in lambdas:
        policies = {
            "ac_optimal": ACOptimalPolicy(n_state_dims, lam=lam),
            "adaptive_ac": AdaptiveACPolicy(n_state_dims, lam=lam),
        }
        strategy_labels = {"ac_optimal": "AC Optimal", "adaptive_ac": "Adaptive AC"}
        lam_result = {}

        for strategy_key, policy in policies.items():
            print(f"    [lambda={lam:.0e} {strategy_labels[strategy_key]}]", end=" ", flush=True)

            env = StrataExecEnv(
                mode="historical",
                lob_date=date,
                agg_date=date,
                n_state_dims=n_state_dims,
                fixed_steps=fixed_steps,
                fixed_size=fixed_size,
                no_time_features=n_state_dims < 16,
            )

            all_is = []
            for ep in range(n_episodes):
                is_pct, _ep_reward, _forced, _actions = _run_episode(policy, env, seed=ep)
                if is_pct is not None:
                    all_is.append(is_pct)
            env.close()

            arr = np.array(all_is) if all_is else np.array([0.0])
            cvar_95 = compute_cvar_95(arr)
            cvar_ci = bootstrap_cvar_95_ci(arr, n_bootstrap=n_bootstrap, seed=bootstrap_seed)

            lam_result[strategy_key] = {
                "strategy": strategy_labels[strategy_key],
                "lambda": lam,
                "date": date,
                "mean_IS": float(np.mean(arr)),
                "std_IS": float(np.std(arr)),
                "cvar_95": cvar_95,
                "cvar_95_ci_lower": cvar_ci["ci_lower"],
                "cvar_95_ci_upper": cvar_ci["ci_upper"],
                "is_sharpe": float(np.mean(arr) / (np.std(arr) + 1e-10)),
                "n_episodes": n_episodes,
                "is_samples": all_is,
            }
            print(f"IS={lam_result[strategy_key]['mean_IS']:.4f}%  CVaR95={cvar_95:.4f}%")

        grid_results[lam] = lam_result

    return grid_results


def print_unified_comparison(all_results: dict):
    """
    Print full strategy comparison table from unified evaluation.
    all_results: {date: {strategy_key: metrics_dict}}
    """
    STRATEGY_ORDER = [
        "twap", "heuristic", "ac_optimal", "adaptive_ac", "rl_agent"
    ]
    STRATEGY_LABELS = {
        "twap":        "TWAP",
        "heuristic":   "Heuristic",
        "ac_optimal":  "AC Optimal",
        "adaptive_ac": "Adaptive AC",
        "rl_agent":    "RL Agent",
    }

    print("\n" + "=" * 90)
    print("UNIFIED COMPARISON - ALL STRATEGIES IN COUNTERFACTUAL LOB")
    print("Same OW impact model | Same real Binance price path | Same arrival price")
    print("More negative IS = higher execution cost | CVaR-95 = avg of worst 5% episodes")
    print("=" * 90)

    for date, date_results in all_results.items():
        if not date_results:
            continue

        regime = DATES.get(date) or TEST_DATES.get(date, "unknown")
        print(f"\n{'-' * 70}")
        print(f"  {date}  ({regime})")
        print(f"{'-' * 70}")

        header = (f"  {'Strategy':<18} {'Mean IS':>9} {'CVaR-95':>9} "
                  f"{'IS Sharpe':>10} {'Forced Liq':>11}")
        print(header)
        print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*10} {'-'*11}")

        rl_is = date_results.get("rl_agent", {}).get("mean_IS")

        for key in STRATEGY_ORDER:
            r = date_results.get(key)
            if r is None:
                continue

            mean_is = r["mean_IS"]
            cvar    = r["cvar_95"]
            sharpe  = r["is_sharpe"]
            fl_rate = r["forced_liquidation_rate"]

            if rl_is is not None and key != "rl_agent":
                vs_rl = "RL better" if rl_is > mean_is else "RL worse"
                vs_rl_str = f"  <- {vs_rl}"
            else:
                vs_rl_str = ""

            label = STRATEGY_LABELS[key]
            if key == "rl_agent":
                label = f"* {label}"

            print(f"  {label:<18} {mean_is:>+9.4f}% {cvar:>+9.4f}% "
                  f"{sharpe:>+10.4f} {fl_rate:>10.1%}"
                  f"{vs_rl_str}")

    print("\n" + "=" * 90)
    print("CVAR-95 COMPARISON (lower = worse tail risk)")
    print("=" * 90)

    dates_list = list(all_results.keys())
    if dates_list:
        print(f"\n  {'Strategy':<18}", end="")
        for date in dates_list:
            print(f"  {date}", end="")
        print()
        print(f"  {'-'*18}", end="")
        for _ in dates_list:
            print(f"  {'-'*10}", end="")
        print()

        for key in STRATEGY_ORDER:
            label = STRATEGY_LABELS[key]
            print(f"  {label:<18}", end="")
            for date in dates_list:
                r = all_results[date].get(key)
                if r:
                    print(f"  {r['cvar_95']:>+9.4f}%", end="")
                else:
                    print(f"  {'N/A':>10}", end="")
            print()


def log_eval_results_to_mlflow(
    run_id: str,
    unified_results: dict,
    strategies: list = None,
) -> None:
    """
    Phase 3 (concurrent with Phase 2): log final per-date evaluation metrics
    to an EXISTING mlflow run by run_id, rather than starting a fresh
    untracked run. This is the piece that was missing entirely -- train.py
    has mlflow.start_run/log_params/log_artifact but zero log_metric calls,
    and until this function existed, evaluate.py had no mlflow integration
    at all, so a training run's final eval numbers (the ones that actually
    matter -- mean IS, CVaR-95 per date) were never attached to that run's
    tracked history. experiment_tracker.py's get_best_run() has always
    returned {} as a direct consequence: there was nothing to query.

    unified_results: {date: {strategy_key: metrics_dict}}, the return shape
    of evaluate_all_strategies_on_date collected across multiple dates (see
    run_full_analysis's --unified-comparison path, or
    rl/phase1_tail_risk_report.py).

    strategies: which strategy_keys to log (default: just "rl_agent", since
    for a training run's own tracked history the model's own performance is
    what's relevant -- the classical baselines don't change run to run and
    don't need re-logging against every model's run_id).

    Uses mlflow.start_run(run_id=..., nested=False) to REOPEN the existing
    run rather than create a new one -- this is the standard mlflow pattern
    for attaching metrics after the original `with mlflow.start_run():`
    block has already exited (e.g. because this is called from a separate
    process/script after training finished).
    """
    if strategies is None:
        strategies = ["rl_agent"]

    with mlflow.start_run(run_id=run_id):
        for date, date_results in unified_results.items():
            for strategy_key in strategies:
                r = date_results.get(strategy_key)
                if r is None:
                    continue
                safe_date = date.replace("-", "")
                mlflow.log_metric(f"eval_mean_is_{safe_date}_{strategy_key}", r["mean_IS"])
                mlflow.log_metric(f"eval_cvar95_{safe_date}_{strategy_key}", r["cvar_95"])
                if "cvar_95_ci_lower" in r:
                    mlflow.log_metric(f"eval_cvar95_ci_lower_{safe_date}_{strategy_key}", r["cvar_95_ci_lower"])
                    mlflow.log_metric(f"eval_cvar95_ci_upper_{safe_date}_{strategy_key}", r["cvar_95_ci_upper"])
                mlflow.log_metric(f"eval_forced_liq_{safe_date}_{strategy_key}", r["forced_liquidation_rate"])

        # Aggregate summary metrics across all logged dates, so
        # get_best_run(metric="eval_mean_is_avg", ...) has something to
        # compare across runs without needing to know every date's name.
        for strategy_key in strategies:
            means = [
                date_results[strategy_key]["mean_IS"]
                for date_results in unified_results.values()
                if strategy_key in date_results
            ]
            cvars = [
                date_results[strategy_key]["cvar_95"]
                for date_results in unified_results.values()
                if strategy_key in date_results
            ]
            if means:
                mlflow.log_metric(f"eval_mean_is_avg_{strategy_key}", float(np.mean(means)))
            if cvars:
                mlflow.log_metric(f"eval_cvar95_avg_{strategy_key}", float(np.mean(cvars)))


def run_full_analysis(
    passive_model_path: str,
    adversarial_model_path: str = None,
    n_episodes: int = 50,
    output_dir: str = "rl/results/",
    n_state_dims: int = 8,
    eval_counterfactual: bool = False,
    btc_target: float = 50000.0,
    include_test_dates: bool = False,
    unified_comparison: bool = True,
    legacy_mixed_comparison: bool = False,
    ac_lambda: float = AC_DEFAULT_LAMBDA,
):
    """
    unified_comparison (default True): run every strategy (TWAP, Heuristic,
        AC Optimal, Adaptive AC, RL) through the same counterfactual LOB
        path via evaluate_all_strategies_on_date -- the only methodologically
        valid cross-strategy comparison in this file.
    legacy_mixed_comparison (default False): also print/save the OLD
        strategy-comparison table that puts Monte-Carlo-sourced TWAP/
        Heuristic/AC constants (STATIC_OPTIMAL_IS etc.) next to live
        LOB-replay RL numbers. This mixes two different simulators and is
        not a valid comparison -- only enable it for historical/debugging
        reference, never to support a beats-AC claim.
    ac_lambda: risk-aversion for the AC-family baselines in the unified
        comparison (see AC_DEFAULT_LAMBDA).
    """
    os.makedirs(output_dir, exist_ok=True)

    passive_model = load_model(passive_model_path)
    adv_model = load_model(adversarial_model_path) \
                if adversarial_model_path else None

    print("\n" + "=" * 70)
    print("PHASE 4 - HONEST FAILURE ANALYSIS")
    print("RL Agent vs Static Optimal AC across 5 Market Regimes")
    print("=" * 70)

    print("\nEvaluating on synthetic (training environment)...")
    synth = evaluate_on_synthetic(passive_model, n_episodes=200,
                                  n_state_dims=n_state_dims)
    print(f"  Passive RL synthetic IS: {synth['mean_IS']:.4f}%")
    print(f"  Variance: {synth['var_IS']:.4f}")

    results = []

    for date, regime in DATES.items():
        print(f"\nEvaluating {date} ({regime})...")

        pr = evaluate_model_on_date(
            passive_model, date,
            adversarial=False,
            n_episodes=n_episodes,
            n_state_dims=n_state_dims)

        row = {
            "date": date,
            "regime": regime,
            "rl_passive_IS": pr["mean_IS"],
            "rl_passive_var": pr["var_IS"],
            "rl_passive_cvar95": pr["cvar95"],
            "rl_passive_forced_liq": pr["forced_liquidation_rate"],
            "rl_passive_degradation":
                pr["mean_IS"] - synth["mean_IS"],
            "mean_action": pr["mean_action"],
            "action_entropy": pr["action_entropy"],
            "action_distribution": pr["action_distribution"],
        }

        print(f"  Mean action: {pr['mean_action']:.2f}")
        print(f"  Action entropy: {pr['action_entropy']:.3f}")
        top3 = sorted(
            pr["action_distribution"].items(),
            key=lambda x: -x[1],
        )[:3]
        print(f"  Top 3 actions: {top3}")
        print(f"  RL IS (LOB replay): {pr['mean_IS']:.4f}%")

        if legacy_mixed_comparison:
            # WARNING: STATIC_OPTIMAL_IS is a Monte Carlo point estimate,
            # pr["is_samples"] are counterfactual-LOB-replay samples. This
            # t-test compares across two different simulators and is not a
            # methodologically valid significance test -- see
            # evaluate_all_strategies_on_date for the valid comparison.
            stat_test = test_vs_baseline(
                pr["is_samples"],
                STATIC_OPTIMAL_IS[date],
            )
            row.update({
                "twap_IS_MONTE_CARLO_DO_NOT_COMPARE": TWAP_IS.get(date),
                "heuristic_IS_MONTE_CARLO_DO_NOT_COMPARE": HEURISTIC_IS.get(date),
                "static_optimal_IS_MONTE_CARLO_DO_NOT_COMPARE": STATIC_OPTIMAL_IS[date],
                "adaptive_optimal_IS_MONTE_CARLO_DO_NOT_COMPARE": ADAPTIVE_OPTIMAL_IS[date],
                "p_value_vs_ac_INVALID_MIXED_SIMULATOR": stat_test["p_value"],
                "ci_lower": stat_test["ci_lower"],
                "ci_upper": stat_test["ci_upper"],
                "significantly_better_INVALID_MIXED_SIMULATOR":
                    stat_test["significantly_better"],
            })
            baseline = STATIC_OPTIMAL_IS[date]
            print(
                f"  [LEGACY, MIXED SIMULATORS -- NOT A VALID COMPARISON] "
                f"RL 95% CI [{stat_test['ci_lower']:.4f}%, "
                f"{stat_test['ci_upper']:.4f}%]"
            )
            if stat_test["p_value"] is None:
                print(
                    f"  [LEGACY] vs AC Monte Carlo ({baseline:.4f}%): "
                    f"p=N/A (zero variance — deterministic policy)"
                )
            else:
                sig = stat_test["significantly_better"]
                label = "SIGNIFICANT" if sig else "not significant"
                print(
                    f"  [LEGACY] vs AC Monte Carlo ({baseline:.4f}%): "
                    f"p={stat_test['p_value']:.4f} {label}"
                )

        if adv_model:
            ar = evaluate_model_on_date(
                adv_model, date,
                adversarial=False,
                n_episodes=n_episodes,
                n_state_dims=n_state_dims)
            row.update({
                "rl_adv_IS": ar["mean_IS"],
                "rl_adv_var": ar["var_IS"],
                "rl_adv_cvar95": ar["cvar95"],
                "rl_adv_forced_liq": ar["forced_liquidation_rate"],
                "rl_adv_degradation":
                    ar["mean_IS"] - synth["mean_IS"],
            })

        results.append(row)

    df = pd.DataFrame(results)

    if legacy_mixed_comparison:
        print("\n" + "=" * 90)
        print("[LEGACY, MIXED SIMULATORS -- NOT A VALID COMPARISON]")
        print("STRATEGY COMPARISON - IS% across 5 dates")
        print("TWAP/Heuristic/StatOpt/AdaptOpt are Monte Carlo point estimates;")
        print("RL-Pass is a live counterfactual-LOB-replay mean. Different")
        print("simulators -- see evaluate_all_strategies_on_date for the valid")
        print("apples-to-apples comparison instead.")
        print("=" * 90)

        header = (
            f"{'Date':<12} {'Regime':<22} "
            f"{'TWAP':>8} {'Heuristic':>10} "
            f"{'StatOpt':>8} {'AdaptOpt':>9} {'RL-Pass':>8}"
        )
        if adv_model:
            header += f" {'RL-Adv':>8}"
        print(header)
        print("-" * len(header))

        for _, row in df.iterrows():
            twap = row.get("twap_IS_MONTE_CARLO_DO_NOT_COMPARE")
            heur = row.get("heuristic_IS_MONTE_CARLO_DO_NOT_COMPARE")
            twap_s = f"{twap:>7.3f}%" if twap is not None else f"{'N/A':>8}"
            heur_s = (
                f"{heur:>9.3f}%" if heur is not None else f"{'N/A':>10}"
            )
            line = (
                f"{row['date']:<12} {row['regime']:<22} "
                f"{twap_s} {heur_s} "
                f"{row['static_optimal_IS_MONTE_CARLO_DO_NOT_COMPARE']:>7.3f}% "
                f"{row['adaptive_optimal_IS_MONTE_CARLO_DO_NOT_COMPARE']:>8.3f}% "
                f"{row['rl_passive_IS']:>7.3f}%"
            )
            if adv_model:
                line += f" {row['rl_adv_IS']:>7.3f}%"
            print(line)

    print("\n" + "=" * 70)
    print("SIM-TO-REAL DEGRADATION - RL (passive training)")
    print(f"Synthetic baseline IS: {synth['mean_IS']:.4f}%")
    print("=" * 70)

    for _, row in df.iterrows():
        deg = row['rl_passive_degradation']
        fl  = row['rl_passive_forced_liq']
        print(f"{row['date']} ({row['regime']}):")
        print(f"  IS degradation: {deg:+.4f}pp vs synthetic")
        print(f"  Forced liquidations: {fl:.1%} of episodes")
        print(f"  CVaR95: {row['rl_passive_cvar95']:.4f}%")

    print("\n" + "=" * 70)
    print("VARIANCE ANALYSIS - is RL reducing variance vs TWAP?")
    print("=" * 70)
    print("(Compare rl_passive_var against TWAP IS variance ~930M")
    print(" and Static Optimal IS variance from Phase 2 results)")

    for _, row in df.iterrows():
        print(f"{row['date']}: RL var = {row['rl_passive_var']:.2f}")

    csv_path = f"{output_dir}failure_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    if eval_counterfactual:
        print("\n" + "=" * 70)
        print("COUNTERFACTUAL EVALUATION (training environment)")
        print("=" * 70)

        cf_results = []
        for date, regime in DATES.items():
            r = evaluate_model_counterfactual(
                passive_model, date,
                n_episodes=n_episodes,
                btc_target=btc_target,
                n_state_dims=n_state_dims,
            )
            cf_results.append({
                "date": date,
                "regime": regime,
                "cf_IS": r["mean_IS"],
                "cf_var": r["var_IS"],
                "cf_std": r["std_IS"],
            })
            print(f"{date} ({regime}):")
            print(f"  IS: {r['mean_IS']:.4f}% +/- {r['std_IS']:.4f}%")
            print(f"  Variance: {r['var_IS']:.6f}")
            print(f"  CVaR95: {r['cvar95']:.4f}%")

        pd.DataFrame(cf_results).to_csv(
            f"{output_dir}counterfactual_eval.csv", index=False)

    if include_test_dates:
        print("\n" + "=" * 70)
        print("OUT-OF-SAMPLE TEST DATES (never used in training)")
        print("=" * 70)

        test_results = []
        for date, regime in TEST_DATES.items():
            lob_path = (f"TradeData/BookDepth/"
                        f"BTCUSDT-bookDepth-{date}.csv")
            if not os.path.exists(lob_path):
                print(f"  SKIPPING {date} -- file not found")
                continue

            r = evaluate_model_on_date(
                passive_model, date,
                n_episodes=n_episodes,
                n_state_dims=n_state_dims,
            )
            test_results.append({
                "date": date,
                "regime": regime,
                "RL_IS": r["mean_IS"],
                "RL_var": r["var_IS"],
            })
            print(f"{date} ({regime}): IS={r['mean_IS']:.4f}%")

        if test_results:
            pd.DataFrame(test_results).to_csv(
                f"{output_dir}test_dates_eval.csv", index=False)

    avg_degradation = df['rl_passive_degradation'].mean()
    worst_date = df.loc[
        df['rl_passive_degradation'].idxmax(), 'date']
    worst_regime = df.loc[
        df['rl_passive_degradation'].idxmax(), 'regime']

    print("\n" + "=" * 70)
    print("FINDING SUMMARY")
    print("=" * 70)
    print(f"Mean sim-to-real degradation: {avg_degradation:+.4f}pp")
    print(f"Worst date: {worst_date} ({worst_regime})")
    print(f"  This regime differs most from GARCH training data.")
    print(f"  Likely cause: "
          f"{'jump events' if 'crash' in worst_regime else 'vol regime mismatch'}")

    if adv_model:
        avg_adv_deg = df['rl_adv_degradation'].mean()
        improvement = avg_degradation - avg_adv_deg
        print(f"\nAdversarial training effect: {improvement:+.4f}pp")
        if improvement > 0:
            print("  Adversarial training reduces sim-to-real gap.")
        else:
            print("  Adversarial training did not reduce gap on passive book.")
            print("  Likely cause: adversarial agents don't match real market.")

    if unified_comparison:
        print("\n" + "=" * 70)
        print("RUNNING UNIFIED COMPARISON")
        print("Evaluating all 5 strategies in same counterfactual LOB...")
        print("=" * 70)

        unified_results = {}
        dates_to_evaluate = list(DATES.keys())
        if include_test_dates:
            dates_to_evaluate += [
                d for d in TEST_DATES
                if os.path.exists(
                    f"TradeData/BookDepth/BTCUSDT-bookDepth-{d}.csv"
                )
            ]

        for date in dates_to_evaluate:
            lob_path = (f"TradeData/BookDepth/"
                        f"BTCUSDT-bookDepth-{date}.csv")
            if not os.path.exists(lob_path):
                print(f"  SKIPPING {date} — LOB file not found")
                continue

            print(f"\n  Evaluating {date}...")
            date_results = evaluate_all_strategies_on_date(
                rl_model=passive_model,
                date=date,
                n_episodes=n_episodes,
                n_state_dims=n_state_dims,
                ac_lambda=ac_lambda,
            )
            unified_results[date] = date_results

        print_unified_comparison(unified_results)

        unified_rows = []
        for date, date_results in unified_results.items():
            for strategy_key, r in date_results.items():
                unified_rows.append({
                    "date": date,
                    "regime": DATES.get(date) or TEST_DATES.get(date, ""),
                    "strategy": r["strategy"],
                    "mean_IS": r["mean_IS"],
                    "std_IS": r["std_IS"],
                    "cvar_95": r["cvar_95"],
                    "is_99th_pct": r["is_99th_pct"],
                    "is_sharpe": r["is_sharpe"],
                    "forced_liquidation_rate": r["forced_liquidation_rate"],
                })

        unified_df = pd.DataFrame(unified_rows)
        unified_csv = f"{output_dir}unified_comparison.csv"
        unified_df.to_csv(unified_csv, index=False)
        print(f"\nUnified results saved: {unified_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--passive-model", required=True)
    p.add_argument("--adversarial-model", default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--n-state-dims", type=int, default=8)
    p.add_argument("--eval-counterfactual", action="store_true",
        help="Also evaluate in counterfactual LOB (training env)")
    p.add_argument("--btc-target", type=float, default=50000.0)
    p.add_argument("--include-test-dates", action="store_true",
        help="Also evaluate on held-out test dates")
    p.add_argument("--unified-comparison", action=argparse.BooleanOptionalAction,
        default=True,
        help="Run all 5 strategies in same counterfactual LOB for "
             "methodologically valid comparison (default: on). Use "
             "--no-unified-comparison to skip it.")
    p.add_argument("--legacy-mixed-comparison", action="store_true",
        help="Also print/save the OLD strategy-comparison table that mixes "
             "Monte-Carlo-sourced TWAP/Heuristic/AC constants with live "
             "LOB-replay RL numbers. NOT a valid comparison -- historical "
             "reference only, off by default.")
    p.add_argument("--ac-lambda", type=float, default=AC_DEFAULT_LAMBDA,
        help=f"Risk-aversion for the AC-family baselines in the unified "
             f"comparison (default {AC_DEFAULT_LAMBDA}; Rust research-sim's "
             f"default is 1e-4 -- the two were never reconciled, override "
             f"explicitly if you need to match one).")
    args = p.parse_args()

    run_full_analysis(
        args.passive_model,
        args.adversarial_model,
        args.episodes,
        n_state_dims=args.n_state_dims,
        eval_counterfactual=args.eval_counterfactual,
        btc_target=args.btc_target,
        include_test_dates=args.include_test_dates,
        unified_comparison=args.unified_comparison,
        legacy_mixed_comparison=args.legacy_mixed_comparison,
        ac_lambda=args.ac_lambda,
    )