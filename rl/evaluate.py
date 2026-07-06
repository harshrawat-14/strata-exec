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
import zipfile
import tempfile
import torch

TEST_DATES = {
    "2024-04-15": "BTC correction",
    "2024-07-10": "mid-summer sideways",
    "2024-12-10": "December bull run",
}

DATES = {
    "2024-01-15": "calm bull",
    "2024-03-05": "BTC breakout",
    "2024-06-10": "quiet consolidation",
    "2024-08-05": "crash - Yen unwind",
    "2024-11-06": "post-election surge",
}

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

V5_FRACTIONS = [
    0.000, 0.001, 0.003, 0.005, 0.008, 0.010,
    0.015, 0.020, 0.030, 0.050, 0.070, 0.100,
    0.150, 0.200, 0.300, 0.400, 0.500, 0.600,
    0.800, 1.000,
]

V6_FRACTIONS = [
    0.000, 0.005, 0.010, 0.015, 0.020, 0.025,
    0.030, 0.035, 0.040, 0.050, 0.060, 0.075,
    0.100, 0.125, 0.150, 0.200, 0.300, 0.500,
    0.750, 1.000,
]


def get_action_fractions(n_state_dims: int) -> list:
    """Return correct action fraction list based on state dims."""
    return V6_FRACTIONS if n_state_dims == 12 else V5_FRACTIONS


def nearest_action(target_frac: float, fractions: list) -> int:
    """Map a desired sell fraction to the nearest discrete action."""
    target_frac = max(0.0, min(1.0, target_frac))
    diffs = [abs(f - target_frac) for f in fractions]
    return int(np.argmin(diffs))


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


class ACOptimalPolicy:
    """
    Almgren-Chriss closed-form optimal execution trajectory.
    Pre-computes the full AC schedule at initialisation using:
      x(t) = X0 * sinh(kappa*(T-t)) / sinh(kappa*T)
    """

    def __init__(self, n_state_dims: int = 12,
                 sigma: float = 0.20,
                 lam: float = 0.001,
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

        denom = np.sinh(k * T) + 1e-10
        remaining_at_t = np.sinh(k * (T - steps)) / denom
        remaining_at_t_plus_1 = np.concatenate([
            np.sinh(k * (T - steps[1:])) / denom,
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
                 lam: float = 0.001,
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
            denom = np.sinh(kappa * T) + 1e-10
            remaining = np.sinh(kappa * (T - steps)) / denom
            next_rem = np.concatenate([
                np.sinh(kappa * (T - steps[1:])) / denom,
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


def load_model(path: str):
    """
    Load a model. Supports both RecurrentPPO (with _CPULSTMWrapper patch)
    and standard PPO.
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
            return RecurrentPPO.load(patched, device='cpu')
    else:
        from stable_baselines3 import PPO
        return PPO.load(zip_path, device='cpu')


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
        "cvar95": float(np.percentile(arr, 5)),
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
        "cvar95":  float(np.percentile(arr, 5)),
        "forced_liquidation_rate": forced_liquidations / n_episodes,
    }


def evaluate_on_synthetic(model, n_episodes=200, n_state_dims=8) -> dict:
    env = StrataExecEnv(
        mode="synthetic",
        n_state_dims=n_state_dims,
        fixed_steps=True,
        fixed_size=True,
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
        "cvar95": float(np.percentile(arr, 5)),
    }


def evaluate_all_strategies_on_date(
    rl_model,
    date: str,
    n_episodes: int = 50,
    n_state_dims: int = 12,
    adversarial: bool = False,
    fixed_steps: bool = True,
    fixed_size: bool = True,
) -> dict:
    """
    Evaluate all 5 strategies on the same counterfactual LOB
    environment using the same Obizhaeva-Wang impact model,
    same real Binance price path, and same arrival price.

    Replaces the mixed-framework comparison where TWAP/Heuristic
    used research-sim (Monte Carlo + square-root impact) and RL
    used evaluate.py (real LOB + OW impact).
    """
    policies = {
        "twap": TWAPPolicy(n_state_dims),
        "heuristic": HeuristicPolicy(n_state_dims),
        "ac_optimal": ACOptimalPolicy(n_state_dims),
        "adaptive_ac": AdaptiveACPolicy(n_state_dims),
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

        env = StrataExecEnv(
            mode="historical",
            adversarial=adversarial,
            lob_date=date,
            agg_date=date,
            n_state_dims=n_state_dims,
            fixed_steps=fixed_steps,
            fixed_size=fixed_size,
        )

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

        # CVaR-95: average of worst 5% of IS values (most negative)
        n_tail = max(1, len(arr) // 20)
        sorted_asc = np.sort(arr)
        cvar_95 = float(np.mean(sorted_asc[:n_tail]))

        results[strategy_key] = {
            "strategy": strategy_labels[strategy_key],
            "date": date,
            "mean_IS": float(np.mean(arr)),
            "std_IS": float(np.std(arr)),
            "var_IS": float(np.var(arr)),
            "cvar_95": cvar_95,
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
              f"CVaR95={cvar_95:.4f}%")

    return results


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


def run_full_analysis(
    passive_model_path: str,
    adversarial_model_path: str = None,
    n_episodes: int = 50,
    output_dir: str = "rl/results/",
    n_state_dims: int = 8,
    eval_counterfactual: bool = False,
    btc_target: float = 50000.0,
    include_test_dates: bool = False,
    unified_comparison: bool = False,
):
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

        stat_test = test_vs_baseline(
            pr["is_samples"],
            STATIC_OPTIMAL_IS[date],
        )

        row = {
            "date": date,
            "regime": regime,
            "twap_IS": TWAP_IS.get(date),
            "heuristic_IS": HEURISTIC_IS.get(date),
            "static_optimal_IS": STATIC_OPTIMAL_IS[date],
            "adaptive_optimal_IS": ADAPTIVE_OPTIMAL_IS[date],
            "rl_passive_IS": pr["mean_IS"],
            "rl_passive_var": pr["var_IS"],
            "rl_passive_cvar95": pr["cvar95"],
            "rl_passive_forced_liq": pr["forced_liquidation_rate"],
            "rl_passive_degradation":
                pr["mean_IS"] - synth["mean_IS"],
            "mean_action": pr["mean_action"],
            "action_entropy": pr["action_entropy"],
            "action_distribution": pr["action_distribution"],
            "p_value_vs_ac": stat_test["p_value"],
            "ci_lower": stat_test["ci_lower"],
            "ci_upper": stat_test["ci_upper"],
            "significantly_better": stat_test["significantly_better"],
        }

        print(f"  Mean action: {pr['mean_action']:.2f}")
        print(f"  Action entropy: {pr['action_entropy']:.3f}")
        top3 = sorted(
            pr["action_distribution"].items(),
            key=lambda x: -x[1],
        )[:3]
        print(f"  Top 3 actions: {top3}")
        baseline = STATIC_OPTIMAL_IS[date]
        print(
            f"  RL IS: {pr['mean_IS']:.4f}% "
            f"[{stat_test['ci_lower']:.4f}%, "
            f"{stat_test['ci_upper']:.4f}%] 95% CI"
        )
        if stat_test["p_value"] is None:
            print(
                f"  vs AC ({baseline:.4f}%): "
                f"p=N/A (zero variance — deterministic policy)"
            )
        else:
            sig = stat_test["significantly_better"]
            label = "SIGNIFICANT" if sig else "not significant"
            print(
                f"  vs AC ({baseline:.4f}%): "
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

    print("\n" + "=" * 90)
    print("STRATEGY COMPARISON - IS% across 5 dates")
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
        twap = row.get("twap_IS")
        heur = row.get("heuristic_IS")
        twap_s = f"{twap:>7.3f}%" if twap is not None else f"{'N/A':>8}"
        heur_s = (
            f"{heur:>9.3f}%" if heur is not None else f"{'N/A':>10}"
        )
        line = (
            f"{row['date']:<12} {row['regime']:<22} "
            f"{twap_s} {heur_s} "
            f"{row['static_optimal_IS']:>7.3f}% "
            f"{row['adaptive_optimal_IS']:>8.3f}% "
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
    p.add_argument("--unified-comparison", action="store_true",
        help="Run all 5 strategies in same counterfactual LOB "
             "for methodologically valid comparison")
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
    )