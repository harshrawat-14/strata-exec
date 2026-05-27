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


def load_model(path: str) -> RecurrentPPO:
    """
    Load a RecurrentPPO model saved with _CPULSTMWrapper.
    Remaps lstm_actor.lstm.* -> lstm_actor.* in policy.pth.
    """
    zip_path = path if path.endswith('.zip') else path + '.zip'

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

    is_pct = None
    if fill_qtys:
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


def run_full_analysis(
    passive_model_path: str,
    adversarial_model_path: str = None,
    n_episodes: int = 50,
    output_dir: str = "rl/results/",
    n_state_dims: int = 8,
    eval_counterfactual: bool = False,
    btc_target: float = 50000.0,
    include_test_dates: bool = False,
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
    args = p.parse_args()

    run_full_analysis(
        args.passive_model,
        args.adversarial_model,
        args.episodes,
        n_state_dims=args.n_state_dims,
        eval_counterfactual=args.eval_counterfactual,
        btc_target=args.btc_target,
        include_test_dates=args.include_test_dates,
    )