"""
THE HONEST FAILURE ANALYSIS.
This is the core research deliverable of Phase 4.
"""

from sb3_contrib import RecurrentPPO
from environment import StrataExecEnv
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
    "2024-01-15": -0.990,
    "2024-03-05": -1.228,
    "2024-06-10": -0.868,
    "2024-08-05": -1.071,
    "2024-11-06": -1.212,
}
ADAPTIVE_OPTIMAL_IS = {
    "2024-01-15": -0.494,
    "2024-03-05": -0.586,
    "2024-06-10": -0.449,
    "2024-08-05": -0.585,
    "2024-11-06": -0.582,
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
    Returns (is_pct, episode_reward, forced_liquidation).
    """
    obs, info = env.reset(seed=seed)
    arrival_price = info.get("arrival_price", 100.0)

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    done = False
    ep_reward = 0.0
    fill_qtys = []
    fill_prices = []

    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start = np.zeros((1,), dtype=bool)

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

    return is_pct, ep_reward, forced


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
    )

    all_is = []
    all_rewards = []
    forced_liquidations = 0

    for ep in range(n_episodes):
        is_pct, ep_reward, forced = _run_episode(model, env, seed=ep)
        if forced:
            forced_liquidations += 1
        if is_pct is not None:
            all_is.append(is_pct)
        all_rewards.append(ep_reward)

    env.close()

    arr = np.array(all_is)
    return {
        "date": date,
        "mean_IS": float(np.mean(arr)),
        "std_IS": float(np.std(arr)),
        "var_IS": float(np.var(arr)),
        "cvar95": float(np.percentile(arr, 5)),
        "mean_reward": float(np.mean(all_rewards)),
        "forced_liquidation_rate": forced_liquidations / n_episodes,
        "n_episodes": n_episodes,
    }


def evaluate_model_counterfactual(
    model,
    date: str,
    n_episodes: int = 50,
    btc_target: float = 50000.0,
    n_state_dims: int = 10,
) -> dict:
    env = StrataExecEnv(
        mode="counterfactual",
        lob_date=date,
        agg_date=date,
        btc_target=btc_target,
        n_state_dims=n_state_dims,
    )

    all_is = []
    all_rewards = []
    forced_liquidations = 0

    for ep in range(n_episodes):
        is_pct, ep_reward, forced = _run_episode(model, env, seed=ep)
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
    )
    all_is = []
    all_rewards = []

    for ep in range(n_episodes):
        is_pct, ep_reward, _ = _run_episode(model, env, seed=ep + 10000)
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

        row = {
            "date": date,
            "regime": regime,
            "static_optimal_IS": STATIC_OPTIMAL_IS[date],
            "adaptive_optimal_IS": ADAPTIVE_OPTIMAL_IS[date],
            "rl_passive_IS": pr["mean_IS"],
            "rl_passive_var": pr["var_IS"],
            "rl_passive_cvar95": pr["cvar95"],
            "rl_passive_forced_liq": pr["forced_liquidation_rate"],
            "rl_passive_degradation":
                pr["mean_IS"] - synth["mean_IS"],
        }

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

    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON - IS% across 5 dates")
    print("=" * 70)

    header = f"{'Date':<12} {'Regime':<22} {'StatOpt':>8} "
    header += f"{'AdaptOpt':>9} {'RL-Pass':>8}"
    if adv_model:
        header += f" {'RL-Adv':>8}"
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        line = (f"{row['date']:<12} {row['regime']:<22} "
                f"{row['static_optimal_IS']:>7.3f}% "
                f"{row['adaptive_optimal_IS']:>8.3f}% "
                f"{row['rl_passive_IS']:>7.3f}%")
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