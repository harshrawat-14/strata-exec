"""
Phase 3 Step 1: tail-validation analysis on top of the per-episode dump
produced by rl/phase3_tail_validation_eval.py.

Tests the four-part evidence bar from docs/phase3_plan.md, "The tail claim
-- evidence bar and current status":
  #2 severity up -> RL CVaR advantage up
  #3 mean-controlled tail advantage (the crux) -- residualize each date's
     RL/baseline episode arrays by their own means, then compare the
     residual (mean-removed) distributions' CVaR/dispersion/lower-quantile
     shape. If RL's residual tail is not distinguishable from the
     baseline's, the CVaR edge is the mean edge restated, not a separate
     tail mechanism.
  #4 robustness -- bootstrap CIs on the raw CVaR deltas at 90/95/99%,
     paired by matched episode seed (same method as
     rl/phase2r_ablation_honest_lambda.py's paired_bootstrap_delta and
     evaluate.py's bootstrap_cvar_95_ci, generalized to CVaR-of-paired-
     difference and, for the crux test, to CVaR-of-mean-centered-residual).
     No new statistical method: this is those two existing methods
     composed, not invented.

Reads rl/results/phase3_tail_validation_episodes.csv (must exist --
run rl/phase3_tail_validation_eval.py first).

Usage: rl/.venv/bin/python rl/phase3_tail_validation_analysis.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from evaluate import compute_cvar_95

EPISODES_CSV = "rl/results/phase3_tail_validation_episodes.csv"
OUTPUT_DIR = "rl/results/"
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0

BASELINE_KEYS = ["ac_optimal", "adaptive_ac", "twap"]
BASELINE_LABELS = {"ac_optimal": "AC-Optimal", "adaptive_ac": "Adaptive-AC", "twap": "TWAP"}
QUANTILES = [0.90, 0.95, 0.99]

FLAGSHIP_CRASH_DATES = ["2024-03-05", "2024-08-05"]


# ── CVaR at an arbitrary quantile, reusing evaluate.compute_cvar_95 exactly
# for q=0.95 (delegation, not reimplementation) and generalizing its same
# tail-mean method for 90/99. ──────────────────────────────────────────────
def cvar_at_quantile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=float)
    if abs(q - 0.95) < 1e-9:
        return compute_cvar_95(arr)
    if len(arr) == 0:
        return float("nan")
    n_tail = max(1, int(len(arr) * (1 - q)))
    sorted_asc = np.sort(arr)
    return float(np.mean(sorted_asc[:n_tail]))


def cvar_at_quantile_vectorized(matrix: np.ndarray, q: float) -> np.ndarray:
    """matrix shape (n_bootstrap, n) -> array of length n_bootstrap."""
    n = matrix.shape[1]
    n_tail = max(1, int(n * (1 - q)))
    sorted_mat = np.sort(matrix, axis=1)
    return sorted_mat[:, :n_tail].mean(axis=1)


def paired_bootstrap_cvar_delta(
    a_samples, b_samples, q: float = 0.95,
    n_bootstrap: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
    centered: bool = False,
) -> dict:
    """
    Paired bootstrap on a CVaR-q delta between two matched-seed episode
    arrays (a - b, more negative b delta means a is worse). Same
    index-resampling mechanics as
    phase2r_ablation_honest_lambda.paired_bootstrap_delta (resample the
    SAME episode indices for both arms, preserving the seed-pairing) and
    evaluate.bootstrap_cvar_95_ci (bootstrap a CVaR estimate's own sampling
    distribution) -- composed here rather than reimplemented from scratch.

    centered=True: subtract each arm's OWN mean (computed on the resample,
    not the full sample -- a proper bootstrap of the "CVaR of the residual
    distribution" statistic) before computing CVaR-q. This is the crux-test
    statistic for requirement #3: does RL's tail still look better once
    location is removed?
    """
    a = np.asarray(a_samples, dtype=float)
    b = np.asarray(b_samples, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    def _prep(x):
        return x - x.mean() if centered else x

    point = cvar_at_quantile(_prep(a), q) - cvar_at_quantile(_prep(b), q)

    if n == 0:
        return {"point": point, "ci_lower": point, "ci_upper": point, "p_value": None, "n": n}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    a_resamp = a[idx]
    b_resamp = b[idx]
    if centered:
        a_resamp = a_resamp - a_resamp.mean(axis=1, keepdims=True)
        b_resamp = b_resamp - b_resamp.mean(axis=1, keepdims=True)

    a_cvars = cvar_at_quantile_vectorized(a_resamp, q)
    b_cvars = cvar_at_quantile_vectorized(b_resamp, q)
    deltas = a_cvars - b_cvars

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_value = float((deltas <= 0).mean())
    return {"point": point, "ci_lower": ci_lower, "ci_upper": ci_upper, "p_value": p_value, "n": n}


def paired_bootstrap_std_delta(a_samples, b_samples, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED) -> dict:
    """
    Same paired-index-resample mechanics, applied to a plain dispersion
    statistic (std of the ALREADY-centered residual, i.e. of a - mean(a)),
    as a second, simpler crux-test statistic alongside centered-CVaR:
    a smaller std for RL than baseline (after both are centered on their
    own mean) means RL's outcomes are less spread out around its own
    average, independent of where that average sits.
    """
    a = np.asarray(a_samples, dtype=float)
    b = np.asarray(b_samples, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    point = float(a.std() - b.std())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    a_resamp = a[idx]
    b_resamp = b[idx]
    deltas = a_resamp.std(axis=1) - b_resamp.std(axis=1)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_value = float((deltas >= 0).mean())  # RL std smaller = advantage -> test point < 0
    return {"point": point, "ci_lower": ci_lower, "ci_upper": ci_upper, "p_value": p_value, "n": n}


def load_episodes() -> pd.DataFrame:
    if not os.path.exists(EPISODES_CSV):
        raise FileNotFoundError(
            f"{EPISODES_CSV} not found -- run rl/phase3_tail_validation_eval.py first"
        )
    return pd.read_csv(EPISODES_CSV)


def get_arr(df: pd.DataFrame, date: str, strategy: str) -> np.ndarray:
    sub = df[(df["date"] == date) & (df["strategy"] == strategy)].sort_values("episode_seed")
    return sub["is_pct"].to_numpy()


def main():
    df = load_episodes()
    dates = sorted(df["date"].unique(), key=lambda d: df[df["date"] == d]["is_pct"].mean())
    # order dates by RL16 mean IS ascending isn't meaningful for severity;
    # use each date's regime_v2/severity metadata for display instead.
    meta = df[["date", "regime_v2", "window_v2", "split_v2", "split_v1"]].drop_duplicates().set_index("date")

    # Compute AC-honest-lambda mean_IS per date directly from this run's
    # own episodes (not the frozen SEVERITY_RANK_V2 constant) -- internal
    # consistency, same severity proxy the project already uses.
    ac_mean_by_date = {
        d: get_arr(df, d, "ac_optimal").mean() for d in df["date"].unique()
    }
    dates_by_severity = sorted(ac_mean_by_date.keys(), key=lambda d: ac_mean_by_date[d])

    print("=" * 104)
    print("REQUIREMENT #4: raw CVaR-90/95/99 deltas (RL16 vs AC/Adaptive-AC/TWAP), paired bootstrap CI")
    print("=" * 104)

    raw_rows = []
    for date in dates_by_severity:
        rl_arr = get_arr(df, date, "rl_agent")
        m = meta.loc[date]
        for base_key in BASELINE_KEYS:
            base_arr = get_arr(df, date, base_key)
            for q in QUANTILES:
                res = paired_bootstrap_cvar_delta(rl_arr, base_arr, q=q, centered=False)
                raw_rows.append({
                    "date": date, "regime_v2": m["regime_v2"], "split_v2": m["split_v2"],
                    "split_v1": m["split_v1"], "ac_mean_is": ac_mean_by_date[date],
                    "baseline": base_key, "q": q,
                    "rl_cvar": cvar_at_quantile(rl_arr, q),
                    "base_cvar": cvar_at_quantile(base_arr, q),
                    "delta": res["point"], "ci_lower": res["ci_lower"], "ci_upper": res["ci_upper"],
                    "p_value": res["p_value"], "rl_beats_baseline_cvar": res["point"] > 0,
                })
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(f"{OUTPUT_DIR}phase3_tail_validation_cvar_bootstrap.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}phase3_tail_validation_cvar_bootstrap.csv ({len(raw_df)} rows)\n")

    # Flagship dates at q=0.95, printed for direct comparison against the
    # phase3_plan.md table this task is validating.
    for date in FLAGSHIP_CRASH_DATES:
        sub = raw_df[(raw_df["date"] == date) & (raw_df["q"] == 0.95)]
        print(f"  {date}:")
        for _, r in sub.iterrows():
            sig = "SIGNIFICANT" if (r["p_value"] is not None and r["p_value"] < 0.05) else "not significant"
            print(f"    RL16 CVaR95={r['rl_cvar']:+.4f}%  {BASELINE_LABELS[r['baseline']]} CVaR95={r['base_cvar']:+.4f}%  "
                  f"delta={r['delta']:+.4f}pp [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]  p={r['p_value']:.4f}  {sig}")

    print("\n" + "=" * 104)
    print("REQUIREMENT #2: severity -> RL CVaR-95 advantage (Spearman correlation + top-N table)")
    print("=" * 104)

    q95 = raw_df[raw_df["q"] == 0.95].copy()
    # One row per date per baseline; use AC-Optimal as the primary baseline
    # (same choice the mean-severity result in phase2r_severity_expansion.md
    # used implicitly via ac_mean_is as the severity metric).
    ac_rows = q95[q95["baseline"] == "ac_optimal"].copy()
    ac_rows["severity_rank"] = ac_rows["ac_mean_is"].rank(method="first")  # 1 = most severe (most negative)
    ac_rows = ac_rows.sort_values("ac_mean_is")

    rho, p_rho = scipy_stats.spearmanr(ac_rows["ac_mean_is"], ac_rows["delta"])
    print(f"\nSpearman rho(ac_mean_is, CVaR95_delta_RL_vs_AC) across {len(ac_rows)} dates: "
          f"rho={rho:.3f}, p={p_rho:.4f}")
    print("(ac_mean_is more negative = more severe; if severity -> larger CVaR edge, expect NEGATIVE rho:")
    print(" more severe (more negative severity) pairs with LARGER delta.)")

    print(f"\n{'date':>12s} {'regime':>8s} {'ac_mean_is':>11s} {'rl16_cvar95':>12s} {'ac_cvar95':>10s} "
          f"{'delta':>8s} {'p':>7s} {'win?':>5s}")
    for _, r in ac_rows.iterrows():
        win = "WIN" if r["delta"] > 0 else "lose"
        print(f"{r['date']:>12s} {r['regime_v2']:>8s} {r['ac_mean_is']:>+11.4f} {r['rl_cvar']:>+12.4f} "
              f"{r['base_cvar']:>+10.4f} {r['delta']:>+8.4f} {r['p_value']:>7.4f} {win:>5s}")

    n = len(ac_rows)
    for top_n in [4, 7, 12]:
        top = ac_rows.head(top_n)
        wins = (top["delta"] > 0).sum()
        print(f"\nTop {top_n} most severe dates: RL16 CVaR95 beats AC-honest on {wins}/{top_n}")
    bottom = ac_rows.tail(n - 7) if n > 7 else ac_rows.iloc[0:0]
    if len(bottom):
        wins_bottom = (bottom["delta"] > 0).sum()
        print(f"Remaining {len(bottom)} (mild/calm) dates: RL16 CVaR95 beats AC-honest on {wins_bottom}/{len(bottom)}")

    print("\n" + "=" * 104)
    print("REQUIREMENT #3 (THE CRUX): mean-controlled tail advantage")
    print("Residualize each date's RL/baseline episode arrays by their OWN mean, then compare the")
    print("residual distributions' CVaR-95 (tail shape) and std (dispersion) -- does RL still look")
    print("better on the tail once the location/mean shift is removed?")
    print("=" * 104)

    crux_rows = []

    def run_crux(label, rl_arr, base_arr, base_key, date_tag):
        cvar_res = paired_bootstrap_cvar_delta(rl_arr, base_arr, q=0.95, centered=True)
        std_res = paired_bootstrap_std_delta(rl_arr, base_arr)
        rl_c = rl_arr - rl_arr.mean()
        base_c = base_arr - base_arr.mean()
        p5_rl = float(np.percentile(rl_c, 5))
        p5_base = float(np.percentile(base_c, 5))
        row = {
            "scope": label, "date": date_tag, "baseline": base_key, "n": len(rl_arr),
            "rl_std": float(rl_arr.std()), "base_std": float(base_arr.std()),
            "std_delta": std_res["point"], "std_ci_lower": std_res["ci_lower"],
            "std_ci_upper": std_res["ci_upper"], "std_p_value": std_res["p_value"],
            "centered_cvar95_rl": cvar_at_quantile(rl_c, 0.95),
            "centered_cvar95_base": cvar_at_quantile(base_c, 0.95),
            "centered_cvar95_delta": cvar_res["point"],
            "centered_cvar95_ci_lower": cvar_res["ci_lower"],
            "centered_cvar95_ci_upper": cvar_res["ci_upper"],
            "centered_cvar95_p_value": cvar_res["p_value"],
            "p5_residual_rl": p5_rl, "p5_residual_base": p5_base,
        }
        crux_rows.append(row)
        return row

    print("\n--- Per-date (flagship crash dates) ---")
    for date in FLAGSHIP_CRASH_DATES:
        rl_arr = get_arr(df, date, "rl_agent")
        for base_key in BASELINE_KEYS:
            base_arr = get_arr(df, date, base_key)
            row = run_crux("per_date", rl_arr, base_arr, base_key, date)
            sig = "SIGNIFICANT" if row["centered_cvar95_p_value"] is not None and row["centered_cvar95_p_value"] < 0.05 else "not significant"
            print(f"  {date} vs {BASELINE_LABELS[base_key]}: "
                  f"raw_std(RL)={row['rl_std']:.4f} raw_std(base)={row['base_std']:.4f}  "
                  f"std_delta={row['std_delta']:+.4f} [{row['std_ci_lower']:+.4f},{row['std_ci_upper']:+.4f}] p={row['std_p_value']:.4f}")
            print(f"    centered CVaR95: RL={row['centered_cvar95_rl']:+.4f}  base={row['centered_cvar95_base']:+.4f}  "
                  f"delta={row['centered_cvar95_delta']:+.4f} [{row['centered_cvar95_ci_lower']:+.4f},"
                  f"{row['centered_cvar95_ci_upper']:+.4f}] p={row['centered_cvar95_p_value']:.4f}  {sig}")
            print(f"    residual p5: RL={row['p5_residual_rl']:+.4f}  base={row['p5_residual_base']:+.4f}")

    print("\n--- Pooled across all V2 'crash'-regime dates (both train and held-out; larger n) ---")
    crash_dates = meta[meta["regime_v2"] == "crash"].index.tolist()
    print(f"  crash-regime dates (n={len(crash_dates)}): {crash_dates}")
    for base_key in BASELINE_KEYS:
        rl_pool = []
        base_pool = []
        for date in crash_dates:
            rl_arr = get_arr(df, date, "rl_agent")
            base_arr = get_arr(df, date, base_key)
            # Center EACH date's arrays by that date's own mean before
            # pooling -- otherwise pooling raw values across dates with very
            # different price-path means would reintroduce the location
            # effect this test exists to remove.
            rl_pool.append(rl_arr - rl_arr.mean())
            base_pool.append(base_arr - base_arr.mean())
        rl_pool = np.concatenate(rl_pool)
        base_pool = np.concatenate(base_pool)
        # Already centered per-date; pass centered=False here to avoid
        # double-centering the pooled array as a whole.
        cvar_res = paired_bootstrap_cvar_delta(rl_pool, base_pool, q=0.95, centered=False)
        std_res = paired_bootstrap_std_delta(rl_pool, base_pool)
        row = {
            "scope": "pooled_crash_regime", "date": "ALL_CRASH_V2", "baseline": base_key, "n": len(rl_pool),
            "rl_std": float(rl_pool.std()), "base_std": float(base_pool.std()),
            "std_delta": std_res["point"], "std_ci_lower": std_res["ci_lower"],
            "std_ci_upper": std_res["ci_upper"], "std_p_value": std_res["p_value"],
            "centered_cvar95_rl": cvar_at_quantile(rl_pool, 0.95),
            "centered_cvar95_base": cvar_at_quantile(base_pool, 0.95),
            "centered_cvar95_delta": cvar_res["point"],
            "centered_cvar95_ci_lower": cvar_res["ci_lower"],
            "centered_cvar95_ci_upper": cvar_res["ci_upper"],
            "centered_cvar95_p_value": cvar_res["p_value"],
            "p5_residual_rl": float(np.percentile(rl_pool, 5)),
            "p5_residual_base": float(np.percentile(base_pool, 5)),
        }
        crux_rows.append(row)
        sig = "SIGNIFICANT" if row["centered_cvar95_p_value"] is not None and row["centered_cvar95_p_value"] < 0.05 else "not significant"
        print(f"  pooled crash-regime (n={row['n']} episodes) vs {BASELINE_LABELS[base_key]}: "
              f"std_delta={row['std_delta']:+.4f} [{row['std_ci_lower']:+.4f},{row['std_ci_upper']:+.4f}] p={row['std_p_value']:.4f}")
        print(f"    centered CVaR95: RL={row['centered_cvar95_rl']:+.4f}  base={row['centered_cvar95_base']:+.4f}  "
              f"delta={row['centered_cvar95_delta']:+.4f} [{row['centered_cvar95_ci_lower']:+.4f},"
              f"{row['centered_cvar95_ci_upper']:+.4f}] p={row['centered_cvar95_p_value']:.4f}  {sig}")

    crux_df = pd.DataFrame(crux_rows)
    crux_df.to_csv(f"{OUTPUT_DIR}phase3_tail_validation_crux.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR}phase3_tail_validation_crux.csv ({len(crux_df)} rows)")


if __name__ == "__main__":
    main()
