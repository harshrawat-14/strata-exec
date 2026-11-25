"""
Doc 33: refit the simulator's impact decay (pi = permanent share, rho = transient decay
rate per day) to the measured response curve of doc 31.

The simulator's response to one small sale is G(tau)/G(0) = sqrt(pi + (1 - pi) exp(-rho tau dt)),
dt = 1/2880 day. The real curve comes from doc 31's pooled propagator kernel on the V4
training days. The fit is least squares on a log-spaced lag grid (each part of the curve
weighted alike, so the fast early reversion is not swamped by the long plateau). Held-out
dates are not read.

    python rl/experiments/impact_decay_calibrate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from evaluation.dates import TRAINING_DATES_V4, VALIDATION_DATES_V4
from impact_decay_fit import BINS, MAX_LAG, date_data, design

DT = 1 / 2880
OLD = (0.3, 5.0)
GRID = np.unique(np.round(np.geomspace(1, MAX_LAG, 40)).astype(int))
CHECK = {"30min": 60, "2h": 240, "6h": 720}
N_BOOT = 300
OUT = "rl/results/impact_decay_calibrate.csv"


def sim_curve(pi, rho, taus):
    return np.sqrt(pi + (1 - pi) * np.exp(-rho * taus * DT))


def real_curve(beta, taus):
    """G(tau)/G(0) from per-bin coefficients (constant within a bin)."""
    g = np.array([sum(beta[b] * (min(hi, t) - lo + 1) for b, (lo, hi) in enumerate(BINS) if lo <= t)
                  for t in taus])
    return g / beta[0]


def fit(curve):
    res = least_squares(lambda p: sim_curve(p[0], np.exp(p[1]), GRID) - curve,
                        x0=[0.45, np.log(500.0)], bounds=([0.0, np.log(1.0)], [1.0, np.log(1e5)]))
    return res.x[0], float(np.exp(res.x[1]))


def half_life_min(rho):
    return np.log(2) / rho * 24 * 60


def main():
    from multiprocessing import Pool
    dates = TRAINING_DATES_V4 + VALIDATION_DATES_V4
    with Pool(8) as pool:
        data = {d["date"]: d for d in pool.map(date_data, dates)}
    gram = {}
    for d in dates:
        X, y = design(data[d], np.array([np.inf, np.inf]), "pooled")  # cuts unused when pooled
        gram[d] = (X.T @ X, X.T @ y)

    def beta(ds):
        return np.linalg.solve(sum(gram[d][0] for d in ds), sum(gram[d][1] for d in ds))

    train_curve = real_curve(beta(TRAINING_DATES_V4), GRID)
    val_curve = real_curve(beta(VALIDATION_DATES_V4), GRID)
    pi, rho = fit(train_curve)

    rng = np.random.default_rng(0)
    boot_curves, boot_params = [], []
    for _ in range(N_BOOT):
        s = list(rng.choice(TRAINING_DATES_V4, len(TRAINING_DATES_V4)))
        c = real_curve(beta(s), GRID)
        boot_curves.append(c)
        boot_params.append(fit(c))
    boot_curves, boot_params = np.array(boot_curves), np.array(boot_params)
    lo_c, hi_c = np.quantile(boot_curves, [0.05, 0.95], axis=0)
    (pi_lo, rho_lo), (pi_hi, rho_hi) = np.quantile(boot_params, [0.05, 0.95], axis=0)

    print(f"fitted on {len(TRAINING_DATES_V4)} training days:")
    print(f"  pi  = {pi:.3f}  [{pi_lo:.3f}, {pi_hi:.3f}]   (old {OLD[0]})")
    print(f"  rho = {rho:.0f}/day  [{rho_lo:.0f}, {rho_hi:.0f}]  half-life {half_life_min(rho):.2f} min "
          f"[{half_life_min(rho_hi):.2f}, {half_life_min(rho_lo):.2f}]   (old {OLD[1]}/day, "
          f"{half_life_min(OLD[1]):.0f} min)")

    new_fit, old_fit = sim_curve(pi, rho, GRID), sim_curve(*OLD, GRID)
    print("\nA. fitted curve inside the training 90% band at the check lags:")
    ok_a = True
    for name, tau in CHECK.items():
        i = int(np.argmin(np.abs(GRID - tau)))
        inside = lo_c[i] <= new_fit[i] <= hi_c[i]
        ok_a &= inside
        print(f"  {name:>6} (lag {GRID[i]}): real {train_curve[i]:.3f} [{lo_c[i]:.3f}, {hi_c[i]:.3f}]  "
              f"new {new_fit[i]:.3f}  old {old_fit[i]:.3f}  -> {'inside' if inside else 'OUTSIDE'}")
    inside_all = np.mean((new_fit >= lo_c) & (new_fit <= hi_c))
    print(f"  share of all {len(GRID)} grid lags inside the band: {inside_all:.0%}")

    rmse_new = np.sqrt(np.mean((new_fit - val_curve) ** 2))
    rmse_old = np.sqrt(np.mean((old_fit - val_curve) ** 2))
    ok_b = rmse_new <= 0.5 * rmse_old
    print(f"\nB. validation curve RMSE: new {rmse_new:.4f}  old {rmse_old:.4f}  "
          f"({rmse_new / rmse_old:.0%} of old; need <= 50%) -> {ok_b}")
    print(f"\nACCEPT: {ok_a and ok_b}")

    pd.DataFrame(dict(lag=GRID, real_train=train_curve, band_lo=lo_c, band_hi=hi_c, real_val=val_curve,
                      sim_new=new_fit, sim_old=old_fit)).to_csv(OUT, index=False)
    pd.DataFrame([dict(pi=pi, pi_lo=pi_lo, pi_hi=pi_hi, rho=rho, rho_lo=rho_lo, rho_hi=rho_hi,
                       rmse_new=rmse_new, rmse_old=rmse_old)]).to_csv(OUT.replace(".csv", "_params.csv"), index=False)


if __name__ == "__main__":
    main()
