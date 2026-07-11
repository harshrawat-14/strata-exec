"""
Phase 1 — Tail-Risk Metrics on Existing Strategies (no new models trained).

Runs the Phase 0 unified harness (evaluate_all_strategies_on_date) across
every date with clean historical data, reports mean_IS/CVaR-95 (with
bootstrap CI)/is_p1/is_std/is_sharpe per strategy per date, broken out by
regime, and the AC risk-aversion (lambda) grid on the two held-out crash
dates. No model is retrained here -- this measures the CURRENT model
(ppo_lstm_v5_adaptive_best) as-is.

IMPORTANT HONESTY NOTE (Global Rule 9 / Phase 0's date-split audit): this
model was trained under the OLD, contaminated split -- both 2024-08-05 and
2024-03-05 (the crash dates) were in its training data. The corrected split
(TRAINING_DATES / HELD_OUT_CRASH_DATES in evaluate.py) has been declared but
NOT yet trained on (that's Phase 2). Every crash-day number in this report
is therefore IN-SAMPLE for the RL strategy and must be read as such -- it is
not evidence for or against the CVaR thesis. This is stated in the report
output itself, not just in this docstring, so it can't be missed by a
reader who only sees the printed/saved output.

Usage: rl/.venv/bin/python rl/phase1_tail_risk_report.py
"""
from __future__ import annotations

import os

import pandas as pd

from evaluate import (
    load_model,
    evaluate_all_strategies_on_date,
    evaluate_ac_lambda_grid,
    HELD_OUT_CRASH_DATES,
    TEST_DATES,
)

MODEL_PATH = "rl/models/ppo_lstm_v5_adaptive_best/best_model"
N_EPISODES = 50
N_STATE_DIMS = 12
OUTPUT_DIR = "rl/results/"

# Regime label for every date with clean BookDepth+AggTrades data (Phase 0's
# survey, docs/system_design/... commit a7fe081). Coarse bucket is one of
# calm / crash / trending / volatile (volatile = meaningful price swings but
# no coincident book-depth-thinning signature, i.e. not a genuine
# liquidity-cascade stress event -- see the Phase 0 date-split rationale).
DATE_REGIMES = {
    "2024-01-15": ("calm bull", "calm"),
    "2024-02-20": ("moderate volatility (no liquidity-event signature)", "volatile"),
    "2024-03-05": ("crash - flash liquidation cascade (HELD OUT)", "crash"),
    "2024-04-15": ("BTC correction (moderate stress, in training set)", "crash"),
    "2024-06-10": ("quiet consolidation", "calm"),
    "2024-07-10": ("mid-summer sideways", "calm"),
    "2024-08-05": ("crash - Yen unwind (HELD OUT)", "crash"),
    "2024-09-15": ("moderate volatility (no liquidity-event signature)", "volatile"),
    "2024-10-15": ("moderate volatility (no liquidity-event signature)", "volatile"),
    "2024-11-06": ("post-election surge", "trending"),
    "2024-12-10": ("December bull run", "trending"),
}


def _available_dates() -> list:
    dates = []
    for d in sorted(DATE_REGIMES.keys()):
        path = f"TradeData/BookDepth/BTCUSDT-bookDepth-{d}.csv"
        if os.path.exists(path):
            dates.append(d)
        else:
            print(f"  SKIPPING {d} -- LOB file not found")
    return dates


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model(MODEL_PATH)

    print("=" * 90)
    print("PHASE 1 -- TAIL-RISK METRICS ON EXISTING STRATEGIES (no new model trained)")
    print("=" * 90)
    print(
        "\n*** HONESTY NOTE: this model was trained on the OLD split -- both crash "
        "dates (2024-08-05, 2024-03-05) were in its training data. The corrected "
        "held-out split is declared (HELD_OUT_CRASH_DATES) but not yet trained on "
        "(Phase 2). Every crash-day RL number below is IN-SAMPLE and must be read "
        "as such -- not evidence for or against the CVaR thesis. ***\n"
    )

    dates = _available_dates()
    rows = []

    for date in dates:
        regime_label, regime_bucket = DATE_REGIMES[date]
        crash_flag = " [HELD-OUT CRASH DATE, IN-SAMPLE FOR RL]" if date in HELD_OUT_CRASH_DATES else ""
        print(f"\n{'-' * 70}\n  {date}  ({regime_label}){crash_flag}\n{'-' * 70}")

        date_results = evaluate_all_strategies_on_date(
            rl_model=model,
            date=date,
            n_episodes=N_EPISODES,
            n_state_dims=N_STATE_DIMS,
        )

        for strategy_key, r in date_results.items():
            rows.append({
                "date": date,
                "regime_label": regime_label,
                "regime_bucket": regime_bucket,
                "is_held_out_crash_date": date in HELD_OUT_CRASH_DATES,
                # Old (pre-correction) split, per README's documented --multi-date
                # training invocation: everything except the 3 explicit TEST_DATES
                # was trained on, including both crash dates -- see honesty note.
                "is_training_date_under_current_model": date not in TEST_DATES,
                "strategy": r["strategy"],
                "mean_IS": r["mean_IS"],
                "std_IS": r["std_IS"],
                "cvar_95": r["cvar_95"],
                "cvar_95_ci_lower": r["cvar_95_ci_lower"],
                "cvar_95_ci_upper": r["cvar_95_ci_upper"],
                "cvar_95_ci_width": r["cvar_95_ci_width"],
                "cvar_95_n_tail": r["cvar_95_n_tail"],
                "is_p1": r["is_99th_pct"],
                "is_sharpe": r["is_sharpe"],
                "forced_liquidation_rate": r["forced_liquidation_rate"],
                "n_episodes": r["n_episodes"],
            })

    df = pd.DataFrame(rows)
    csv_path = f"{OUTPUT_DIR}phase1_tail_risk.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull table saved: {csv_path}")

    # ---- Regime breakdown summary ----
    print("\n" + "=" * 90)
    print("REGIME BREAKDOWN -- mean CVaR-95 by strategy x regime bucket")
    print("=" * 90)
    pivot = df.pivot_table(index="strategy", columns="regime_bucket", values="cvar_95", aggfunc="mean")
    print(pivot.round(4).to_string())

    # ---- Crash-day callout (explicit, per Phase 1 acceptance criteria) ----
    print("\n" + "=" * 90)
    print("CRASH-DAY CALLOUT: 2024-08-05 (Yen unwind) and 2024-03-05 (flash cascade)")
    print("*** BOTH ARE IN-SAMPLE FOR THE RL AGENT UNDER THE CURRENT MODEL ***")
    print("=" * 90)
    crash_df = df[df["date"].isin(HELD_OUT_CRASH_DATES)]
    print(
        crash_df[["date", "strategy", "mean_IS", "cvar_95", "cvar_95_ci_lower", "cvar_95_ci_upper"]]
        .sort_values(["date", "strategy"])
        .to_string(index=False)
    )

    # ---- AC lambda-grid sweep on both crash dates (Phase 1 task 3a) ----
    print("\n" + "=" * 90)
    print("AC RISK-AVERSION (LAMBDA) GRID -- crash dates only")
    print("=" * 90)
    lambda_rows = []
    for date in HELD_OUT_CRASH_DATES:
        if date not in dates:
            continue
        print(f"\n  {date}:")
        grid = evaluate_ac_lambda_grid(date=date, n_episodes=N_EPISODES)
        for lam, lam_result in grid.items():
            for strategy_key, r in lam_result.items():
                lambda_rows.append({
                    "date": date,
                    "lambda": lam,
                    "strategy": r["strategy"],
                    "mean_IS": r["mean_IS"],
                    "cvar_95": r["cvar_95"],
                })
    lambda_df = pd.DataFrame(lambda_rows)
    lambda_csv = f"{OUTPUT_DIR}phase1_ac_lambda_grid.csv"
    lambda_df.to_csv(lambda_csv, index=False)
    print(f"\nLambda-grid table saved: {lambda_csv}")
    print(lambda_df.to_string(index=False))

    # ---- CI width sanity flag (Global Rule 4) ----
    print("\n" + "=" * 90)
    print("CVaR-95 CI WIDTH CHECK (Global Rule 4: flag fragile tail estimates)")
    print("=" * 90)
    wide = df[df["cvar_95_ci_width"] > df["cvar_95"].abs() * 0.5]
    if len(wide) > 0:
        print(
            f"{len(wide)} of {len(df)} strategy x date rows have a CVaR-95 CI wider than "
            f"50% of the point estimate's magnitude -- these tail estimates are fragile "
            f"at n_episodes={N_EPISODES} and should not be over-read. Raise episode count "
            f"for any final headline table (Phase 4)."
        )
        print(wide[["date", "strategy", "cvar_95", "cvar_95_ci_lower", "cvar_95_ci_upper", "cvar_95_n_tail"]].to_string(index=False))
    else:
        print(f"No rows exceeded the 50%-of-magnitude CI-width flag at n_episodes={N_EPISODES}.")


if __name__ == "__main__":
    main()
