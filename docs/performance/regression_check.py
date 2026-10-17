"""
Phase S regression-verify script (hard gate per the Phase S plan).

Runs the actual Phase 0 unified comparison harness (evaluate_all_strategies_on_date)
against all 5 strategies on one date, so it can be diffed before/after the Phase S
rewrite to confirm the transport/caching/allocation changes did not change simulation
semantics -- only performance. Run from repo root:

    rl/.venv/bin/python docs/performance/regression_check.py

To reproduce the actual before/after comparison documented in
docs/performance/regression-verification.md: build target/release/rl-env at the last pre-Phase-S
commit (a7fe081) into a separate path, run this script against each binary in turn
(copying it to target/release/rl-env before each run, since environment.py hardcodes
that relative path), and diff the two outputs.
"""
import sys

sys.path.insert(0, "rl")

from evaluate import load_model, evaluate_all_strategies_on_date  # noqa: E402

MODEL_PATH = "rl/models/ppo_lstm_v5_adaptive_best/best_model"
DATE = "2024-01-15"
N_EPISODES = 10


def main():
    model = load_model(MODEL_PATH)
    results = evaluate_all_strategies_on_date(
        rl_model=model,
        date=DATE,
        n_episodes=N_EPISODES,
        n_state_dims=12,
    )
    for key, r in results.items():
        print(
            f"{key}: mean_IS={r['mean_IS']:.6f} cvar_95={r['cvar_95']:.6f} "
            f"std_IS={r['std_IS']:.6f} is_sharpe={r['is_sharpe']:.6f}"
        )


if __name__ == "__main__":
    main()
