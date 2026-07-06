"""
scripts/precompute_demo_data.py

Run this locally before deploying to Render:
  python scripts/precompute_demo_data.py

Generates demo_results/ with pre-computed simulation and
evaluation results for all defined presets and dates.
"""

import sys
from pathlib import Path

# Add project root to sys.path so we can import backend services
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import asyncio
import json
import numpy as np
from datetime import datetime
from web.services.rust_runner import RustRunner

DEMO_DIR = ROOT / "demo_results"
DEMO_DIR.mkdir(exist_ok=True)
(DEMO_DIR / "simulation").mkdir(exist_ok=True)
(DEMO_DIR / "evaluation").mkdir(exist_ok=True)

# ─── SIMULATOR PRESETS ─────────────────────────────────────────────────────
SIMULATION_PRESETS = [
    {
        "key": "garch__0.20__0.001__50000__500",
        "label": "Default (GARCH, moderate)",
        "description": "Standard GARCH volatility model with moderate parameters",
        "params": {
            "price_model": "garch",
            "sigma": 0.20,
            "lambda": 0.001,
            "eta": 0.10,
            "total_notional": 50000.0,
            "horizon_steps": 500,
            "n_paths": 500,
        },
        "tags": ["default"],
    },
    {
        "key": "garch__0.10__0.0001__50000__500",
        "label": "Low Volatility",
        "description": "Calm market conditions — AC optimal shines here",
        "params": {
            "price_model": "garch",
            "sigma": 0.10,
            "lambda": 0.0001,
            "eta": 0.10,
            "total_notional": 50000.0,
            "horizon_steps": 500,
            "n_paths": 500,
        },
        "tags": ["calm"],
    },
    {
        "key": "garch__0.40__0.01__50000__200",
        "label": "High Volatility",
        "description": "Volatile market — front-loading vs patience tradeoff",
        "params": {
            "price_model": "garch",
            "sigma": 0.40,
            "lambda": 0.01,
            "eta": 0.10,
            "total_notional": 50000.0,
            "horizon_steps": 200,
            "n_paths": 500,
        },
        "tags": ["volatile"],
    },
    {
        "key": "jump_garch__0.20__0.001__50000__500",
        "label": "Jump Diffusion",
        "description": "GARCH + Poisson price jumps (crash scenario)",
        "params": {
            "price_model": "jump_garch",
            "sigma": 0.20,
            "lambda": 0.001,
            "eta": 0.10,
            "total_notional": 50000.0,
            "horizon_steps": 500,
            "n_paths": 500,
        },
        "tags": ["crash", "jump"],
    },
    {
        "key": "garch__0.20__0.001__100000__500",
        "label": "Large Order",
        "description": "2× position size — impact costs become dominant",
        "params": {
            "price_model": "garch",
            "sigma": 0.20,
            "lambda": 0.001,
            "eta": 0.10,
            "total_notional": 100000.0,
            "horizon_steps": 500,
            "n_paths": 500,
        },
        "tags": ["large"],
    },
    {
        "key": "gbm__0.20__0.001__50000__500",
        "label": "GBM Baseline",
        "description": "Geometric Brownian Motion — textbook model",
        "params": {
            "price_model": "gbm",
            "sigma": 0.20,
            "lambda": 0.001,
            "eta": 0.10,
            "total_notional": 50000.0,
            "horizon_steps": 500,
            "n_paths": 500,
        },
        "tags": ["baseline", "gbm"],
    },
]

# ─── EVALUATION DATES ──────────────────────────────────────────────────────
EVALUATION_DATES = [
    {
        "date": "2024-01-15",
        "regime": "calm bull",
        "label": "Jan 15 — Calm Bull Market",
        "description": "Low volatility, steady upward trend",
        "is_test_date": False,
    },
    {
        "date": "2024-06-10",
        "regime": "quiet consolidation",
        "label": "Jun 10 — Quiet Consolidation",
        "description": "Low volume, tight spreads, OFI-based strategy optimal",
        "is_test_date": False,
    },
    {
        "date": "2024-08-05",
        "regime": "crash - Yen unwind",
        "label": "Aug 05 — Crash Day",
        "description": "Yen carry trade unwind — sudden 5%+ drop",
        "is_test_date": False,
    },
    {
        "date": "2024-04-15",
        "regime": "BTC correction",
        "label": "Apr 15 — BTC Correction (out-of-sample)",
        "description": "Out-of-sample: never seen during training",
        "is_test_date": True,
    },
]


def run_simulation_preset(preset: dict) -> dict:
    """
    Run research-sim binary for one preset and parse results.
    Reuses backend RustRunner service to maintain structural formatting.
    """
    p = preset["params"]
    runner = RustRunner()

    print(f"  Running simulation: model={p['price_model']} paths={p['n_paths']}")
    try:
        # Run asynchronously and return parsed dict
        data = asyncio.run(
            runner.run_simulation(
                n_paths=p["n_paths"],
                model=p["price_model"],
                params={
                    "sigma": p["sigma"],
                    "lambda": p["lambda"],
                    "eta": p["eta"],
                    "total_notional": p["total_notional"],
                    "horizon_steps": p["horizon_steps"],
                }
            )
        )
        return data
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def run_evaluation_date(date_info: dict) -> dict:
    """
    Run RL evaluation for one date using evaluate.py.
    """
    sys.path.insert(0, str(ROOT / "rl"))

    # Import evaluate module
    from evaluate import load_model, evaluate_model_on_date, test_vs_baseline, \
                         STATIC_OPTIMAL_IS, ADAPTIVE_OPTIMAL_IS, \
                         TWAP_IS, HEURISTIC_IS

    model_path = str(
        ROOT / "rl" / "models" /
        "ppo_lstm_v5_adaptive_best" / "best_model"
    )

    date = date_info["date"]
    print(f"  Evaluating {date} ({date_info['regime']})...")

    model = load_model(model_path)
    result = evaluate_model_on_date(
        model, date,
        n_episodes=50,
        n_state_dims=12,
    )

    # Out-of-sample date might not have baselines defined in DATES/STATIC_OPTIMAL_IS dicts
    baseline_is = STATIC_OPTIMAL_IS.get(date)
    if baseline_is is not None:
        stat_test = test_vs_baseline(result["is_samples"], baseline_is)
    else:
        stat_test = {
            "p_value": None,
            "ci_lower": float(np.percentile(result["is_samples"], 2.5)) if result["is_samples"] else None,
            "ci_upper": float(np.percentile(result["is_samples"], 97.5)) if result["is_samples"] else None,
            "significantly_better": None,
        }

    return {
        "date": date,
        "regime": date_info["regime"],
        "label": date_info["label"],
        "description": date_info["description"],
        "is_test_date": date_info["is_test_date"],
        "rl_is": result["mean_IS"],
        "rl_std": result["std_IS"],
        "rl_ci_lower": stat_test.get("ci_lower", result["mean_IS"]),
        "rl_ci_upper": stat_test.get("ci_upper", result["mean_IS"]),
        "rl_cvar95": result["cvar95"],
        "action_distribution": result.get("action_distribution", {}),
        "mean_action": result.get("mean_action", 7.5),
        "action_entropy": result.get("action_entropy", 0.0),
        "forced_liquidation_rate": result["forced_liquidation_rate"],
        "static_optimal_is": STATIC_OPTIMAL_IS.get(date),
        "adaptive_optimal_is": ADAPTIVE_OPTIMAL_IS.get(date),
        "twap_is": TWAP_IS.get(date),
        "heuristic_is": HEURISTIC_IS.get(date),
        "improvement_vs_ac": result["mean_IS"] - baseline_is if baseline_is is not None else None,
        "p_value": stat_test.get("p_value"),
        "significantly_better": stat_test.get("significantly_better"),
    }


def run_unified_comparison_all_dates(eval_dates) -> dict:
    """
    Run all 5 strategies (TWAP, Heuristic, AC Optimal, Adaptive AC, RL)
    through the same counterfactual LOB / OW impact pipeline for each
    evaluation date, so the comparison is methodologically valid.
    """
    sys.path.insert(0, str(ROOT / "rl"))
    from evaluate import load_model, evaluate_all_strategies_on_date

    model_path = str(
        ROOT / "rl" / "models" /
        "ppo_lstm_v5_adaptive_best" / "best_model"
    )
    model = load_model(model_path)

    unified = {}
    for date_info in eval_dates:
        date = date_info["date"]
        print(f"  Unified comparison: {date} ({date_info['regime']})...")
        date_results = evaluate_all_strategies_on_date(
            rl_model=model,
            date=date,
            n_episodes=50,
            n_state_dims=12,
        )
        unified[date] = {
            "date": date,
            "regime": date_info["regime"],
            "label": date_info["label"],
            "is_test_date": date_info["is_test_date"],
            "strategies": date_results,
        }

    return unified


def generate_manifest(sim_presets, eval_dates):
    """Generate manifest.json so frontend knows what's available."""
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "demo_mode": True,
        "simulation_presets": [
            {
                "key": p["key"],
                "label": p["label"],
                "description": p["description"],
                "tags": p["tags"],
                "params_summary": {
                    "model": p["params"]["price_model"],
                    "sigma": p["params"]["sigma"],
                    "lambda": p["params"]["lambda"],
                    "notional": p["params"]["total_notional"],
                    "horizon": p["params"]["horizon_steps"],
                },
            }
            for p in sim_presets
        ],
        "evaluation_dates": [
            {
                "date": d["date"],
                "label": d["label"],
                "regime": d["regime"],
                "is_test_date": d["is_test_date"],
            }
            for d in eval_dates
        ],
        "unavailable_features": [
            {
                "feature": "custom_lob_upload",
                "message": "Upload your own LOB data",
                "run_locally": "python scripts/run_local.py --lob-data your_file.csv",
            },
            {
                "feature": "custom_model_upload",
                "message": "Upload your own RL model",
                "run_locally": "python rl/evaluate.py --passive-model your_model/best_model",
            },
            {
                "feature": "custom_parameters",
                "message": "Simulate with custom parameter values",
                "run_locally": "cargo run --release --bin research-sim -- --sigma 0.35 ...",
            },
            {
                "feature": "custom_evaluation_date",
                "message": "Evaluate on a specific date",
                "run_locally": "python rl/evaluate.py --date 2024-09-15",
            },
        ],
    }


def main():
    print("=" * 60)
    print("StrataExec Pre-computation Script")
    print("=" * 60)

    # ── SIMULATIONS ───────────────────────────────────────────
    print("\n[1/2] Running simulation presets...")
    sim_results = {}

    for preset in SIMULATION_PRESETS:
        print(f"\nPreset: {preset['label']}")
        result = run_simulation_preset(preset)
        if result:
            output = {
                "preset_key": preset["key"],
                "label": preset["label"],
                "description": preset["description"],
                "tags": preset["tags"],
                "params": preset["params"],
                "results": result,
                "generated_at": datetime.utcnow().isoformat(),
            }
            path = DEMO_DIR / "simulation" / f"{preset['key']}.json"
            path.write_text(json.dumps(output, indent=2))
            sim_results[preset["key"]] = output
            print(f"  Saved: {path.name}")
        else:
            print(f"  SKIPPED (error above)")

    # ── EVALUATION ────────────────────────────────────────────
    print("\n[2/3] Running RL evaluations...")
    eval_results = {}

    for date_info in EVALUATION_DATES:
        print(f"\nDate: {date_info['label']}")
        try:
            result = run_evaluation_date(date_info)
            path = DEMO_DIR / "evaluation" / f"{date_info['date']}.json"
            path.write_text(json.dumps(result, indent=2))
            eval_results[date_info["date"]] = result
            ac_val = f"{result['static_optimal_is']:.3f}%" if result['static_optimal_is'] is not None else "N/A"
            imp_val = f"{result['improvement_vs_ac']:+.3f}pp" if result['improvement_vs_ac'] is not None else "N/A"
            print(f"  Saved: {path.name}")
            print(f"  RL IS: {result['rl_is']:.4f}%  "
                  f"vs AC: {ac_val}  "
                  f"improvement: {imp_val}")
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── ALL-DATES SUMMARY ─────────────────────────────────────
    if eval_results:
        summary_path = DEMO_DIR / "evaluation" / "all_dates.json"
        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "model": "ppo_lstm_v5_adaptive",
            "model_description": "RecurrentPPO LSTM, 64 hidden units, "
                                  "trained on 8 market regimes",
            "n_episodes_per_date": 50,
            "n_state_dims": 12,
            "dates": list(eval_results.values()),
            "summary_stats": {
                "mean_rl_is": sum(
                    r["rl_is"] for r in eval_results.values()
                ) / len(eval_results),
                "mean_improvement_vs_ac": sum(
                    r["improvement_vs_ac"] for r in eval_results.values()
                    if r["improvement_vs_ac"] is not None
                ) / len([r for r in eval_results.values() if r["improvement_vs_ac"] is not None]) if any(r["improvement_vs_ac"] is not None for r in eval_results.values()) else None,
                "dates_beating_ac": sum(
                    1 for r in eval_results.values()
                    if r["improvement_vs_ac"] is not None and r["improvement_vs_ac"] > 0
                ),
                "total_dates": len(eval_results),
                "oos_dates": sum(
                    1 for r in eval_results.values()
                    if r["is_test_date"]
                ),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved all-dates summary: {summary_path.name}")

    # ── UNIFIED COMPARISON ────────────────────────────────────
    print("\n[3/3] Running unified comparison (all strategies, same "
          "counterfactual LOB)...")
    try:
        unified = run_unified_comparison_all_dates(EVALUATION_DATES)
        unified_path = DEMO_DIR / "evaluation" / "unified_comparison.json"
        unified_path.write_text(json.dumps({
            "generated_at": datetime.utcnow().isoformat(),
            "n_episodes_per_date": 50,
            "n_state_dims": 12,
            "dates": unified,
        }, indent=2))
        print(f"  Saved: {unified_path.name}")
    except Exception as e:
        print(f"  SKIPPED unified comparison: {e}")

    # ── MANIFEST ──────────────────────────────────────────────
    manifest = generate_manifest(SIMULATION_PRESETS, EVALUATION_DATES)
    manifest_path = DEMO_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved manifest: {manifest_path}")

    # ── SUMMARY ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pre-computation complete.")
    print(f"  Simulations:  {len(sim_results)} / {len(SIMULATION_PRESETS)}")
    print(f"  Evaluations:  {len(eval_results)} / {len(EVALUATION_DATES)}")
    print(f"\nNext steps:")
    print(f"  git add demo_results/")
    print(f"  git commit -m 'add pre-computed demo results'")
    print(f"  git push")
    print("=" * 60)


if __name__ == "__main__":
    main()
