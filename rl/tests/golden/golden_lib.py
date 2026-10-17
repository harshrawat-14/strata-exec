"""
Golden-output regression harness for behaviour-preserving refactors.

Records what the simulator and the evaluation harness produce TODAY for a
fixed set of configs, seeds and action sequences, so a refactor can be
checked for exact (bit-level) equivalence rather than "the tests still pass".

Three layers are captured:
  1. rl-env over JSON/stdio  -- the raw output line of every reset/step,
     hashed per episode (exact bytes, so float formatting, info-key order and
     every info field are covered).
  2. rl-env batched (Arrow)  -- states/rewards/dones bytes per step, hashed.
  3. evaluate.py harness     -- per-strategy summary metrics on training
     dates, compared exactly.

Only TRAINING_DATES_V2 are used. HELD_OUT_DATES_V2 are never touched here.

Regenerate fixtures ONLY on a commit whose behaviour is known-good:
    python rl/tests/golden/generate.py
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent
ENV_FIXTURE = GOLDEN_DIR / "env_golden.json"
EVAL_FIXTURE = GOLDEN_DIR / "eval_golden.json"

N_ACTIONS = 20
HONEST_LAMBDA = 1e-6

CALM_DATE = "2024-01-15"
CRASH_DATE = "2024-07-05"


def _book(date: str) -> str:
    return f"TradeData/BookDepth/BTCUSDT-bookDepth-{date}.csv"


def _agg(date: str) -> str:
    return f"TradeData/AggTrades/BTCUSDT-aggTrades-{date}.csv"


# name -> extra CLI args. Chosen so every RlMode, the adversarial path, the
# random-horizon/size draws, both state widths and forced liquidation (via
# the all-wait policy) are exercised.
STDIO_CONFIGS = {
    "synthetic_default": ["--rl-mode", "synthetic"],
    "synthetic_adversarial": ["--rl-mode", "synthetic", "--adversarial"],
    "historical_calm": [
        "--rl-mode", "historical", "--lob-data", _book(CALM_DATE),
        "--agg-trades", _agg(CALM_DATE), "--btc-target", "50000",
        "--fixed-steps", "--fixed-size",
    ],
    "counterfactual_calm_16d_fixed": [
        "--rl-mode", "counterfactual", "--lob-data", _book(CALM_DATE),
        "--agg-trades", _agg(CALM_DATE), "--btc-target", "50000",
        "--fixed-steps", "--fixed-size",
    ],
    "counterfactual_crash_16d_fixed": [
        "--rl-mode", "counterfactual", "--lob-data", _book(CRASH_DATE),
        "--agg-trades", _agg(CRASH_DATE), "--btc-target", "50000",
        "--fixed-steps", "--fixed-size",
    ],
    "counterfactual_crash_12d_random_horizon": [
        "--rl-mode", "counterfactual", "--lob-data", _book(CRASH_DATE),
        "--agg-trades", _agg(CRASH_DATE), "--btc-target", "50000",
        "--no-time-features",
    ],
}

POLICIES = ("random", "all_wait", "all_max")
SEEDS = (0, 17)
MAX_STEPS = 5000  # safety stop; every config terminates well before this


def _policy_action(policy: str, rng: random.Random) -> int:
    if policy == "random":
        return rng.randrange(N_ACTIONS)
    if policy == "all_wait":
        return 0
    return N_ACTIONS - 1


def _run_stdio_config(binary: str, args: list[str]) -> dict:
    """One process per config; episodes run back-to-back in it, so state
    leaking across reset() is part of what is checked."""
    proc = subprocess.Popen(
        [binary, *args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    def send(obj: dict) -> bytes:
        proc.stdin.write((json.dumps(obj) + "\n").encode())
        proc.stdin.flush()
        return proc.stdout.readline()

    out = {}
    try:
        for policy in POLICIES:
            for seed in SEEDS:
                rng = random.Random(seed * 1000 + 7)
                h = hashlib.sha256()
                line = send({"reset": True, "seed": seed})
                h.update(line)
                n_steps, total_reward, last = 0, 0.0, json.loads(line)
                while not last["done"] and n_steps < MAX_STEPS:
                    line = send({"action": _policy_action(policy, rng)})
                    h.update(line)
                    last = json.loads(line)
                    total_reward += last["reward"]
                    n_steps += 1
                out[f"{policy}/seed{seed}"] = {
                    "sha256": h.hexdigest(),
                    "n_steps": n_steps,
                    "total_reward": total_reward,
                    "final_info": last["info"],
                }
    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait()
    return out


def _run_cli_query(binary: str, args: list[str]) -> str:
    return subprocess.run(
        [binary, *args], capture_output=True, text=True, check=True, timeout=120,
    ).stdout


def _run_batched(mode: str, date: str | None, seeds: list[int]) -> dict:
    import numpy as np
    from batched_environment import BatchedStrataExecEnv

    rng = random.Random(4242)
    h = hashlib.sha256()
    n_steps = 0
    with BatchedStrataExecEnv(
        batch_size=len(seeds), mode=mode, lob_date=date, agg_date=date,
        fixed_steps=True, fixed_size=True, no_time_features=False,
    ) as env:
        states = env.reset_batch(seeds)
        h.update(np.ascontiguousarray(states).tobytes())
        # Fixed number of steps: batched envs auto-continue after done, and
        # what happens across that boundary is part of the contract too.
        for _ in range(400):
            actions = [rng.randrange(N_ACTIONS) for _ in seeds]
            states, rewards, dones = env.step_batch(actions)
            for arr in (states, rewards, dones):
                h.update(np.ascontiguousarray(arr).tobytes())
            n_steps += 1
    return {"sha256": h.hexdigest(), "n_steps": n_steps}


def collect_env_golden(binary: str) -> dict:
    return {
        "stdio": {name: _run_stdio_config(binary, args)
                  for name, args in STDIO_CONFIGS.items()},
        "batched": {
            "synthetic": _run_batched("synthetic", None, [0, 1, 2, 3]),
            "counterfactual_crash": _run_batched("counterfactual", CRASH_DATE, [0, 1, 2, 3]),
        },
        "print_action_fractions": _run_cli_query(binary, ["--print-action-fractions"]),
        "print_calibration": _run_cli_query(binary, [
            "--rl-mode", "counterfactual", "--lob-data", _book(CRASH_DATE),
            "--agg-trades", _agg(CRASH_DATE), "--btc-target", "50000",
            "--print-calibration",
        ]),
    }


# Step 4 checkpoint: the current headline model.
EVAL_MODEL = "rl/models/phase3_step4_capacity_entropy_best/best_model"
EVAL_CASES = {
    "counterfactual_calm": dict(date=CALM_DATE, mode="counterfactual"),
    "counterfactual_crash": dict(date=CRASH_DATE, mode="counterfactual"),
    "historical_crash": dict(date=CRASH_DATE, mode="historical"),
}
EVAL_FIELDS = ("mean_IS", "std_IS", "cvar_95", "cvar_95_ci_lower",
               "cvar_95_ci_upper", "n_episodes")


def collect_eval_golden() -> dict:
    from evaluate import evaluate_all_strategies_on_date, load_model

    model = load_model(EVAL_MODEL)
    out = {}
    for name, case in EVAL_CASES.items():
        results = evaluate_all_strategies_on_date(
            rl_model=model, n_episodes=3, n_state_dims=16,
            btc_target=50_000.0, ac_lambda=HONEST_LAMBDA, n_bootstrap=200,
            include_sqrt_impact_optimal=(case["mode"] == "counterfactual"),
            sqrt_impact_n_fit_seeds=1, sqrt_impact_maxiter=3,
            **case,
        )
        out[name] = {
            r["strategy"]: {f: _jsonable(r[f]) for f in EVAL_FIELDS}
            for r in results.values()
        }
    return out


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, (int,)) or type(v).__name__.startswith("int"):
        return int(v)
    return float(v)
