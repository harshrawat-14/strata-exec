"""
Parallel simulation execution engine.
Runs multiple single-path Rust simulation subprocesses concurrently
and aggregates their results.
"""
from __future__ import annotations

import asyncio
import csv
import json
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from web.config import get_settings
from web.services.rust_runner import RustRunnerError

settings = get_settings()


def parse_comparison_csv(csv_path: str) -> dict[str, Any]:
    """Parse a single comparison.csv file into structured lists."""
    if not os.path.exists(csv_path):
        return {"price_path": [], "strategies": {}}

    price_path = []
    strategies = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Find strategies from header (e.g. twap_remaining)
        strategy_prefixes = []
        for h in headers:
            if h.endswith("_remaining"):
                strategy_prefixes.append(h[:-10])

        for prefix in strategy_prefixes:
            strategies[prefix] = {"trajectory": [], "cost_series": []}

        for row in reader:
            try:
                price_path.append(float(row.get("Price", 0.0)))
            except ValueError:
                price_path.append(0.0)

            for prefix in strategy_prefixes:
                try:
                    strategies[prefix]["trajectory"].append(
                        float(row.get(f"{prefix}_remaining", 0.0))
                    )
                    strategies[prefix]["cost_series"].append(
                        float(row.get(f"{prefix}_cost", 0.0))
                    )
                except ValueError:
                    strategies[prefix]["trajectory"].append(0.0)
                    strategies[prefix]["cost_series"].append(0.0)

    return {"price_path": price_path, "strategies": strategies}


def aggregate_runs(
    runs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate the outputs of multiple single-path runs:
    1. Average the prices and strategy trajectories/cost series step-by-step.
    2. Compute statistical distributions (Mean, Variance, CVaR 95%) from the path samples.
    """
    if not runs or not summaries:
        return {"strategies": [], "price_path": [], "total_steps": 0}

    # 1. Average step trajectories
    total_steps = len(runs[0]["price_path"])
    price_arrays = [np.array(r["price_path"]) for r in runs if len(r["price_path"]) == total_steps]
    avg_price = np.mean(price_arrays, axis=0).tolist() if price_arrays else []

    strategy_names = list(runs[0]["strategies"].keys())
    avg_strategies = []

    for strat_prefix in strategy_names:
        trajectories = []
        cost_series = []
        for r in runs:
            s_data = r["strategies"].get(strat_prefix)
            if s_data and len(s_data["trajectory"]) == total_steps:
                trajectories.append(np.array(s_data["trajectory"]))
                cost_series.append(np.array(s_data["cost_series"]))

        avg_traj = np.mean(trajectories, axis=0).tolist() if trajectories else []
        avg_costs = np.mean(cost_series, axis=0).tolist() if cost_series else []

        # 2. Gather strategy-wise statistics across paths
        shortfalls = []
        is_variances = []
        ac_objectives = []
        trade_counts = []
        avg_exec_prices = []
        spread_costs = []
        temp_impacts = []
        perm_impacts = []
        timing_costs = []
        opportunity_costs = []

        # Map display name from prefix
        display_name = {
            "twap": "TWAP",
            "heuristic": "Heuristic",
            "optimal": "Optimal (AC)",
            "adaptiveoptimal": "Adaptive Optimal",
            "regimeac": "RegimeAC",
            "rlagent": "RL Agent",
        }.get(strat_prefix, strat_prefix)

        for s in summaries:
            strat_info = s.get(display_name)
            if not strat_info:
                # Try prefix match or fallback
                for k, v in s.items():
                    if k.lower().replace(" ", "").replace("-", "").replace("(", "").replace(")", "") == strat_prefix:
                        strat_info = v
                        break

            if strat_info:
                shortfalls.append(strat_info.get("mean_is_pct", 0.0))
                ac_objectives.append(strat_info.get("ac_objective", 0.0))
                trade_counts.append(strat_info.get("trade_count", 0.0))
                avg_exec_prices.append(strat_info.get("avg_exec_price", 0.0))

                decomp = strat_info.get("cost_decomposition", {})
                spread_costs.append(decomp.get("spread_cost", 0.0))
                temp_impacts.append(decomp.get("temporary_impact", 0.0))
                perm_impacts.append(decomp.get("permanent_impact", 0.0))
                timing_costs.append(decomp.get("timing_cost", 0.0))
                opportunity_costs.append(decomp.get("opportunity_cost", 0.0))

        # Calculate moments
        n = len(shortfalls)
        if n > 0:
            mean_is = float(np.mean(shortfalls))
            # is_variance: variance of IS across paths (cross-path dispersion)
            is_var = float(np.var(shortfalls, ddof=1)) if n > 1 else 0.0
            # CVaR 95% = average of the worst 5% of outcomes (meaning highest shortfall/cost)
            # Since shortfall is negative for savings / positive for cost, sort descending
            sorted_is = sorted(shortfalls, reverse=True)
            tail_idx = max(1, int(np.ceil(0.05 * n)))
            cvar95 = float(np.mean(sorted_is[:tail_idx]))

            mean_ac = float(np.mean(ac_objectives))
            mean_tc = float(np.mean(trade_counts))
            mean_ap = float(np.mean(avg_exec_prices))

            mean_spread = float(np.mean(spread_costs))
            mean_temp = float(np.mean(temp_impacts))
            mean_perm = float(np.mean(perm_impacts))
            mean_timing = float(np.mean(timing_costs))
            mean_opp = float(np.mean(opportunity_costs))
        else:
            mean_is, is_var, cvar95, mean_ac = 0.0, 0.0, 0.0, 0.0
            mean_tc, mean_ap = 0.0, 0.0
            mean_spread, mean_temp, mean_perm, mean_timing, mean_opp = 0.0, 0.0, 0.0, 0.0, 0.0

        avg_strategies.append({
            "name": display_name,
            "trajectory": avg_traj,
            "cost_series": avg_costs,
            "mean_is_pct": mean_is,
            "is_variance": is_var,
            "cvar95": cvar95,
            "ac_objective": mean_ac,
            "trade_count": mean_tc,
            "avg_exec_price": mean_ap,
            "cost_decomposition": {
                "spread_cost": mean_spread,
                "temporary_impact": mean_temp,
                "permanent_impact": mean_perm,
                "timing_cost": mean_timing,
                "opportunity_cost": mean_opp,
            }
        })

    return {
        "strategies": avg_strategies,
        "price_path": avg_price,
        "total_steps": total_steps,
    }


def simulate_rl_path(model, price_path, params):
    import numpy as np
    
    ACTION_FRACTIONS = [
        0.000, 0.001, 0.003, 0.005, 0.008,
        0.010, 0.015, 0.020, 0.030, 0.050,
        0.070, 0.100, 0.150, 0.200, 0.300,
        0.400, 0.500, 0.700, 0.900, 1.000
    ]
    
    total_inv = 1_000_000.0
    sigma_ref = 0.02
    if params:
        total_inv = params.get("total_notional", 1_000_000.0)
        sigma_ref = params.get("sigma", 0.02)
        
    daily_volume = 5_000_000.0
    impact_coef = 0.5
    gamma = 0.005
    rho = 50.0
    
    n_steps = len(price_path) - 1
    if n_steps <= 0:
        n_steps = 1
    dt = 1.0 / n_steps
    
    remaining = total_inv
    price_adjustment = 0.0
    trades = []
    
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    
    trajectory = [remaining]
    cost_series = [0.0]
    
    spread_cost_acc = 0.0
    temporary_impact_acc = 0.0
    permanent_impact_acc = 0.0
    timing_cost_acc = 0.0
    
    price_history = []
    return_buffer = []
    last_slippage = 0.0
    
    total_filled = 0.0
    total_notional_received = 0.0
    
    for step in range(n_steps):
        price = price_path[step]
        price_history.append(price)
        if len(price_history) > 16:
            price_history.pop(0)
            
        current_time = step * dt
        
        # Volatility
        volatility = sigma_ref
        if step > 0:
            log_ret = np.log(price / price_path[step - 1])
            if np.isfinite(log_ret):
                return_buffer.append(log_ret)
                if len(return_buffer) > 50:
                    return_buffer.pop(0)
        
        if len(return_buffer) >= 5:
            per_step_vol = np.std(return_buffer)
            realised_vol = per_step_vol * np.sqrt(n_steps)
        else:
            realised_vol = sigma_ref
            
        half_spread = 0.5 * volatility * np.sqrt(dt) * price
        
        # Build state
        s0 = remaining / total_inv
        s1 = (n_steps - step) / n_steps
        s2 = np.tanh(realised_vol / sigma_ref) if sigma_ref > 0 else 0.0
        s3 = np.tanh(half_spread / (0.01 * price))
        s4 = 0.0
        s5 = 0.0
        
        if len(price_history) >= 11:
            now = price_history[-1]
            then = price_history[-11]
            drift = (now - then) / then if then > 0 else 0.0
        else:
            drift = 0.0
        s6 = np.tanh(drift / sigma_ref) if sigma_ref > 0 else 0.0
        s7 = np.tanh(last_slippage / (0.01 * price))
        
        # Check model observation space shape (some models are trained on 12-dim counterfactual mode)
        obs_dim = 8
        if hasattr(model, "observation_space") and hasattr(model.observation_space, "shape"):
            obs_dim = model.observation_space.shape[0]

        if obs_dim == 12:
            s8 = 0.0  # perm impact fraction
            s9 = 0.0  # temp impact fraction
            s10 = 0.5  # normal vol regime
            s11 = 0.5  # normal liq regime
            state_vec = np.array([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11], dtype=np.float32)
        else:
            state_vec = np.array([s0, s1, s2, s3, s4, s5, s6, s7], dtype=np.float32)

        action, lstm_states = model.predict(
            state_vec,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True
        )
        episode_start = np.zeros((1,), dtype=bool)
        
        fraction = ACTION_FRACTIONS[int(action)]
        traded = fraction * remaining
        
        transient_impact = 0.0
        for t, base in trades:
            delta_t = current_time - t
            if delta_t >= 0.0:
                transient_impact += base * np.exp(-rho * delta_t)
                
        effective_price = price * (1.0 - price_adjustment)
        exec_price = price * (1.0 - price_adjustment - transient_impact)
        
        if traded > 0.0:
            q_over_v = traded / daily_volume
            base_impact = impact_coef * volatility * np.sqrt(q_over_v)
            trades.append((current_time, base_impact))
            
            last_slippage = exec_price - effective_price
            delta_permanent = gamma * (traded / total_inv)
            price_adjustment += delta_permanent
            
            total_filled += traded
            total_notional_received += traded * exec_price
            remaining -= traded
            
            spread_cost = traded * half_spread
            temp_impact_cost = traded * transient_impact * price
            perm_impact_cost = traded * delta_permanent * price
            realized_cost = traded * (exec_price - price_path[0])
            timing_cost = realized_cost - spread_cost - temp_impact_cost - perm_impact_cost
            
            spread_cost_acc += spread_cost
            temporary_impact_acc += temp_impact_cost
            permanent_impact_acc += perm_impact_cost
            timing_cost_acc += timing_cost
        else:
            last_slippage = 0.0
            
        trajectory.append(remaining)
        
        ideal = price_path[0] * (total_inv - remaining)
        shortfall = ideal - total_notional_received
        cost_series.append(shortfall)
        
    opportunity_cost = 0.0
    if remaining > 0.0:
        liquidated_at = price_path[-1] * (1.0 - price_adjustment)
        forced_cost = remaining * (liquidated_at - price_path[0])
        total_notional_received += remaining * liquidated_at
        opportunity_cost = forced_cost
        remaining = 0.0
        trajectory[-1] = 0.0
        
    ideal = price_path[0] * total_inv
    shortfall = ideal - total_notional_received
    cost_series[-1] = shortfall
    
    mean_is_pct = shortfall / ideal * 100.0 if ideal > 0 else 0.0
    
    return {
        "trajectory": trajectory,
        "cost_series": cost_series,
        "mean_is_pct": mean_is_pct,
        "trade_count": len([t for t in trades if t[1] > 0]) + (1 if opportunity_cost > 0 else 0),
        "avg_exec_price": total_notional_received / total_inv if total_inv > 0 else price_path[0],
        "cost_decomposition": {
            "spread_cost": spread_cost_acc,
            "temporary_impact": temporary_impact_acc,
            "permanent_impact": permanent_impact_acc,
            "timing_cost": timing_cost_acc,
            "opportunity_cost": opportunity_cost,
        }
    }


async def run_parallel_simulation(
    n_paths: int,
    model: str = "gbm",
    lob_path: str | None = None,
    agg_path: str | None = None,
    include_regime_ac: bool = False,
    params: dict[str, Any] | None = None,
    on_progress: Callable | None = None,
    max_concurrent: int = 16,
    job_id: str | None = None,
    rl_model: Any | None = None,
) -> dict[str, Any]:
    """
    Run N simulation paths concurrently.
    Each path spawns a separate `research-sim` subprocess.
    """
    research_sim_path = str(settings.research_sim_path)
    if not os.path.isfile(research_sim_path):
        raise RustRunnerError(f"Binary 'research-sim' not found at '{research_sim_path}'. Run 'cargo build --release' first.")

    cwd = Path(settings.rust_binary_path).parent.parent
    if not cwd.exists():
        cwd = Path(".")

    semaphore = asyncio.Semaphore(max_concurrent)
    completed_paths = 0
    runs_results: list[dict] = []
    summaries_results: list[dict] = []
    task_lock = asyncio.Lock()

    async def run_single_path(path_id: int):
        nonlocal completed_paths
        async with semaphore:
            # Create a unique temporary results directory for this path
            temp_dir = tempfile.mkdtemp(prefix=f"strata_sim_{path_id}_")
            try:
                cmd = [
                    research_sim_path,
                    "--paths", "1",
                    "--seed", str(42 + path_id),
                    "--model", model,
                    "--results-dir", temp_dir,
                ]
                if lob_path:
                    cmd.extend(["--lob-data", lob_path])
                if agg_path:
                    cmd.extend(["--agg-trades", agg_path])
                if include_regime_ac:
                    cmd.append("--include-regime-ac")

                if params:
                    if "sigma" in params and params["sigma"] is not None:
                        cmd.extend(["--sigma", str(params["sigma"])])
                    if "eta" in params and params["eta"] is not None:
                        cmd.extend(["--eta", str(params["eta"])])
                    if "lambda" in params and params["lambda"] is not None:
                        cmd.extend(["--lambda", str(params["lambda"])])
                    if "total_notional" in params and params["total_notional"] is not None:
                        cmd.extend(["--notional", str(params["total_notional"])])
                    if "horizon_steps" in params and params["horizon_steps"] is not None:
                        cmd.extend(["--horizon", str(params["horizon_steps"])])

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd),
                )
                if job_id:
                    from web.services.job_registry import register_subprocess
                    await register_subprocess(job_id, proc)

                try:
                    stdout, stderr = await proc.communicate()
                finally:
                    if job_id:
                        from web.services.job_registry import unregister_subprocess
                        await unregister_subprocess(job_id, proc)

                if proc.returncode != 0:
                    err_msg = stderr.decode(errors="replace")
                    raise RustRunnerError(f"Path {path_id} failed with exit code {proc.returncode}. Stderr: {err_msg}")

                # Read and parse CSV & Summary JSON
                csv_file = os.path.join(temp_dir, "comparison.csv")
                summary_file = os.path.join(temp_dir, "summary.json")

                run_data = parse_comparison_csv(csv_file)
                
                summary_data = {}
                if os.path.exists(summary_file):
                    with open(summary_file) as sf:
                        summary_data = json.load(sf)

                if rl_model is not None and len(run_data["price_path"]) > 0:
                    try:
                        rl_res = simulate_rl_path(rl_model, run_data["price_path"], params)
                        run_data["strategies"]["rlagent"] = {
                            "trajectory": rl_res["trajectory"],
                            "cost_series": rl_res["cost_series"],
                        }
                        summary_data["RL Agent"] = {
                            "mean_is_pct": rl_res["mean_is_pct"],
                            # is_variance is 0 per path — aggregated across paths in aggregate_runs
                            "is_variance": 0.0,
                            "ac_objective": rl_res["mean_is_pct"],
                            "trade_count": rl_res["trade_count"],
                            "avg_exec_price": rl_res["avg_exec_price"],
                            "cost_decomposition": rl_res["cost_decomposition"],
                        }
                    except Exception as e:
                        import sys, traceback
                        print(f"[RL] Error on path {path_id}: {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)

                async with task_lock:
                    runs_results.append(run_data)
                    summaries_results.append(summary_data)
                    completed_paths += 1
                    
                    partial_agg = None
                    try:
                        partial_agg = aggregate_runs(list(runs_results), list(summaries_results))
                    except Exception:
                        pass

                    if on_progress:
                        await on_progress(completed_paths, n_paths, partial_agg)

            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)

    tasks = [run_single_path(i) for i in range(n_paths)]
    await asyncio.gather(*tasks)

    # Perform statistical aggregation across all paths
    aggregated = aggregate_runs(runs_results, summaries_results)
    return aggregated
