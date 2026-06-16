"""
Background worker functions for simulation, RL evaluation, and sweeps.
These run under the ARQ distributed worker pool.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from web.config import get_settings
from web.models.database import AsyncSessionLocal, SimulationJob, EvaluationJob, SweepJob
from web.services.model_cache import model_cache
from web.services.parallel_sim import run_parallel_simulation
from web.services.rust_runner import get_rust_runner

settings = get_settings()


async def publish_progress(redis_conn: Any, job_id: str, msg: dict):
    """Publish progress updates to Redis pub/sub channel for SSE streaming."""
    if redis_conn:
        try:
            # 1. Publish to Redis Pub/Sub channel
            await redis_conn.publish(f"job:{job_id}", json.dumps(msg))

            # 2. Update job state snapshot in Redis for tab reconnection persistence
            redis_key = f"job:progress:{job_id}"
            
            # Fetch existing snapshot if any
            existing_raw = await redis_conn.get(redis_key)
            snapshot = {}
            if existing_raw:
                try:
                    snapshot = json.loads(existing_raw)
                except Exception:
                    pass

            msg_type = msg.get("type")
            
            if msg_type == "status" and msg.get("status") == "running":
                snapshot = {
                    "job_id": job_id,
                    "progress": 0,
                    "status": "running",
                    "partial_results": {},
                    "started_at": datetime.utcnow().isoformat(),
                    "last_updated": datetime.utcnow().isoformat(),
                }
            elif msg_type == "progress":
                completed = msg.get("completed", 0)
                total = msg.get("total", 1)
                snapshot["progress"] = round((completed / total) * 100, 1)
                snapshot["status"] = "running"
                snapshot["last_updated"] = datetime.utcnow().isoformat()
                if "started_at" not in snapshot:
                    snapshot["started_at"] = datetime.utcnow().isoformat()
            elif msg_type == "paths_update":
                paths_done = msg.get("paths_done", 0)
                paths_total = msg.get("paths_total", 1)
                snapshot["progress"] = round((paths_done / paths_total) * 100, 1)
                snapshot["status"] = "running"
                snapshot["partial_results"] = msg.get("partial_results", {})
                snapshot["last_updated"] = datetime.utcnow().isoformat()
                if "started_at" not in snapshot:
                    snapshot["started_at"] = datetime.utcnow().isoformat()
            elif msg_type == "date_complete":
                dates_done = msg.get("dates_done", 0)
                dates_total = msg.get("dates_total", 1)
                snapshot["progress"] = round((dates_done / dates_total) * 100, 1)
                snapshot["status"] = "running"
                
                # Accumulate partial evaluation dates
                if not isinstance(snapshot.get("partial_results"), list):
                    snapshot["partial_results"] = []
                
                existing_dates = [d.get("date") for d in snapshot["partial_results"]]
                if msg.get("date") not in existing_dates:
                    snapshot["partial_results"].append({
                        "date": msg.get("date"),
                        "regime": msg.get("regime"),
                        "rl_is": msg.get("rl_is"),
                        "ac_is": msg.get("ac_is"),
                        "improvement_pp": msg.get("improvement_pp"),
                    })
                
                snapshot["last_updated"] = datetime.utcnow().isoformat()
                if "started_at" not in snapshot:
                    snapshot["started_at"] = datetime.utcnow().isoformat()
            elif msg_type == "complete":
                snapshot = {
                    "job_id": job_id,
                    "progress": 100,
                    "status": "complete",
                    "results": msg.get("results"),
                    "completed_at": datetime.utcnow().isoformat(),
                }
            elif msg_type == "error":
                snapshot = {
                    "job_id": job_id,
                    "progress": 100,
                    "status": "failed",
                    "error": msg.get("message"),
                    "completed_at": datetime.utcnow().isoformat(),
                }

            if snapshot:
                await redis_conn.setex(redis_key, 3600, json.dumps(snapshot))

        except Exception as e:
            print(f"Error in publish_progress state caching: {e}", file=sys.stderr)


async def run_simulation_job(
    ctx: dict,
    job_id: str,
    request_data: dict[str, Any],
    lob_path: str | None = None,
    agg_path: str | None = None,
):
    """
    ARQ Task: Runs multi-path simulation using the parallel execution engine.
    Publishes real-time progress via Redis pub/sub.
    """
    start = time.time()
    redis_conn = ctx.get("redis")

    async def _update_status(status: str, extra: dict | None = None):
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            result = await db.execute(
                select(SimulationJob).where(SimulationJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = status
                if extra:
                    for k, v in extra.items():
                        setattr(job, k, v)
                await db.commit()

    async def _publish(msg: dict):
        await publish_progress(redis_conn, job_id, msg)

    # Register in job registry
    from web.services.job_registry import register_task, unregister_task
    await register_task(job_id, asyncio.current_task())

    try:
        await _update_status("running")
        await _publish({"type": "status", "status": "running"})

        n_paths = request_data.get("n_paths", 100)
        model = request_data.get("price_model", "gbm")
        include_regime_ac = request_data.get("include_regime_ac", False)

        def decimate_list(lst, target_len=50):
            if not lst:
                return []
            if len(lst) <= target_len:
                return lst
            step = len(lst) / target_len
            return [lst[int(i * step)] for i in range(target_len)]

        async def on_progress(completed: int, total: int, partial_agg: dict | None = None):
            # Format partial results
            partial_results = {}
            if partial_agg and "strategies" in partial_agg:
                for s in partial_agg["strategies"]:
                    name_lower = s["name"].lower().replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
                    key = {
                        "twap": "twap",
                        "heuristic": "heuristic",
                        "optimalac": "optimal",
                        "adaptiveoptimal": "adaptive",
                        "rlagent": "rl",
                    }.get(name_lower, name_lower)
                    partial_results[key] = {
                        "mean_is": round(s.get("mean_is_pct", 0.0), 4),
                        "variance": round(s.get("is_variance", 0.0), 4),
                        "cost_series": decimate_list(s.get("cost_series", [])),
                    }

            await _publish({
                "type": "paths_update",
                "paths_done": completed,
                "paths_total": total,
                "partial_results": partial_results,
            })

        # Resolve and load RL model if requested
        include_rl = request_data.get("include_rl", False)
        rl_model = None
        if include_rl:
            rl_model_path = str(Path(settings.model_path) / "ppo_lstm_v5_adaptive_best" / "best_model.zip")
            if request_data.get("rl_model_id"):
                model_id = request_data["rl_model_id"]
                async with AsyncSessionLocal() as db:
                    from sqlalchemy import select
                    from web.models.database import UploadedModel
                    result_db = await db.execute(
                        select(UploadedModel).where(UploadedModel.id == model_id)
                    )
                    model_obj = result_db.scalar_one_or_none()
                    if model_obj:
                        rl_model_path = model_obj.stored_path
            
            rl_model = await model_cache.get(rl_model_path)

        # Run paths concurrently using our parallel_sim service
        result = await run_parallel_simulation(
            n_paths=n_paths,
            model=model,
            lob_path=lob_path,
            agg_path=agg_path,
            include_regime_ac=include_regime_ac,
            params=request_data.get("params"),
            on_progress=on_progress,
            max_concurrent=16,  # Enforce limit of concurrent subprocesses
            job_id=job_id,
            rl_model=rl_model,
        )

        duration = time.time() - start
        result["job_id"] = job_id
        result["status"] = "complete"
        result["params_used"] = request_data.get("params", {})
        result["duration_seconds"] = round(duration, 2)
        result["created_at"] = datetime.utcnow().isoformat()

        await _update_status("complete", {
            "result_data": json.dumps(result),
            "completed_at": datetime.utcnow(),
            "duration_seconds": duration,
        })

        await _publish({
            "type": "complete",
            "job_id": job_id,
            "results_url": f"/api/compare/{job_id}",
            "results": result,
        })

    except asyncio.CancelledError:
        err_msg = "Cancelled by user"
        await _update_status("failed", {"error_message": err_msg})
        await _publish({"type": "error", "message": err_msg})
        raise
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"Simulation failed: {e}\n{tb}"
        await _update_status("failed", {"error_message": err_msg[:2000]})
        await _publish({"type": "error", "message": str(e)})
    finally:
        await unregister_task(job_id)


async def run_evaluation_job(ctx: dict, job_id: str, request_data: dict[str, Any]):
    """
    ARQ Task: Runs RL evaluations using PyTorch model loading cache.
    """
    start = time.time()
    redis_conn = ctx.get("redis")

    async def _update_status(status: str, extra: dict | None = None):
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            result = await db.execute(
                select(EvaluationJob).where(EvaluationJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = status
                if extra:
                    for k, v in extra.items():
                        setattr(job, k, v)
                await db.commit()

    async def _publish(msg: dict):
        await publish_progress(redis_conn, job_id, msg)

    # Register in job registry
    from web.services.job_registry import register_task, unregister_task
    await register_task(job_id, asyncio.current_task())

    try:
        await _update_status("running")
        await _publish({"type": "status", "status": "running"})

        model_id = request_data["model_id"]
        dates = request_data.get("dates", [])
        n_episodes = request_data.get("n_episodes", 50)

        # Resolve model path from Database
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            from web.models.database import UploadedModel
            result = await db.execute(
                select(UploadedModel).where(UploadedModel.id == model_id)
            )
            model_obj = result.scalar_one_or_none()

        if not model_obj:
            raise ValueError(f"Model {model_id} not found in database")

        model_path = model_obj.stored_path

        # Load model using the LRU model loader cache
        model = await model_cache.get(model_path)

        # Auto-detect state dimensions from the loaded model
        n_state_dims = 8
        if hasattr(model, "observation_space") and model.observation_space is not None:
            n_state_dims = model.observation_space.shape[0]

        # Run evaluation in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        # Run each date sequentially to allow progress updates
        date_results = []
        dates_total = len(dates)

        # Define evaluation regimes mapping
        REGIMES = {
            "2024-01-15": "calm bull",
            "2024-03-05": "BTC breakout",
            "2024-06-10": "quiet consolidation",
            "2024-08-05": "crash - Yen unwind",
            "2024-11-06": "post-election surge",
        }

        loop = asyncio.get_event_loop()

        for idx, date in enumerate(dates):
            def _run_single_date(d):
                # Add project root to sys.path
                project_root = str(Path(settings.rust_binary_path).parent.parent)
                rl_dir = str(Path(project_root) / "rl")
                for p in [project_root, rl_dir]:
                    if p not in sys.path:
                        sys.path.insert(0, p)

                from evaluate import evaluate_model_on_date, test_vs_baseline, STATIC_OPTIMAL_IS, ADAPTIVE_OPTIMAL_IS, TWAP_IS, HEURISTIC_IS

                regime = REGIMES.get(d, "unknown")
                r = evaluate_model_on_date(model, d, n_episodes=n_episodes, n_state_dims=n_state_dims)
                stat_test = test_vs_baseline(r.get("is_samples", []), STATIC_OPTIMAL_IS.get(d, 0.0))


                return {
                    "date": d,
                    "regime": regime,
                    "mean_is_pct": r.get("mean_IS"),
                    "std_is": r.get("std_IS"),
                    "cvar95": r.get("cvar95"),
                    "forced_liquidation_rate": r.get("forced_liquidation_rate"),
                    "action_distribution": {
                        str(k): v for k, v in r.get("action_distribution", {}).items()
                    },
                    "mean_action": r.get("mean_action"),
                    "action_entropy": r.get("action_entropy"),
                    "static_optimal_is": STATIC_OPTIMAL_IS.get(d),
                    "adaptive_optimal_is": ADAPTIVE_OPTIMAL_IS.get(d),
                    "twap_is": TWAP_IS.get(d),
                    "heuristic_is": HEURISTIC_IS.get(d),
                    "p_value": stat_test.get("p_value"),
                    "ci_lower": stat_test.get("ci_lower"),
                    "ci_upper": stat_test.get("ci_upper"),
                    "significantly_better": stat_test.get("significantly_better"),
                }

            # Execute single date evaluation in thread pool
            res = await loop.run_in_executor(None, _run_single_date, date)
            date_results.append(res)

            # Emit date_complete progress update
            dates_done = idx + 1
            rl_is = res["mean_is_pct"]
            ac_is = res["static_optimal_is"]
            improvement_pp = ac_is - rl_is  # positive value = RL has lower cost (better)
            
            await _publish({
                "type": "date_complete",
                "date": date,
                "regime": res["regime"],
                "rl_is": round(rl_is, 4),
                "ac_is": round(ac_is, 4),
                "improvement_pp": round(improvement_pp, 4),
                "dates_done": dates_done,
                "dates_total": dates_total,
            })

        duration = time.time() - start
        result = {
            "job_id": job_id,
            "status": "complete",
            "model_name": model_obj.name,
            "date_results": date_results,
            "duration_seconds": round(duration, 2),
            "created_at": datetime.utcnow().isoformat(),
        }

        await _update_status("complete", {
            "result_data": json.dumps(result),
            "completed_at": datetime.utcnow(),
            "duration_seconds": duration,
        })
        await _publish({
            "type": "complete",
            "job_id": job_id,
            "results_url": f"/api/evaluate/result/{job_id}",
            "results": result,
        })

    except asyncio.CancelledError:
        err_msg = "Cancelled by user"
        await _update_status("failed", {"error_message": err_msg})
        await _publish({"type": "error", "message": err_msg})
        raise
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"Evaluation failed: {e}\n{tb}"
        await _update_status("failed", {"error_message": err_msg[:2000]})
        await _publish({"type": "error", "message": str(e)})
    finally:
        await unregister_task(job_id)


async def run_sweep_job(ctx: dict, job_id: str, request_data: dict[str, Any]):
    """
    ARQ Task: Runs multi-dimensional parameter sweeps using research-sim.
    """
    start = time.time()
    redis_conn = ctx.get("redis")

    async def _update_status(status: str, extra: dict | None = None):
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            result = await db.execute(
                select(SweepJob).where(SweepJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = status
                if extra:
                    for k, v in extra.items():
                        setattr(job, k, v)
                await db.commit()

    async def _publish(msg: dict):
        await publish_progress(redis_conn, job_id, msg)

    # Register in job registry
    from web.services.job_registry import register_task, unregister_task
    await register_task(job_id, asyncio.current_task())

    try:
        await _update_status("running")
        await _publish({"type": "status", "status": "running"})

        runner = get_rust_runner()
        n_paths = request_data.get("n_paths", 100)

        # Runs parameter sweep config internally inside Rust
        sweep_data = await runner.run_sweep(n_paths=n_paths, job_id=job_id)

        duration = time.time() - start
        result = {
            "job_id": job_id,
            "status": "complete",
            "sweep_dimension": request_data.get("sweep_dimension"),
            "grid_values": request_data.get("grid_values", []),
            "sweep_data": sweep_data,
            "duration_seconds": round(duration, 2),
            "created_at": datetime.utcnow().isoformat(),
        }

        await _update_status("complete", {
            "result_data": json.dumps(result),
            "completed_at": datetime.utcnow(),
            "duration_seconds": duration,
        })
        await _publish({
            "type": "complete",
            "job_id": job_id,
            "results": result,
        })

    except asyncio.CancelledError:
        err_msg = "Cancelled by user"
        await _update_status("failed", {"error_message": err_msg})
        await _publish({"type": "error", "message": err_msg})
        raise
    except Exception as e:
        err_msg = f"Sweep run failed: {e}"
        await _update_status("failed", {"error_message": err_msg})
        await _publish({"type": "error", "message": err_msg})
    finally:
        await unregister_task(job_id)
