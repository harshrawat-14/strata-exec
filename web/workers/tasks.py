"""
Background worker functions for simulation and RL evaluation.
These run in asyncio tasks (or can be wrapped in Celery for multi-process scaling).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import redis.asyncio as aioredis

from web.config import get_settings
from web.models.database import AsyncSessionLocal, SimulationJob, EvaluationJob, SweepJob
from web.services.rust_runner import get_rust_runner, RustRunnerError

settings = get_settings()


async def get_redis() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def publish_progress(redis: aioredis.Redis, job_id: str, msg: dict):
    """Publish a WebSocket message to the job's Redis channel."""
    await redis.publish(f"job:{job_id}", json.dumps(msg))


async def run_simulation_task(
    job_id: str,
    request_data: dict[str, Any],
    lob_path: str | None = None,
    agg_path: str | None = None,
):
    """
    Async task: run research-sim, stream progress via Redis, save results to DB.
    """
    start = time.time()

    try:
        redis = await get_redis()
        await redis.ping()
    except Exception as e:
        redis = None  # Redis optional — gracefully degrade

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
        if redis:
            await publish_progress(redis, job_id, msg)

    await _update_status("running")
    await _publish({"type": "status", "status": "running"})

    try:
        runner = get_rust_runner()
        n_paths = request_data.get("n_paths", 100)
        model = request_data.get("price_model", "gbm")
        
        paths_done = 0

        def on_progress(completed: int, total: int):
            nonlocal paths_done
            paths_done = completed
            asyncio.get_event_loop().call_soon_threadsafe(
                asyncio.ensure_future,
                _publish({"type": "progress", "completed": completed, "total": total})
            )

        result = await runner.run_simulation(
            n_paths=n_paths,
            model=model,
            lob_path=lob_path,
            agg_path=agg_path,
            on_progress=on_progress,
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
        })

    except RustRunnerError as e:
        err_msg = str(e)
        await _update_status("failed", {"error_message": err_msg})
        await _publish({"type": "error", "message": err_msg})
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"Unexpected error: {e}\n{tb}"
        await _update_status("failed", {"error_message": err_msg[:2000]})
        await _publish({"type": "error", "message": str(e)})
    finally:
        if redis:
            await redis.aclose()


async def run_evaluation_task(job_id: str, request_data: dict[str, Any]):
    """
    Async task: run RL evaluation via rl/evaluate.py functions,
    stream per-date progress via Redis, save results to DB.
    """
    start = time.time()

    try:
        redis = await get_redis()
        await redis.ping()
    except Exception:
        redis = None

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
        if redis:
            await publish_progress(redis, job_id, msg)

    await _update_status("running")
    await _publish({"type": "status", "status": "running"})

    try:
        model_id = request_data["model_id"]
        dates = request_data.get("dates", [])
        n_episodes = request_data.get("n_episodes", 50)

        # Resolve model path
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

        # Run evaluation in a thread pool to avoid blocking the event loop
        # (PyTorch + SB3 evaluation is CPU-bound)
        loop = asyncio.get_event_loop()

        def _run_sync():
            # Add project root to sys.path so rl/evaluate.py can import rl/environment.py
            project_root = str(Path(settings.rust_binary_path).parent.parent)
            rl_dir = str(Path(project_root) / "rl")
            for p in [project_root, rl_dir]:
                if p not in sys.path:
                    sys.path.insert(0, p)

            from evaluate import load_model, evaluate_model_on_date, STATIC_OPTIMAL_IS, ADAPTIVE_OPTIMAL_IS, TWAP_IS, HEURISTIC_IS

            REGIMES = {
                "2024-01-15": "calm bull",
                "2024-03-05": "BTC breakout",
                "2024-06-10": "quiet consolidation",
                "2024-08-05": "crash - Yen unwind",
                "2024-11-06": "post-election surge",
            }

            model = load_model(model_path)
            date_results = []

            for date in dates:
                regime = REGIMES.get(date, "unknown")
                r = evaluate_model_on_date(model, date, n_episodes=n_episodes)

                date_results.append({
                    "date": date,
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
                    "static_optimal_is": STATIC_OPTIMAL_IS.get(date),
                    "adaptive_optimal_is": ADAPTIVE_OPTIMAL_IS.get(date),
                    "twap_is": TWAP_IS.get(date),
                    "heuristic_is": HEURISTIC_IS.get(date),
                })

            return date_results

        date_results = await loop.run_in_executor(None, _run_sync)

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
        })

    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"{e}\n{tb}"
        await _update_status("failed", {"error_message": err_msg[:2000]})
        await _publish({"type": "error", "message": str(e)})
    finally:
        if redis:
            await redis.aclose()


async def run_sweep_task(job_id: str, request_data: dict[str, Any]):
    """Run parameter sweep via research-sim --experiments."""
    start = time.time()

    try:
        redis = await get_redis()
        await redis.ping()
    except Exception:
        redis = None

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
        if redis:
            await publish_progress(redis, job_id, msg)

    await _update_status("running")
    await _publish({"type": "status", "status": "running"})

    try:
        runner = get_rust_runner()
        n_paths = request_data.get("n_paths", 100)

        sweep_data = await runner.run_sweep(n_paths=n_paths)

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
        await _publish({"type": "complete", "job_id": job_id})

    except Exception as e:
        err_msg = str(e)
        await _update_status("failed", {"error_message": err_msg})
        await _publish({"type": "error", "message": err_msg})
    finally:
        if redis:
            await redis.aclose()
