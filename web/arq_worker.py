"""
ARQ Background worker settings and lifecycle hooks.
Configures Redis connection, registers jobs, and applies sandbox resource limits.
"""
from __future__ import annotations

import os
import resource
import sys
from pathlib import Path
from arq.connections import RedisSettings

from web.config import get_settings
# Ensure project root is in python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from web.workers.tasks import run_simulation_job, run_evaluation_job, run_sweep_job

settings = get_settings()


async def abort_listener(redis_url: str):
    """Listens on Redis 'job_abort' channel and cancels jobs locally."""
    import redis.asyncio as aioredis
    import json
    import asyncio
    from web.services.job_registry import abort_job_local

    redis = aioredis.from_url(redis_url, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe("job_abort")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    job_id = data.get("job_id")
                    if job_id:
                        print(f"Received abort request for job {job_id}")
                        await abort_job_local(job_id)
                except Exception as e:
                    print(f"Error in abort listener parsing: {e}")
    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe("job_abort")
        await redis.aclose()


async def startup(ctx):
    """Lifecycle hook: Enforces CPU and virtual memory sandbox constraints."""
    print("Initializing StrataExec ARQ worker...")
    
    # Start the abort listener task
    import asyncio
    ctx["abort_listener_task"] = asyncio.create_task(
        abort_listener(settings.redis_url)
    )
    
    # ── CPU Limit (soft/hard) ─────────────────────────────────────────────
    # Max 300 seconds CPU time for the worker and any spawned processes
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        print("Sandbox resource constraint applied: CPU Time limit = 300s")
    except Exception as e:
        print(f"Warning: Failed to apply CPU sandbox limit: {e}", file=sys.stderr)

    # ── Memory Limit (soft/hard) ──────────────────────────────────────────
    # Max 4GB Virtual Memory per process to prevent runaway memory leaks / OOM
    try:
        mem_limit = 4 * 1024 * 1024 * 1024  # 4 GB
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        print("Sandbox resource constraint applied: Virtual Memory limit = 4GB")
    except Exception as e:
        print(f"Warning: Failed to apply Memory sandbox limit: {e}", file=sys.stderr)


async def shutdown(ctx):
    """Lifecycle hook: Shutdown cleanup."""
    print("Shutting down StrataExec ARQ worker...")
    import asyncio
    task = ctx.get("abort_listener_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# Parse Redis DSN details
redis_host = "localhost"
redis_port = 6379
if "redis://" in settings.redis_url:
    try:
        parts = settings.redis_url.split("redis://")[1].split(":")
        redis_host = parts[0]
        if "/" in parts[1]:
            redis_port = int(parts[1].split("/")[0])
        else:
            redis_port = int(parts[1])
    except Exception:
        pass


class WorkerSettings:
    """ARQ Worker configuration class."""
    functions = [run_simulation_job, run_evaluation_job, run_sweep_job]
    redis_settings = RedisSettings(host=redis_host, port=redis_port)
    on_startup = startup
    on_shutdown = shutdown
    
    # Max concurrent jobs to spawn
    max_jobs = 16
    
    # Enforce job timeout limit
    job_timeout = 300  # 5 minutes
    
    # Keep results in redis for 1 hour
    keep_result = 3600
