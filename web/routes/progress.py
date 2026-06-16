"""
Server-Sent Events (SSE) progress stream router.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.config import get_settings
from web.models.database import AsyncSessionLocal, SimulationJob, EvaluationJob, SweepJob, get_db
from web.services.auth import get_current_user

router = APIRouter(prefix="/api/jobs", tags=["progress"], dependencies=[Depends(get_current_user)])
settings = get_settings()


async def _get_job_status(job_id: str) -> dict | None:
    """Check current job status across all job tables."""
    async with AsyncSessionLocal() as db:
        for Model in (SimulationJob, EvaluationJob, SweepJob):
            result = await db.execute(
                select(Model).where(Model.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                return {
                    "status": job.status,
                    "error": getattr(job, "error_message", None),
                }
    return None


@router.get("/{job_id}/progress")
async def stream_progress(job_id: str):
    """
    Stream job progress using Server-Sent Events (SSE).
    Subscribes to Redis pub/sub channel 'job:{job_id}'.
    Falls back to database polling if Redis is down.
    """
    # Check if job exists first
    initial_status = await _get_job_status(job_id)
    if not initial_status:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        # If job is already complete or failed, yield status and terminate
        if initial_status["status"] in ("complete", "failed"):
            yield f"data: {json.dumps({'type': 'status', 'status': initial_status['status']})}\n\n"
            return

        try:
            import redis.asyncio as aioredis
            redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            pubsub = redis.pubsub()
            await pubsub.subscribe(f"job:{job_id}")

            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = message["data"]
                        yield f"data: {data}\n\n"

                        parsed = json.loads(data)
                        if parsed.get("type") in ("complete", "error") or parsed.get("status") in ("complete", "failed"):
                            break
            finally:
                await pubsub.unsubscribe(f"job:{job_id}")
                await redis.aclose()

        except Exception:
            # Fallback: Poll database status every 1.5 seconds if Redis is down
            last_status = None
            while True:
                status_info = await _get_job_status(job_id)
                if not status_info:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                    break

                current_status = status_info["status"]
                if current_status != last_status:
                    yield f"data: {json.dumps({'type': 'status', 'status': current_status})}\n\n"
                    last_status = current_status

                if current_status == "complete":
                    yield f"data: {json.dumps({'type': 'complete', 'job_id': job_id, 'results_url': f'/api/compare/{job_id}'})}\n\n"
                    break
                elif current_status == "failed":
                    yield f"data: {json.dumps({'type': 'error', 'message': status_info.get('error') or 'Job failed'})}\n\n"
                    break

                await asyncio.sleep(1.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
            "Connection": "keep-alive",
        }
    )


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a running job.
    1. Mark job as failed/cancelled in the database.
    2. Cancel the task locally (for same-process mode).
    3. Broadcast abort message to Redis channel 'job_abort' for background workers.
    """
    found = False
    for Model in (SimulationJob, EvaluationJob, SweepJob):
        result = await db.execute(
            select(Model).where(Model.id == job_id)
        )
        job = result.scalar_one_or_none()
        if job:
            found = True
            if job.status in ("queued", "running"):
                job.status = "failed"
                job.error_message = "Cancelled by user"
                await db.commit()
            break

    if not found:
        raise HTTPException(404, "Job not found")

    # Local cancellation (for fallback / same process execution)
    from web.services.job_registry import abort_job_local
    await abort_job_local(job_id)

    # Publish to Redis channel 'job_abort'
    try:
        import redis.asyncio as aioredis
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        # Notify workers to abort the job
        await redis.publish("job_abort", json.dumps({"job_id": job_id}))
        # Notify WebSocket/SSE subscribers that the job failed/cancelled
        await redis.publish(
            f"job:{job_id}",
            json.dumps({"type": "error", "message": "Cancelled by user"})
        )
        await redis.aclose()
    except Exception:
        pass

    return {"status": "ok", "message": "Job cancellation triggered"}


@router.get("/{job_id}/state")
async def get_job_state(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Return current job state from Redis.
    Used by frontend on reconnection to restore progress.
    """
    # Try Redis first
    try:
        import redis.asyncio as aioredis
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        raw = await redis.get(f"job:progress:{job_id}")
        await redis.aclose()
        if raw:
            return json.loads(raw)
    except Exception:
        pass

    # Fall back to database
    for Model in (SimulationJob, EvaluationJob, SweepJob):
        result = await db.execute(
            select(Model).where(Model.id == job_id)
        )
        job = result.scalar_one_or_none()
        if job:
            progress_val = 100 if job.status == "complete" else 0
            res_data = None
            if job.status == "complete" and getattr(job, "result_data", None):
                try:
                    res_data = json.loads(job.result_data)
                except Exception:
                    pass
            return {
                "job_id": job_id,
                "status": job.status,
                "progress": progress_val,
                "results": res_data,
                "error": getattr(job, "error_message", None),
            }

    raise HTTPException(status_code=404, detail="Job not found")
