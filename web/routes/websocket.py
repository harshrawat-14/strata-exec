"""
WebSocket endpoint for real-time job progress.
WS /ws/job/{job_id}

Subscribes to Redis pub/sub channel "job:{job_id}" and streams
messages to the connected WebSocket client.
Falls back to polling if Redis is unavailable.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from web.config import get_settings
from web.models.database import AsyncSessionLocal, SimulationJob, EvaluationJob, SweepJob

router = APIRouter(tags=["websocket"])
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


@router.websocket("/ws/job/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket for real-time progress.

    Message format:
      {"type": "progress", "completed": 42, "total": 500}
      {"type": "status", "status": "running"}
      {"type": "complete", "job_id": "...", "results_url": "..."}
      {"type": "error", "message": "..."}
    """
    await websocket.accept()

    # Try Redis pub/sub first
    try:
        import redis.asyncio as aioredis
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"job:{job_id}")

        try:
            while True:
                # Check for incoming message from Redis (non-blocking with timeout)
                msg = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0,
                )
                if msg and msg["type"] == "message":
                    data = json.loads(msg["data"])
                    await websocket.send_json(data)

                    # Stop streaming once job is terminal
                    if data.get("type") in ("complete", "error"):
                        break

                # Keepalive ping
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                except asyncio.TimeoutError:
                    pass

        except WebSocketDisconnect:
            pass
        finally:
            await pubsub.unsubscribe(f"job:{job_id}")
            await redis.aclose()

    except Exception:
        # Redis unavailable — fall back to polling
        try:
            while True:
                status_info = await _get_job_status(job_id)
                if status_info is None:
                    await websocket.send_json({"type": "error", "message": "Job not found"})
                    break

                status = status_info["status"]
                await websocket.send_json({"type": "status", "status": status})

                if status == "complete":
                    await websocket.send_json({
                        "type": "complete",
                        "job_id": job_id,
                        "results_url": f"/api/compare/{job_id}",
                    })
                    break
                elif status == "failed":
                    await websocket.send_json({
                        "type": "error",
                        "message": status_info.get("error") or "Job failed",
                    })
                    break

                await asyncio.sleep(1.0)

        except WebSocketDisconnect:
            pass
