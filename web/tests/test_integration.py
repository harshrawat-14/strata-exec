"""
Integration tests for database operations and SSE progress streaming.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from sqlalchemy import select

from web.config import get_settings
from web.models.database import SimulationJob, EvaluationJob
from web.routes.progress import _get_job_status, stream_progress

settings = get_settings()


# ── Database Persistence Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_database_persistence(db_session):
    # 1. Insert simulation job
    job = SimulationJob(
        id="integration-test-job-uuid",
        status="queued",
        price_model="gbm",
        strategies="['twap', 'optimal']",
        n_paths=100,
        params="{}",
    )
    db_session.add(job)
    await db_session.commit()

    # 2. Retrieve job
    result = await db_session.execute(
        select(SimulationJob).where(SimulationJob.id == "integration-test-job-uuid")
    )
    retrieved = result.scalar_one_or_none()
    assert retrieved is not None
    assert retrieved.status == "queued"
    assert retrieved.price_model == "gbm"
    assert retrieved.n_paths == 100

    # 3. Update job status
    retrieved.status = "running"
    await db_session.commit()

    # Verify update
    result_updated = await db_session.execute(
        select(SimulationJob).where(SimulationJob.id == "integration-test-job-uuid")
    )
    retrieved_updated = result_updated.scalar_one_or_none()
    assert retrieved_updated.status == "running"


# ── SSE Progress Router Integration Tests ─────────────────────────────────────

@pytest.mark.asyncio
async def test_sse_progress_router_fallback(db_session):
    # Insert a job in the database
    job_id = "sse-test-job"
    job = SimulationJob(
        id=job_id,
        status="running",
        price_model="gbm",
        strategies="['twap']",
        n_paths=50,
        params="{}",
    )
    db_session.add(job)
    await db_session.commit()

    # Verify status helper works
    status_info = await _get_job_status(job_id)
    assert status_info["status"] == "running"

    original_redis_url = settings.redis_url
    settings.__dict__["redis_url"] = "redis://127.0.0.1:9999"

    try:
        # Call stream_progress endpoint. It returns a StreamingResponse.
        response = await stream_progress(job_id)
        assert response.media_type == "text/event-stream"
        
        # We retrieve the generator from the StreamingResponse body
        event_generator = response.body_iterator

        # We will fetch a few events from the generator in the background
        # since it runs an infinite polling loop until complete/failed.
        events = []
        
        async def collect_events():
            async for event in event_generator:
                events.append(event)
                if "complete" in event or "error" in event or len(events) >= 3:
                    break

        # Start collector task
        collector_task = asyncio.create_task(collect_events())

        # Wait a bit, then change job status to complete in the database
        await asyncio.sleep(0.5)
        async def update_db():
            # Get active session
            job.status = "complete"
            db_session.add(job)
            await db_session.commit()
        
        await update_db()
        
        # Wait for the generator to pick it up and terminate
        try:
            await asyncio.wait_for(collector_task, timeout=5.0)
        except asyncio.TimeoutError:
            collector_task.cancel()

        # Validate output events
        assert len(events) > 0
        # First event should yield initial status
        assert "running" in events[0]
        # Subsequent event should yield complete
        assert any("complete" in e for e in events)
    finally:
        settings.__dict__["redis_url"] = original_redis_url
