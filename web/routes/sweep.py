"""
Parameter sweep endpoint.
POST /api/sweep             — Start a parameter sweep job
GET  /api/sweep/{job_id}   — Fetch completed sweep results
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.models.database import SweepJob, get_db
from web.models.schemas import JobStatus, SweepRequest, SweepResult
from web.workers.tasks import run_sweep_task

router = APIRouter(prefix="/api/sweep", tags=["sweep"])


@router.post("", response_model=JobStatus)
async def start_sweep(
    req: SweepRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Queue a parameter sweep job."""
    job_id = str(uuid.uuid4())
    job = SweepJob(
        id=job_id,
        status="queued",
        sweep_dimension=req.sweep_dimension,
        grid_values=json.dumps(req.grid_values),
        n_paths=req.n_paths,
        strategies=json.dumps(req.strategies),
    )
    db.add(job)
    await db.commit()

    background_tasks.add_task(
        run_sweep_task,
        job_id=job_id,
        request_data=req.model_dump(),
    )

    return JobStatus(
        job_id=job_id,
        status="queued",
        websocket_url=f"/ws/job/{job_id}",
    )


@router.get("/{job_id}")
async def get_sweep_result(job_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch completed sweep results."""
    result = await db.execute(
        select(SweepJob).where(SweepJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Sweep job not found")

    if job.status == "failed":
        raise HTTPException(500, job.error_message or "Sweep failed")

    if job.status != "complete":
        return {
            "job_id": job_id,
            "status": job.status,
            "sweep_dimension": job.sweep_dimension,
            "grid_values": json.loads(job.grid_values),
            "cells": [],
        }

    raw = json.loads(job.result_data or "{}")
    return {
        "job_id": job_id,
        "status": "complete",
        "sweep_dimension": job.sweep_dimension,
        "grid_values": json.loads(job.grid_values),
        "sweep_data": raw.get("sweep_data", {}),
        "duration_seconds": job.duration_seconds,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }
