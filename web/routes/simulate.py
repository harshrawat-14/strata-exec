"""
Simulation endpoints.
POST /api/simulate          — Start a simulation job
GET  /api/compare/{job_id} — Fetch completed results
GET  /api/jobs             — List recent jobs
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from web.models.database import SimulationJob, UploadedFile, get_db
from web.models.schemas import (
    JobStatus,
    SimulationRequest,
    SimulationResult,
    StrategyResult,
)
from web.workers.tasks import run_simulation_task

router = APIRouter(prefix="/api", tags=["simulation"])


@router.post("/simulate", response_model=JobStatus)
async def start_simulation(
    req: SimulationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Queue a Monte Carlo simulation job."""
    job_id = str(uuid.uuid4())

    # Resolve file paths
    lob_path: str | None = None
    agg_path: str | None = None

    if req.lob_file_id:
        result = await db.execute(
            select(UploadedFile).where(UploadedFile.id == req.lob_file_id)
        )
        f = result.scalar_one_or_none()
        if not f:
            raise HTTPException(404, f"LOB file {req.lob_file_id} not found")
        lob_path = f.stored_path

    if req.agg_file_id:
        result = await db.execute(
            select(UploadedFile).where(UploadedFile.id == req.agg_file_id)
        )
        f = result.scalar_one_or_none()
        if not f:
            raise HTTPException(404, f"AggTrades file {req.agg_file_id} not found")
        agg_path = f.stored_path

    # Persist job record
    job = SimulationJob(
        id=job_id,
        status="queued",
        price_model=req.price_model,
        strategies=json.dumps(req.strategies),
        n_paths=req.n_paths,
        params=req.params.model_dump_json(),
        lob_file_id=req.lob_file_id,
        agg_file_id=req.agg_file_id,
    )
    db.add(job)
    await db.commit()

    # Launch background task
    request_data = req.model_dump()
    background_tasks.add_task(
        run_simulation_task,
        job_id=job_id,
        request_data=request_data,
        lob_path=lob_path,
        agg_path=agg_path,
    )

    return JobStatus(
        job_id=job_id,
        status="queued",
        websocket_url=f"/ws/job/{job_id}",
    )


@router.get("/compare/{job_id}", response_model=SimulationResult)
async def get_simulation_result(job_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch completed simulation results."""
    result = await db.execute(
        select(SimulationJob).where(SimulationJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status == "failed":
        raise HTTPException(500, job.error_message or "Simulation failed")

    if job.status != "complete":
        return SimulationResult(
            job_id=job_id,
            status=job.status,
            created_at=job.created_at,
        )

    raw = json.loads(job.result_data or "{}")
    strategies = [
        StrategyResult(
            name=s["name"],
            mean_is_pct=s.get("mean_is_pct"),
            is_variance=s.get("is_variance"),
            cvar95=s.get("cvar95"),
            ac_objective=s.get("ac_objective"),
            trajectory=s.get("trajectory", []),
            cost_decomposition=s.get("cost_decomposition", {}),
        )
        for s in raw.get("strategies", [])
    ]

    return SimulationResult(
        job_id=job_id,
        status=job.status,
        strategies=strategies,
        price_path=raw.get("price_path", []),
        params_used=raw.get("params_used", {}),
        duration_seconds=job.duration_seconds,
        created_at=job.created_at,
    )


@router.get("/jobs")
async def list_jobs(limit: int = 20, db: AsyncSession = Depends(get_db)):
    """List recent simulation + evaluation jobs."""
    result = await db.execute(
        select(SimulationJob).order_by(desc(SimulationJob.created_at)).limit(limit)
    )
    jobs = result.scalars().all()

    return [
        {
            "job_id": j.id,
            "type": "simulation",
            "status": j.status,
            "price_model": j.price_model,
            "n_paths": j.n_paths,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "duration_seconds": j.duration_seconds,
        }
        for j in jobs
    ]
