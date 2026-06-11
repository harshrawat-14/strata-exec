"""
RL Evaluation endpoints.
POST /api/evaluate              — Start an RL evaluation job
GET  /api/evaluate/result/{id} — Fetch completed evaluation results
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.models.database import EvaluationJob, UploadedModel, get_db
from web.models.schemas import (
    DateResult,
    EvaluationRequest,
    EvaluationResult,
    JobStatus,
)
from web.workers.tasks import run_evaluation_task

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])


@router.post("", response_model=JobStatus)
async def start_evaluation(
    req: EvaluationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Queue an RL model evaluation job."""
    # Verify model exists
    result = await db.execute(
        select(UploadedModel).where(UploadedModel.id == req.model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(404, f"Model {req.model_id} not found")

    job_id = str(uuid.uuid4())
    job = EvaluationJob(
        id=job_id,
        status="queued",
        model_id=req.model_id,
        dates=json.dumps(req.dates),
        n_episodes=req.n_episodes,
        compare_with=json.dumps(req.compare_with),
    )
    db.add(job)
    await db.commit()

    background_tasks.add_task(
        run_evaluation_task,
        job_id=job_id,
        request_data=req.model_dump(),
    )

    return JobStatus(
        job_id=job_id,
        status="queued",
        websocket_url=f"/ws/job/{job_id}",
    )


@router.get("/result/{job_id}", response_model=EvaluationResult)
async def get_evaluation_result(job_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch completed RL evaluation results."""
    result = await db.execute(
        select(EvaluationJob).where(EvaluationJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Evaluation job not found")

    if job.status == "failed":
        raise HTTPException(500, job.error_message or "Evaluation failed")

    if job.status != "complete":
        return EvaluationResult(
            job_id=job_id,
            status=job.status,
            created_at=job.created_at,
        )

    raw = json.loads(job.result_data or "{}")
    date_results = [DateResult(**dr) for dr in raw.get("date_results", [])]

    return EvaluationResult(
        job_id=job_id,
        status=job.status,
        model_name=raw.get("model_name"),
        date_results=date_results,
        synthetic_is=raw.get("synthetic_is"),
        duration_seconds=job.duration_seconds,
        created_at=job.created_at,
    )
