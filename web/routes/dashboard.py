"""
Dashboard and strategy info endpoints.
GET /api/dashboard   — Stats for the dashboard page
GET /api/strategies  — Available strategy names and descriptions
GET /api/dates       — Available LOB data dates (built-in + uploaded)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from web.config import get_settings
from web.models.database import (
    EvaluationJob,
    SimulationJob,
    UploadedFile,
    UploadedModel,
    get_db,
)

router = APIRouter(prefix="/api", tags=["dashboard"])
settings = get_settings()

STRATEGY_INFO = [
    {
        "id": "twap",
        "name": "TWAP",
        "description": "Time-Weighted Average Price — uniform slicing over the horizon",
        "color": "#f59e0b",
    },
    {
        "id": "heuristic",
        "name": "Heuristic",
        "description": "Volatility-adaptive chunking: smaller slices in high-vol regimes",
        "color": "#8b5cf6",
    },
    {
        "id": "optimal",
        "name": "Optimal (AC)",
        "description": "Almgren-Chriss closed-form optimal trajectory",
        "color": "#10b981",
    },
    {
        "id": "adaptive",
        "name": "Adaptive Optimal",
        "description": "AC with real-time recalibration at each step",
        "color": "#3b82f6",
    },
]

BUILTIN_DATES = [
    "2024-01-15", "2024-02-20", "2024-03-05", "2024-04-15",
    "2024-06-10", "2024-07-10", "2024-08-05", "2024-09-15",
    "2024-10-15", "2024-11-06", "2024-12-10",
]

DATE_REGIMES = {
    "2024-01-15": "Calm bull",
    "2024-03-05": "BTC breakout",
    "2024-06-10": "Quiet consolidation",
    "2024-08-05": "Crash — Yen unwind",
    "2024-11-06": "Post-election surge",
}


@router.get("/dashboard")
async def dashboard_stats(db: AsyncSession = Depends(get_db)):
    """Return stats for the dashboard overview cards."""
    # Total simulations
    sim_count = await db.scalar(select(func.count()).select_from(SimulationJob))
    eval_count = await db.scalar(select(func.count()).select_from(EvaluationJob))
    file_count = await db.scalar(
        select(func.count()).select_from(UploadedFile).where(UploadedFile.file_type == "lob")
    )
    model_count = await db.scalar(select(func.count()).select_from(UploadedModel))

    # Check which built-in dates have data on disk
    book_depth_dir = Path(settings.data_path) / "BookDepth"
    available_dates = []
    for date in BUILTIN_DATES:
        csv_path = book_depth_dir / f"BTCUSDT-bookDepth-{date}.csv"
        if csv_path.exists():
            available_dates.append({
                "date": date,
                "regime": DATE_REGIMES.get(date, ""),
                "source": "builtin",
            })

    # Recent jobs (last 10)
    sim_result = await db.execute(
        select(SimulationJob).order_by(SimulationJob.created_at.desc()).limit(5)
    )
    recent_sims = [
        {
            "job_id": j.id,
            "type": "simulation",
            "status": j.status,
            "label": f"{j.price_model.upper()} · {j.n_paths} paths",
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "duration_seconds": j.duration_seconds,
        }
        for j in sim_result.scalars().all()
    ]

    eval_result = await db.execute(
        select(EvaluationJob).order_by(EvaluationJob.created_at.desc()).limit(5)
    )
    recent_evals = [
        {
            "job_id": j.id,
            "type": "evaluation",
            "status": j.status,
            "label": f"RL Eval · {j.n_episodes} episodes",
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "duration_seconds": j.duration_seconds,
        }
        for j in eval_result.scalars().all()
    ]

    recent_jobs = sorted(
        recent_sims + recent_evals,
        key=lambda x: x["created_at"] or "",
        reverse=True,
    )[:10]

    return {
        "total_simulations": sim_count or 0,
        "total_evaluations": eval_count or 0,
        "available_lob_files": (file_count or 0) + len(available_dates),
        "available_models": (model_count or 0),
        "available_dates": available_dates,
        "recent_jobs": recent_jobs,
    }


@router.get("/strategies")
async def list_strategies():
    """Return available strategy configurations."""
    return STRATEGY_INFO


@router.get("/dates")
async def list_dates(db: AsyncSession = Depends(get_db)):
    """Return all available LOB data dates (built-in + uploaded)."""
    book_depth_dir = Path(settings.data_path) / "BookDepth"
    dates = []

    # Built-in dates
    for date in BUILTIN_DATES:
        csv_path = book_depth_dir / f"BTCUSDT-bookDepth-{date}.csv"
        if csv_path.exists():
            dates.append({
                "date": date,
                "regime": DATE_REGIMES.get(date, ""),
                "source": "builtin",
                "file_id": None,
            })

    # Uploaded dates
    result = await db.execute(
        select(UploadedFile)
        .where(UploadedFile.file_type == "lob")
        .order_by(UploadedFile.date_str)
    )
    for f in result.scalars().all():
        if f.date_str and f.date_str not in [d["date"] for d in dates]:
            dates.append({
                "date": f.date_str,
                "regime": "uploaded",
                "source": "uploaded",
                "file_id": f.id,
            })

    return sorted(dates, key=lambda x: x["date"])
