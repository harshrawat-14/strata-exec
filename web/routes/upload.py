"""
File upload endpoints.
POST /api/upload/lob         — BookDepth CSV
POST /api/upload/agg-trades  — AggTrades CSV
POST /api/upload/rl-model    — RL model .zip
GET  /api/upload/files       — List uploaded LOB/agg files
GET  /api/upload/models      — List uploaded + built-in models
GET  /api/upload/lob-preview/{file_id}  — Depth + spread series for preview chart
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.config import get_settings
from web.models.database import UploadedFile, UploadedModel, get_db
from web.models.schemas import UploadedFileInfo, UploadedModelInfo, LOBPreview
from web.services.lob_preview import (
    FileValidationError,
    extract_date_from_filename,
    get_lob_depth_preview,
    get_spread_series,
    validate_agg_trades_csv,
    validate_lob_csv,
)

router = APIRouter(prefix="/api/upload", tags=["upload"])
settings = get_settings()

MAX_UPLOAD_MB = 50
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024


# ── LOB upload ────────────────────────────────────────────────────────────────

@router.post("/lob", response_model=UploadedFileInfo)
async def upload_lob(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a Binance BookDepth CSV file."""
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_MB}MB)")

    try:
        n_rows, preview = validate_lob_csv(raw)
    except FileValidationError as e:
        raise HTTPException(422, str(e))

    # Store file
    file_id = str(uuid.uuid4())
    dest = settings.lob_upload_dir / f"{file_id}.csv"
    dest.write_bytes(raw)

    date_str = extract_date_from_filename(file.filename or "")

    record = UploadedFile(
        id=file_id,
        file_type="lob",
        original_name=file.filename or "upload.csv",
        stored_path=str(dest),
        date_str=date_str,
        n_rows=n_rows,
        preview_data=json.dumps(preview),
        file_size_bytes=len(raw),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return UploadedFileInfo(
        file_id=record.id,
        file_type="lob",
        original_name=record.original_name,
        date_str=record.date_str,
        n_rows=record.n_rows,
        file_size_bytes=record.file_size_bytes,
        preview=LOBPreview(**preview) if preview else None,
        created_at=record.created_at,
    )


# ── AggTrades upload ──────────────────────────────────────────────────────────

@router.post("/agg-trades", response_model=UploadedFileInfo)
async def upload_agg_trades(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a Binance AggTrades CSV file."""
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_MB}MB)")

    try:
        n_rows, preview = validate_agg_trades_csv(raw)
    except FileValidationError as e:
        raise HTTPException(422, str(e))

    file_id = str(uuid.uuid4())
    dest = settings.agg_upload_dir / f"{file_id}.csv"
    dest.write_bytes(raw)

    date_str = extract_date_from_filename(file.filename or "")

    record = UploadedFile(
        id=file_id,
        file_type="agg_trades",
        original_name=file.filename or "upload.csv",
        stored_path=str(dest),
        date_str=date_str,
        n_rows=n_rows,
        preview_data=json.dumps(preview),
        file_size_bytes=len(raw),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return UploadedFileInfo(
        file_id=record.id,
        file_type="agg_trades",
        original_name=record.original_name,
        date_str=record.date_str,
        n_rows=record.n_rows,
        file_size_bytes=record.file_size_bytes,
        preview=None,
        created_at=record.created_at,
    )


# ── RL model upload ───────────────────────────────────────────────────────────

@router.post("/rl-model", response_model=UploadedModelInfo)
async def upload_rl_model(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a trained RL model .zip file (SB3/RecurrentPPO format)."""
    fname = file.filename or "model.zip"
    if not fname.endswith(".zip"):
        raise HTTPException(422, "Model must be a .zip file (SB3 format)")

    raw = await file.read()
    if len(raw) > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(400, "Model file too large (max 100MB)")

    # Validate it's a zip with policy.pth
    import zipfile
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(raw)) as zf:
            names = zf.namelist()
            if "policy.pth" not in names:
                raise HTTPException(422, "Not a valid SB3 model: missing 'policy.pth' in zip")
    except zipfile.BadZipFile:
        raise HTTPException(422, "Not a valid zip file")

    model_id = str(uuid.uuid4())
    dest = settings.model_upload_dir / f"{model_id}.zip"
    dest.write_bytes(raw)

    # Use filename without extension as display name
    display_name = Path(fname).stem

    record = UploadedModel(
        id=model_id,
        name=display_name,
        original_name=fname,
        stored_path=str(dest),
        file_size_bytes=len(raw),
        is_builtin=False,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return UploadedModelInfo(
        model_id=record.id,
        name=record.name,
        original_name=record.original_name,
        file_size_bytes=record.file_size_bytes,
        is_builtin=record.is_builtin,
        created_at=record.created_at,
    )


# ── List endpoints ────────────────────────────────────────────────────────────

@router.get("/files", response_model=list[UploadedFileInfo])
async def list_files(
    file_type: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List all uploaded LOB/agg files."""
    q = select(UploadedFile).order_by(UploadedFile.created_at.desc())
    if file_type:
        q = q.where(UploadedFile.file_type == file_type)
    result = await db.execute(q)
    files = result.scalars().all()

    out = []
    for f in files:
        preview = None
        if f.preview_data and f.file_type == "lob":
            try:
                preview = LOBPreview(**json.loads(f.preview_data))
            except Exception:
                pass
        out.append(UploadedFileInfo(
            file_id=f.id,
            file_type=f.file_type,
            original_name=f.original_name,
            date_str=f.date_str,
            n_rows=f.n_rows,
            file_size_bytes=f.file_size_bytes,
            preview=preview,
            created_at=f.created_at,
        ))
    return out


@router.get("/models", response_model=list[UploadedModelInfo])
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all uploaded + built-in RL models."""
    result = await db.execute(
        select(UploadedModel).order_by(UploadedModel.is_builtin.desc(), UploadedModel.created_at.desc())
    )
    models = result.scalars().all()
    return [
        UploadedModelInfo(
            model_id=m.id,
            name=m.name,
            original_name=m.original_name,
            file_size_bytes=m.file_size_bytes,
            is_builtin=m.is_builtin,
            created_at=m.created_at,
        )
        for m in models
    ]


@router.get("/lob-preview/{file_id}")
async def lob_depth_preview(file_id: str, db: AsyncSession = Depends(get_db)):
    """Return order book depth snapshots and spread series for a LOB file."""
    result = await db.execute(
        select(UploadedFile).where(UploadedFile.id == file_id)
    )
    f = result.scalar_one_or_none()
    if not f:
        raise HTTPException(404, "File not found")
    if f.file_type != "lob":
        raise HTTPException(400, "Not a LOB file")

    depth = get_lob_depth_preview(f.stored_path, n_snapshots=50)
    spread = get_spread_series(f.stored_path, every_n=10)

    return {"depth_snapshots": depth, "spread_series": spread}
