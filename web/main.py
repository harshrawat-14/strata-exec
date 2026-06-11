"""
FastAPI application entrypoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web.config import get_settings
from web.models.database import init_db, AsyncSessionLocal, UploadedModel
from web.routes import dashboard, evaluate, simulate, sweep, upload, websocket

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, seed built-in models. Shutdown: nothing."""
    await init_db()
    await _seed_builtin_models()
    yield


async def _seed_builtin_models():
    """
    Register pre-trained RL model zips from rl/models/ as built-in models
    so they appear in the UI model selector without needing to upload them.
    """
    from sqlalchemy import select
    models_dir = Path(settings.model_path)
    if not models_dir.exists():
        return

    async with AsyncSessionLocal() as db:
        for zip_path in sorted(models_dir.glob("*.zip")):
            name = zip_path.stem
            # Check if already seeded
            result = await db.execute(
                select(UploadedModel).where(UploadedModel.name == name, UploadedModel.is_builtin == True)
            )
            existing = result.scalar_one_or_none()
            if existing:
                continue

            record = UploadedModel(
                name=name,
                original_name=zip_path.name,
                stored_path=str(zip_path),
                file_size_bytes=zip_path.stat().st_size,
                is_builtin=True,
            )
            db.add(record)
        await db.commit()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description="StrataExec — execution strategy research platform",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(dashboard.router)
    app.include_router(simulate.router)
    app.include_router(evaluate.router)
    app.include_router(sweep.router)
    app.include_router(upload.router)
    app.include_router(websocket.router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": settings.app_version}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
