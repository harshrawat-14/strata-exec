"""
FastAPI application entrypoint.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web.config import get_settings
from web.models.database import init_db, AsyncSessionLocal, UploadedModel
from web.routes import dashboard, evaluate, simulate, sweep, upload, websocket, progress, auth

settings = get_settings()

# ── Structured Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, seed admin and built-in models, init ARQ pool. Shutdown: close pool."""
    logger.info("StrataExec starting up (version %s)", settings.app_version)
    await init_db()
    await _seed_default_user()
    await _seed_builtin_models()
    
    # Parse Redis settings and create arq_pool
    from arq import create_pool
    from arq.connections import RedisSettings
    
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

    try:
        redis_settings = RedisSettings(host=redis_host, port=redis_port)
        app.state.arq_pool = await create_pool(redis_settings)
        logger.info("ARQ pool connected to Redis at %s:%d", redis_host, redis_port)
    except Exception as e:
        logger.warning("Failed to connect to Redis ARQ pool: %s — running in local fallback mode", e)
        app.state.arq_pool = None

    yield
    
    # Shutdown
    logger.info("StrataExec shutting down")
    if app.state.arq_pool:
        await app.state.arq_pool.aclose()  # use aclose() — close() is deprecated since arq 5.0.1


async def _seed_default_user():
    """Seed the default admin user if no users exist."""
    from sqlalchemy import select
    from web.models.database import User
    from web.services.auth import get_password_hash

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).limit(1))
        user = result.scalar_one_or_none()
        if not user:
            admin_user = User(
                email="admin@strataexec.com",
                hashed_password=get_password_hash("strataexec"),
                is_active=True,
            )
            db.add(admin_user)
            await db.commit()


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
        model_paths = []
        for zip_path in sorted(models_dir.glob("*.zip")):
            model_paths.append((zip_path.stem, zip_path))
        for sub_dir in sorted(models_dir.iterdir()):
            if sub_dir.is_dir():
                best_model = sub_dir / "best_model.zip"
                if best_model.exists():
                    model_paths.append((sub_dir.name, best_model))

        for name, file_path in model_paths:
            # Check if already seeded
            result = await db.execute(
                select(UploadedModel).where(UploadedModel.name == name, UploadedModel.is_builtin == True)
            )
            existing = result.scalar_one_or_none()
            if existing:
                continue

            record = UploadedModel(
                name=name,
                original_name=file_path.name,
                stored_path=str(file_path),
                file_size_bytes=file_path.stat().st_size,
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

    from web.demo_middleware import DemoModeMiddleware
    app.add_middleware(DemoModeMiddleware)

    # ── CORS ──────────────────────────────────────────────────────────────────
    # When origins is ["*"] credentials must be False (browser CORS spec).
    # JWT is carried in the Authorization header so cookies are not needed.
    _allow_credentials = "*" not in settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(auth.router)
    app.include_router(dashboard.router)
    app.include_router(simulate.router)
    app.include_router(evaluate.router)
    app.include_router(sweep.router)
    app.include_router(upload.router)
    app.include_router(websocket.router)
    app.include_router(progress.router)

    from web.routes.demo import router as demo_router
    app.include_router(demo_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": settings.app_version}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
