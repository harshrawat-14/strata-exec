"""
Pytest configuration and shared fixtures for StrataExec Python testing.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# ── Setup test environment variables BEFORE importing settings ────────────────
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_strataexec_db.sqlite"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["DEBUG"] = "False"

from web.config import get_settings
from web.models.database import Base

settings = get_settings()

# ── Shared engine (session-scoped) ────────────────────────────────────────────

_test_engine = None


def _get_test_engine():
    global _test_engine
    if _test_engine is None:
        _test_engine = create_async_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
        )
    return _test_engine


# ── Session-scoped DB setup / teardown ───────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def setup_test_db():
    """Create the schema once per test session, drop it on teardown."""
    db_file = Path("test_strataexec_db.sqlite")
    if db_file.exists():
        db_file.unlink()

    engine = _get_test_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()
    if db_file.exists():
        db_file.unlink()


# ── Per-test transactional session (rolls back after each test) ───────────────

@pytest_asyncio.fixture
async def db_session(setup_test_db):
    """
    Provide a transactional AsyncSession for each test.
    Changes are rolled back on teardown so tests are fully isolated.
    """
    engine = setup_test_db
    AsyncSessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with AsyncSessionLocal() as session:
        yield session
        await session.rollback()


# ── Mock Redis ────────────────────────────────────────────────────────────────

class MockRedis:
    """Minimal in-memory mock Redis for unit tests."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self.published: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str):
        self.published.append((channel, message))

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self._store[key] = value

    async def aclose(self):
        pass


@pytest.fixture
def mock_redis():
    return MockRedis()


# ── Auth helpers ──────────────────────────────────────────────────────────────

@pytest.fixture
def auth_headers():
    """Return Bearer token headers for the seeded admin user."""
    from web.services.auth import create_access_token
    token = create_access_token(data={"sub": "admin@strataexec.com"})
    return {"Authorization": f"Bearer {token}"}
