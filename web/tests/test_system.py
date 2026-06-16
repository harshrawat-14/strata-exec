"""
System integration tests for FastAPI endpoints.
"""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from web.main import app
from web.models.database import SimulationJob, AsyncSessionLocal


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_list_jobs_endpoint(auth_headers):
    with TestClient(app) as client:
        response = client.get("/api/jobs", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_simulate_and_cancel_flow(auth_headers):
    # We use TestClient as a context manager to trigger lifespan events (mocked or database connection)
    with TestClient(app) as client:
        # 1. Trigger Simulation POST request
        payload = {
            "price_model": "gbm",
            "strategies": ["twap", "optimal"],
            "n_paths": 10,
            "params": {
                "sigma": 0.02,
                "eta": 0.001,
                "lambda": 0.0001,
                "total_notional": 1000000.0,
                "horizon_steps": 100
            }
        }
        response = client.post("/api/simulate", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        job_id = data.get("job_id")
        assert job_id is not None
        assert data["status"] == "queued"

        # Verify job is in database as queued
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SimulationJob).where(SimulationJob.id == job_id)
            )
            job_in_db = result.scalar_one_or_none()
            assert job_in_db is not None
            assert job_in_db.status == "queued"

        # 2. Trigger Cancel POST request
        cancel_response = client.post(f"/api/jobs/{job_id}/cancel", headers=auth_headers)
        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "ok"

        # Verify job is marked as failed/cancelled in database
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SimulationJob).where(SimulationJob.id == job_id)
            )
            job_cancelled = result.scalar_one_or_none()
            assert job_cancelled is not None
            assert job_cancelled.status == "failed"
            assert job_cancelled.error_message == "Cancelled by user"
