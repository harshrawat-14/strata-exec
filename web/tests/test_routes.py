"""
Functional tests for API routes:
  - POST /api/evaluate  (model not found → 404; model found → 200)
  - GET  /api/evaluate/result/{id}  (pending, failed, complete)
  - GET  /api/compare/{id}  (pending result)
  - POST /api/simulate  (schema validation edge cases)
  - GET  /api/jobs  (limit parameter)
  - RL evaluation endpoint with model in DB
"""
from __future__ import annotations

import json
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from web.main import app
from web.models.database import (
    AsyncSessionLocal,
    EvaluationJob,
    SimulationJob,
    UploadedModel,
)


# ── /api/evaluate POST ────────────────────────────────────────────────────────

class TestEvaluatePost:
    def test_evaluate_404_when_model_not_found(self, auth_headers):
        """POST /api/evaluate with a non-existent model_id must return 404."""
        with TestClient(app) as client:
            # dates must have min_length=1 per schema, so provide a valid date
            payload = {
                "model_id": str(uuid.uuid4()),
                "dates": ["2024-01-15"],
                "n_episodes": 5,
                "compare_with": ["optimal"],
            }
            resp = client.post("/api/evaluate", json=payload, headers=auth_headers)
            assert resp.status_code == 404
            assert "not found" in resp.json()["detail"].lower()

    async def test_evaluate_200_when_model_exists(self, auth_headers, db_session):
        """POST /api/evaluate with an existing model must queue a job and return 200."""
        # Seed a model directly in the test DB
        model_id = str(uuid.uuid4())
        model = UploadedModel(
            id=model_id,
            name="test_model",
            original_name="test_model.zip",
            stored_path="/tmp/test_model.zip",
            file_size_bytes=1024,
            is_builtin=False,
        )
        db_session.add(model)
        await db_session.commit()

        with TestClient(app) as client:
            payload = {
                "model_id": model_id,
                "dates": ["2024-01-15"],  # must be non-empty per schema
                "n_episodes": 2,
                "compare_with": ["optimal"],
            }
            resp = client.post("/api/evaluate", json=payload, headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "job_id" in data
            assert data["status"] == "queued"

    def test_evaluate_requires_auth(self):
        """POST /api/evaluate without a token must return 401."""
        with TestClient(app) as client:
            resp = client.post("/api/evaluate", json={
                "model_id": str(uuid.uuid4()),
                "dates": [],
                "n_episodes": 5,
                "compare_with": [],
            })
            assert resp.status_code == 401


# ── /api/evaluate/result GET ──────────────────────────────────────────────────

class TestEvaluateResult:
    async def test_pending_job_returns_status(self, auth_headers, db_session):
        """GET /api/evaluate/result/{id} for a running job returns partial status."""
        job_id = str(uuid.uuid4())
        job = EvaluationJob(
            id=job_id,
            status="running",
            model_id=str(uuid.uuid4()),
            dates="[]",
            n_episodes=10,
            compare_with="[]",
        )
        db_session.add(job)
        await db_session.commit()

        with TestClient(app) as client:
            resp = client.get(f"/api/evaluate/result/{job_id}", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["job_id"] == job_id
            assert data["status"] == "running"

    async def test_failed_job_returns_500(self, auth_headers, db_session):
        """GET /api/evaluate/result/{id} for a failed job must return 500."""
        job_id = str(uuid.uuid4())
        job = EvaluationJob(
            id=job_id,
            status="failed",
            model_id=str(uuid.uuid4()),
            dates="[]",
            n_episodes=5,
            compare_with="[]",
            error_message="RL binary crashed",
        )
        db_session.add(job)
        await db_session.commit()

        with TestClient(app) as client:
            resp = client.get(f"/api/evaluate/result/{job_id}", headers=auth_headers)
            assert resp.status_code == 500
            assert "RL binary crashed" in resp.json()["detail"]

    async def test_complete_job_returns_results(self, auth_headers, db_session):
        """GET /api/evaluate/result/{id} for a complete job returns full results."""
        job_id = str(uuid.uuid4())
        result_data = json.dumps({
            "model_name": "test_model",
            "date_results": [],
            "synthetic_is": -0.45,
        })
        job = EvaluationJob(
            id=job_id,
            status="complete",
            model_id=str(uuid.uuid4()),
            dates="[]",
            n_episodes=10,
            compare_with="[]",
            result_data=result_data,
            duration_seconds=12.5,
        )
        db_session.add(job)
        await db_session.commit()

        with TestClient(app) as client:
            resp = client.get(f"/api/evaluate/result/{job_id}", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "complete"
            assert data["model_name"] == "test_model"
            assert data["synthetic_is"] == pytest.approx(-0.45)
            assert data["duration_seconds"] == pytest.approx(12.5)

    def test_404_for_nonexistent_evaluation_job(self, auth_headers):
        """GET /api/evaluate/result/{unknown} must return 404."""
        with TestClient(app) as client:
            resp = client.get(f"/api/evaluate/result/{uuid.uuid4()}", headers=auth_headers)
            assert resp.status_code == 404


# ── /api/compare GET ──────────────────────────────────────────────────────────

class TestCompareResult:
    async def test_pending_sim_returns_status(self, auth_headers, db_session):
        """GET /api/compare/{id} for an in-progress job returns status without results."""
        job_id = str(uuid.uuid4())
        job = SimulationJob(
            id=job_id,
            status="running",
            price_model="gbm",
            strategies='["twap"]',
            n_paths=100,
            params="{}",
        )
        db_session.add(job)
        await db_session.commit()

        with TestClient(app) as client:
            resp = client.get(f"/api/compare/{job_id}", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["job_id"] == job_id
            assert data["status"] == "running"
            assert data.get("strategies") is None or data.get("strategies") == []

    async def test_failed_sim_returns_500(self, auth_headers, db_session):
        """GET /api/compare/{id} for a failed job must return 500."""
        job_id = str(uuid.uuid4())
        job = SimulationJob(
            id=job_id,
            status="failed",
            price_model="garch",
            strategies='["twap"]',
            n_paths=50,
            params="{}",
            error_message="Binary not found",
        )
        db_session.add(job)
        await db_session.commit()

        with TestClient(app) as client:
            resp = client.get(f"/api/compare/{job_id}", headers=auth_headers)
            assert resp.status_code == 500
            assert "Binary not found" in resp.json()["detail"]

    def test_404_for_nonexistent_simulation(self, auth_headers):
        with TestClient(app) as client:
            resp = client.get(f"/api/compare/{uuid.uuid4()}", headers=auth_headers)
            assert resp.status_code == 404


# ── /api/simulate POST — schema validation ────────────────────────────────────

class TestSimulateSchema:
    def test_missing_price_model_uses_default(self, auth_headers):
        """POST /api/simulate without price_model falls back to default 'gbm' (field has default)."""
        with TestClient(app) as client:
            resp = client.post("/api/simulate", json={
                "strategies": ["twap"],
                "n_paths": 10,
                "params": {}
            }, headers=auth_headers)
            # price_model defaults to "gbm" so request is accepted
            assert resp.status_code == 200

    def test_empty_strategies_list_returns_422(self, auth_headers):
        """POST /api/simulate with empty strategies list returns 422 (min_length=1)."""
        with TestClient(app) as client:
            resp = client.post("/api/simulate", json={
                "price_model": "gbm",
                "strategies": [],
                "n_paths": 5,
                "params": {}
            }, headers=auth_headers)
            # strategies has min_length=1 constraint
            assert resp.status_code == 422

    def test_invalid_price_model_returns_422(self, auth_headers):
        """POST /api/simulate with a price_model not matching pattern returns 422."""
        with TestClient(app) as client:
            resp = client.post("/api/simulate", json={
                "price_model": "unknown_model_xyz",
                "strategies": ["twap"],
                "n_paths": 5,
                "params": {}
            }, headers=auth_headers)
            # price_model pattern is ^(gbm|garch)$ — unknown value should fail
            assert resp.status_code == 422

    def test_invalid_n_paths_type_returns_422(self, auth_headers):
        """POST /api/simulate with non-integer n_paths must return 422."""
        with TestClient(app) as client:
            resp = client.post("/api/simulate", json={
                "price_model": "gbm",
                "strategies": ["twap"],
                "n_paths": "not_a_number",
                "params": {}
            }, headers=auth_headers)
            assert resp.status_code == 422


# ── /api/jobs GET — list and limit ────────────────────────────────────────────

class TestListJobs:
    async def test_list_jobs_returns_list(self, auth_headers, db_session):
        """GET /api/jobs must return a JSON list."""
        with TestClient(app) as client:
            resp = client.get("/api/jobs", headers=auth_headers)
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)

    async def test_limit_parameter_respected(self, auth_headers, db_session):
        """GET /api/jobs?limit=2 must return at most 2 items."""
        # Seed 5 jobs
        for i in range(5):
            job = SimulationJob(
                id=str(uuid.uuid4()),
                status="complete",
                price_model="gbm",
                strategies='["twap"]',
                n_paths=10,
                params="{}",
            )
            db_session.add(job)
        await db_session.commit()

        with TestClient(app) as client:
            resp = client.get("/api/jobs?limit=2", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) <= 2

    async def test_jobs_list_contains_expected_fields(self, auth_headers, db_session):
        """Each job in GET /api/jobs must contain required fields."""
        job = SimulationJob(
            id=str(uuid.uuid4()),
            status="queued",
            price_model="garch",
            strategies='["optimal"]',
            n_paths=50,
            params="{}",
        )
        db_session.add(job)
        await db_session.commit()

        with TestClient(app) as client:
            resp = client.get("/api/jobs", headers=auth_headers)
            assert resp.status_code == 200
            jobs = resp.json()
            # At least one job in list
            assert len(jobs) >= 1
            first = jobs[0]
            for field in ("job_id", "type", "status", "price_model", "n_paths", "created_at"):
                assert field in first, f"Missing field: {field}"


# ── /api/strategies + /api/dashboard GET ─────────────────────────────────────

class TestDashboardEndpoints:
    def test_strategies_requires_auth(self):
        """GET /api/strategies must require authentication."""
        with TestClient(app) as client:
            resp = client.get("/api/strategies")
            assert resp.status_code == 401

    def test_strategies_returns_list_when_authenticated(self, auth_headers):
        """GET /api/strategies must return a list of strategy objects."""
        with TestClient(app) as client:
            resp = client.get("/api/strategies", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) >= 1
            # Each strategy must have id, name, description fields
            first = data[0]
            assert "id" in first
            assert "name" in first
            assert "description" in first

    def test_dashboard_stats_requires_auth(self):
        """GET /api/dashboard must require authentication."""
        with TestClient(app) as client:
            resp = client.get("/api/dashboard")
            assert resp.status_code == 401

    def test_dashboard_stats_returns_expected_shape(self, auth_headers):
        """GET /api/dashboard must return the expected stats shape."""
        with TestClient(app) as client:
            resp = client.get("/api/dashboard", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            for field in ("total_simulations", "total_evaluations",
                          "available_lob_files", "available_models",
                          "available_dates", "recent_jobs"):
                assert field in data, f"Missing field: {field}"
