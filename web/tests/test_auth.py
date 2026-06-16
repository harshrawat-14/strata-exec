from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web.main import app
from web.services.auth import get_password_hash


def test_auth_login_success():
    """Verify that logging in with seeded default credentials succeeds."""
    with TestClient(app) as client:
        payload = {
            "email": "admin@strataexec.com",
            "password": "strataexec"
        }
        resp = client.post("/api/auth/login", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"


def test_auth_login_invalid_credentials():
    """Verify that logging in with incorrect credentials fails with 401."""
    with TestClient(app) as client:
        payload = {
            "email": "admin@strataexec.com",
            "password": "wrongpassword"
        }
        resp = client.post("/api/auth/login", json=payload)
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Incorrect email or password"


def test_auth_me_protected_route(auth_headers):
    """Verify /api/auth/me is protected and returns user details when authorized."""
    with TestClient(app) as client:
        # Unauthorized check
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

        # Authorized check
        resp = client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "admin@strataexec.com"
        assert data["is_active"] is True
        assert "id" in data


def test_protected_routes_require_authentication():
    """Verify that protected route returns 401 when no token is provided."""
    with TestClient(app) as client:
        resp = client.get("/api/jobs")
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_token_query_parameter_fallback(auth_headers, db_session):
    """Verify that token query parameter fallback allows accessing job state endpoint (crucial for EventSource)."""
    from web.models.database import SimulationJob
    import redis.asyncio as aioredis_mod
    from unittest.mock import AsyncMock, patch

    # Seed a job so we don't get 404
    job_id = "test-query-token-job"
    job = SimulationJob(
        id=job_id,
        status="complete",
        price_model="gbm",
        strategies="[]",
        n_paths=100,
        params="{}",
    )
    db_session.add(job)
    await db_session.commit()

    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.aclose = AsyncMock()

    token = auth_headers["Authorization"].split(" ")[1]

    with patch.object(aioredis_mod, "from_url", return_value=mock_redis):
        with TestClient(app) as client:
            # Query parameter authentication check
            resp = client.get(f"/api/jobs/{job_id}/state?token={token}")
            assert resp.status_code == 200
            assert resp.json()["job_id"] == job_id


def test_auth_login_invalid_email_format():
    """Verify that logging in with an invalid email format returns 400 Bad Request."""
    with TestClient(app) as client:
        invalid_emails = ["invalidemail", "invalid@email", "@domain.com", "user@.com", "user@domain."]
        for email in invalid_emails:
            payload = {
                "email": email,
                "password": "somepassword"
            }
            resp = client.post("/api/auth/login", json=payload)
            assert resp.status_code == 400
            assert resp.json()["detail"] == "Invalid email format"


def test_auth_login_auto_registration_success():
    """Verify that logging in with a new valid email format registers the user and logs them in."""
    import uuid
    new_email = f"user-{uuid.uuid4()}@strataexec.com"
    with TestClient(app) as client:
        payload = {
            "email": new_email,
            "password": "newuserpassword"
        }
        # First login: creates the user and returns 200 with access token
        resp = client.post("/api/auth/login", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        
        # Second login: verify with correct password
        resp_again = client.post("/api/auth/login", json=payload)
        assert resp_again.status_code == 200

        # Login with incorrect password for the newly created user
        payload_wrong = {
            "email": new_email,
            "password": "wrongpassword"
        }
        resp_wrong = client.post("/api/auth/login", json=payload_wrong)
        assert resp_wrong.status_code == 401
        assert resp_wrong.json()["detail"] == "Incorrect email or password"

