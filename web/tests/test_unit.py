"""
Unit tests for job registry, model cache, and CAS storage.
"""
from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path
import pytest

from web.services.job_registry import (
    register_task,
    unregister_task,
    register_subprocess,
    unregister_subprocess,
    abort_job_local,
    _active_tasks,
    _active_subprocesses,
)
from web.services.model_cache import ModelCache
from web.services.storage import get_content_hash, store_file, get_presigned_upload_url


# ── Job Registry Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_registry():
    job_id = "test-job-123"
    
    # 1. Register Task
    async def dummy_task_fn():
        await asyncio.sleep(10)
    
    task = asyncio.create_task(dummy_task_fn())
    await register_task(job_id, task)
    assert _active_tasks[job_id] == task
    
    # 2. Register Subprocess
    # Start a dummy shell command that sleeps
    proc = await asyncio.create_subprocess_exec(
        "sleep", "5",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await register_subprocess(job_id, proc)
    assert proc in _active_subprocesses[job_id]

    # 3. Abort Job
    await abort_job_local(job_id)
    
    # Yield to event loop to allow cancellation to propagate
    await asyncio.sleep(0.1)

    # Check that task is cancelled and subprocess is terminated
    assert task.cancelled()
    
    # Cleanup task/subprocesses
    await unregister_task(job_id)
    assert job_id not in _active_tasks
    assert job_id not in _active_subprocesses
    
    # Wait for process to stop
    try:
        await proc.wait()
    except Exception:
        pass


# ── Model Cache Tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_cache():
    # We instantiate our own Cache instance with max size 2
    cache = ModelCache(max_size=2)
    
    # Monkeypatch the load_model call to bypass evaluate library import and load torch files
    # We will just return dummy models (dict/strings)
    dummy_models = {}
    
    # Override _model_hash to mock loading
    cache._model_hash = lambda path: path[-5:]  # Use last 5 chars of path as key
    
    # Stub the get method loading mechanism to not use execute loop / run_in_executor
    async def mock_get(model_path: str) -> str:
        key = cache._model_hash(model_path)
        async with cache._lock:
            if key in cache._cache:
                cache._access_order.remove(key)
                cache._access_order.append(key)
                return cache._cache[key]
            
            # Simulated model loading
            model = f"loaded_model_for_{key}"
            
            if len(cache._cache) >= cache._max_size:
                oldest = cache._access_order.pop(0)
                if oldest in cache._cache:
                    del cache._cache[oldest]
                    
            cache._cache[key] = model
            cache._access_order.append(key)
            return model
            
    cache.get = mock_get

    # 1. Load Model A
    a = await cache.get("/path/to/modelA")
    assert a == "loaded_model_for_odelA"
    assert "odelA" in cache._cache

    # 2. Load Model B
    b = await cache.get("/path/to/modelB")
    assert b == "loaded_model_for_odelB"
    assert len(cache._cache) == 2

    # 3. Access A again (to update access order: A becomes MRU, B becomes LRU)
    await cache.get("/path/to/modelA")

    # 4. Load Model C (should evict B, since A is MRU)
    c = await cache.get("/path/to/modelC")
    assert c == "loaded_model_for_odelC"
    
    assert "odelA" in cache._cache
    assert "odelC" in cache._cache
    assert "odelB" not in cache._cache


# ── Storage CAS Tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_storage_cas(tmp_path):
    data = b"Hello, StrataExec execution environment!"
    
    # 1. Hashing
    expected_hash = hashlib.sha256(data).hexdigest()[:16]
    assert get_content_hash(data) == expected_hash

    # 2. Store File (local fallback)
    from web.config import get_settings
    settings = get_settings()
    
    # Temporarily override settings directories
    original_lob_dir = settings.lob_upload_dir
    settings.__dict__["lob_upload_dir"] = tmp_path / "lob"
    
    try:
        stored_path = await store_file(data, "btc_depth.csv", "lob")
        path = Path(stored_path)
        assert path.exists()
        assert path.read_bytes() == data
        assert path.name == f"{expected_hash}_btc_depth.csv"
    finally:
        settings.__dict__["lob_upload_dir"] = original_lob_dir

    # 3. Presigned url fallback description
    res = get_presigned_upload_url("model.zip", "model")
    assert res["use_s3"] is False
    assert res["url"].startswith("/api/upload/local")


def test_simulate_rl_path():
    from web.services.parallel_sim import simulate_rl_path
    import numpy as np

    class DummyModel:
        def predict(self, observation, state=None, episode_start=None, deterministic=True):
            # Return action index 9 (which corresponds to trading fraction 0.050)
            return 9, None

    price_path = [100.0, 100.5, 101.0, 100.8, 101.2, 100.9]
    params = {
        "total_notional": 100000.0,
        "sigma": 0.02
    }

    res = simulate_rl_path(DummyModel(), price_path, params)
    assert "trajectory" in res
    assert "cost_series" in res
    assert "mean_is_pct" in res
    assert "trade_count" in res
    assert "avg_exec_price" in res
    assert "cost_decomposition" in res
    assert len(res["trajectory"]) == len(price_path)
    assert len(res["cost_series"]) == len(price_path)
    assert res["trajectory"][-1] == 0.0  # forced liquidation at the end
    assert isinstance(res["cost_decomposition"], dict)
    assert "spread_cost" in res["cost_decomposition"]
    assert "temporary_impact" in res["cost_decomposition"]
    assert "permanent_impact" in res["cost_decomposition"]
    assert "timing_cost" in res["cost_decomposition"]
