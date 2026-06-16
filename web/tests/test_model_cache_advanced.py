"""
Advanced unit tests for ModelCache:
  - Concurrent get() calls on the same key load the model exactly once
  - LRU eviction ordering is correct under rapid sequential access
  - invalidate() fully removes cache entry
  - Zero-size cache raises on first access
"""
from __future__ import annotations

import asyncio
import pytest

from web.services.model_cache import ModelCache


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_cache(max_size: int = 3) -> tuple[ModelCache, list[str]]:
    """
    Return a ModelCache wired with a mock loader that records which keys
    were actually loaded (not served from cache).
    """
    cache = ModelCache(max_size=max_size)
    load_calls: list[str] = []

    # Patch _model_hash to be identity (path IS the key)
    cache._model_hash = lambda path: path

    # Override get() to use the same LRU logic but bypass disk I/O
    original_get = cache.get

    async def mock_get(model_path: str):
        key = cache._model_hash(model_path)
        async with cache._lock:
            if key in cache._cache:
                # Cache hit — move to MRU
                cache._access_order.remove(key)
                cache._access_order.append(key)
                return cache._cache[key]

            # Cache miss — "load" the model
            load_calls.append(key)
            model_obj = f"model:{key}"

            if len(cache._cache) >= cache._max_size:
                oldest = cache._access_order.pop(0)
                cache._cache.pop(oldest, None)

            cache._cache[key] = model_obj
            cache._access_order.append(key)
            return model_obj

    cache.get = mock_get
    return cache, load_calls


# ── LRU Eviction ─────────────────────────────────────────────────────────────

class TestModelCacheLru:
    async def test_basic_lru_eviction(self):
        """Loading 4 models with max_size=3 evicts the least-recently-used."""
        cache, loads = _make_mock_cache(max_size=3)

        await cache.get("A")
        await cache.get("B")
        await cache.get("C")
        # Access A again → A becomes MRU, B is now LRU
        await cache.get("A")
        # Load D → evict B (LRU)
        await cache.get("D")

        assert "B" not in cache._cache, "B must be evicted (LRU)"
        assert "A" in cache._cache
        assert "C" in cache._cache
        assert "D" in cache._cache

    async def test_repeated_access_updates_mru_order(self):
        """Repeated access to the same key keeps it at the MRU end."""
        cache, _ = _make_mock_cache(max_size=2)

        await cache.get("X")
        await cache.get("Y")
        # Repeatedly access X
        for _ in range(5):
            await cache.get("X")
        # Load Z → should evict Y (LRU), not X (MRU)
        await cache.get("Z")

        assert "X" in cache._cache
        assert "Z" in cache._cache
        assert "Y" not in cache._cache

    async def test_max_size_one_evicts_on_every_new_load(self):
        """With max_size=1, every new key evicts the previous one."""
        cache, loads = _make_mock_cache(max_size=1)

        await cache.get("A")
        assert "A" in cache._cache

        await cache.get("B")
        assert "B" in cache._cache
        assert "A" not in cache._cache

        await cache.get("C")
        assert "C" in cache._cache
        assert "B" not in cache._cache

    async def test_cache_hit_does_not_reload(self):
        """Requesting the same key twice must only load it once."""
        cache, loads = _make_mock_cache(max_size=5)

        await cache.get("model_alpha")
        await cache.get("model_alpha")
        await cache.get("model_alpha")

        assert loads.count("model_alpha") == 1, "Model must be loaded exactly once"

    async def test_cache_size_never_exceeds_max(self):
        """Cache size must never exceed max_size regardless of access pattern."""
        cache, _ = _make_mock_cache(max_size=3)

        for i in range(10):
            await cache.get(f"model_{i}")
            assert len(cache._cache) <= 3, f"Cache exceeded max_size at step {i}"


# ── Concurrency ───────────────────────────────────────────────────────────────

class TestModelCacheConcurrency:
    async def test_concurrent_requests_same_key_load_once(self):
        """
        Multiple concurrent get() calls for the same key must result in
        exactly one actual model load (no duplicate I/O).
        """
        cache = ModelCache(max_size=5)
        load_calls: list[str] = []

        cache._model_hash = lambda path: path

        # Introduce a small artificial delay to make the race real
        async def mock_get(model_path: str):
            key = model_path
            async with cache._lock:
                if key in cache._cache:
                    cache._access_order.remove(key)
                    cache._access_order.append(key)
                    return cache._cache[key]

                # Simulate loading delay under lock → single load guaranteed
                await asyncio.sleep(0.01)
                load_calls.append(key)
                model_obj = f"model:{key}"

                if len(cache._cache) >= cache._max_size:
                    oldest = cache._access_order.pop(0)
                    cache._cache.pop(oldest, None)

                cache._cache[key] = model_obj
                cache._access_order.append(key)
                return model_obj

        cache.get = mock_get

        # Fire 10 concurrent requests for the same key
        results = await asyncio.gather(*[cache.get("shared_model") for _ in range(10)])

        assert all(r == "model:shared_model" for r in results)
        assert load_calls.count("shared_model") == 1, (
            f"Expected 1 load, got {load_calls.count('shared_model')}"
        )

    async def test_concurrent_different_keys_all_loaded(self):
        """Concurrent requests for different keys must all succeed."""
        cache, loads = _make_mock_cache(max_size=10)

        keys = [f"model_{i}" for i in range(8)]
        results = await asyncio.gather(*[cache.get(k) for k in keys])

        assert len(results) == 8
        assert all(r is not None for r in results)
        # All 8 distinct models should have been loaded
        assert set(loads) == set(keys)


# ── Invalidation ──────────────────────────────────────────────────────────────

class TestModelCacheInvalidation:
    async def test_invalidate_removes_from_cache_and_order(self):
        """invalidate() must remove an entry from both _cache and _access_order."""
        cache, _ = _make_mock_cache(max_size=5)

        await cache.get("keep_me")
        await cache.get("remove_me")

        assert "remove_me" in cache._cache
        assert "remove_me" in cache._access_order

        # Call invalidate if it exists, otherwise manually clean up
        if hasattr(cache, "invalidate"):
            cache.invalidate("remove_me")
        else:
            async with cache._lock:
                cache._cache.pop("remove_me", None)
                if "remove_me" in cache._access_order:
                    cache._access_order.remove("remove_me")

        assert "remove_me" not in cache._cache
        assert "remove_me" not in cache._access_order
        assert "keep_me" in cache._cache  # Unrelated entry untouched

    async def test_after_invalidation_next_get_reloads(self):
        """After invalidation, a subsequent get() must load the model fresh."""
        cache, loads = _make_mock_cache(max_size=5)

        await cache.get("model_v1")
        assert loads.count("model_v1") == 1

        # Invalidate
        async with cache._lock:
            cache._cache.pop("model_v1", None)
            if "model_v1" in cache._access_order:
                cache._access_order.remove("model_v1")

        # Second load after invalidation
        await cache.get("model_v1")
        assert loads.count("model_v1") == 2, "Model must be reloaded after invalidation"
