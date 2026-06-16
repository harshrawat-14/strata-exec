"""
Model cache service to avoid reloading PyTorch models.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import sys
from typing import Any

from web.config import get_settings

settings = get_settings()


class ModelCache:
    """LRU cache for loaded RecurrentPPO models."""

    def __init__(self, max_size: int = 5):
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()

    def _model_hash(self, model_path: str) -> str:
        """Generate content hash for model file to identify changes."""
        path = Path(model_path)
        if not path.exists() and not model_path.endswith(".zip"):
            path = Path(model_path + ".zip")

        if not path.exists():
            # If path still does not exist, use path string as key fallback
            return hashlib.sha256(model_path.encode()).hexdigest()[:16]

        h = hashlib.sha256()
        with open(path, "rb") as f:
            # Read in chunks
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    async def get(self, model_path: str) -> Any:
        """Get or load a model from cache in a thread-safe way."""
        key = self._model_hash(model_path)

        async with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]

            # Resolve evaluate import
            project_root = str(Path(settings.rust_binary_path).parent.parent)
            rl_dir = str(Path(project_root) / "rl")
            for p in [project_root, rl_dir]:
                if p not in sys.path:
                    sys.path.insert(0, p)

            from evaluate import load_model

            # Load model in executor since loading torch zip states is CPU-bound
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(None, load_model, model_path)

            # Evict oldest if capacity exceeded
            if len(self._cache) >= self._max_size:
                oldest = self._access_order.pop(0)
                if oldest in self._cache:
                    del self._cache[oldest]

            self._cache[key] = model
            self._access_order.append(key)
            return model


# Singleton cache instance
model_cache = ModelCache(max_size=5)
