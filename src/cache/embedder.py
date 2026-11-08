"""ONNX Runtime embedder (FastEmbed) producing Redis-ready float32 vectors."""
import asyncio
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from fastembed import TextEmbedding

from src.routing.classifier import (
    COMPLEX_PROTOTYPES,
    SIMPLE_PROTOTYPES,
    SemanticCentroids,
    build_semantic_centroids,
)

logger = logging.getLogger(__name__)
_LRU_MAXSIZE = 1024
_MAX_EMBED_CHARS = 2000


def _normalize_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


class OnnxEmbedder:
    """Thin wrapper around FastEmbed that returns normalized float32 bytes with a thread-safe LRU cache."""

    def __init__(
        self,
        model_name: str,
        expected_dim: int,
        model: Any = None,
        cache_dir: str | None = ".cache/fastembed",
    ) -> None:
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._cache_dir = os.getenv("FASTEMBED_CACHE_PATH", cache_dir)
        self._model = model
        if self._model is None:
            try:
                self._model = TextEmbedding(model_name=model_name, cache_dir=self._cache_dir, threads=2)
            except Exception as exc:
                logger.warning("embedder_init_failed_lexical_fallback", extra={"error": str(exc)})
                self._model = None
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        # OrderedDict (not lru_cache) so aembed can serve hits on the event loop without asyncio.to_thread
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    def _get_cached(self, key: str) -> bytes | None:
        with self._cache_lock:
            if (vec := self._cache.get(key)) is not None:
                self._cache.move_to_end(key)
            return vec

    def embed(self, text: str) -> bytes:
        """Synchronously embed one string (<0.05ms on LRU hit, ~1-3ms on cold ONNX inference)."""
        # all-MiniLM-L6-v2 truncates at 256 tokens (~1k chars); slicing [:2000] avoids storing/tokenizing 100k chars
        key = _normalize_key(text)[:_MAX_EMBED_CHARS]
        with self._lock:
            if (hit := self._get_cached(key)) is not None:
                return hit
            if self._model is None:
                raise RuntimeError("ONNX embedding model is unavailable")
            vector: np.ndarray = next(iter(self._model.embed([key]))).astype(np.float32)
            if vector.shape[0] != self._expected_dim:
                raise ValueError(f"Embedding dim {vector.shape[0]} != expected {self._expected_dim}")
            vec = vector.tobytes()
        with self._cache_lock:
            self._cache[key] = vec
            self._cache.move_to_end(key)
            if len(self._cache) > _LRU_MAXSIZE:
                self._cache.popitem(last=False)
        return vec

    async def aembed(self, text: str) -> bytes:
        """Return immediately from memory on LRU hit; offload cold ONNX inference to a worker thread."""
        key = _normalize_key(text)[:_MAX_EMBED_CHARS]
        if (hit := self._get_cached(key)) is not None:
            return hit
        if self._model is None:
            raise RuntimeError("ONNX embedding model is unavailable")
        return await asyncio.to_thread(self.embed, key)

    def warmup(self) -> SemanticCentroids | None:
        """Warm ONNX Runtime in 2 batched calls and return Complex/Simple semantic prototype centroids."""
        if self._model is None:
            return None
        try:
            with self._lock:
                complex_vecs = [
                    np.asarray(v, dtype=np.float32).tobytes()
                    for v in self._model.embed(list(COMPLEX_PROTOTYPES))
                ]
                simple_vecs = [
                    np.asarray(v, dtype=np.float32).tobytes()
                    for v in self._model.embed(list(SIMPLE_PROTOTYPES))
                ]
            centroids = build_semantic_centroids(complex_vecs, simple_vecs)
            logger.info("embedder_ready", extra={"model": self._model_name, "dim": self._expected_dim})
            return centroids
        except Exception as exc:
            logger.warning("embedder_warmup_failed_lexical_fallback", extra={"error": str(exc)})
            return None
