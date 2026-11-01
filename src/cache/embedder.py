"""ONNX Runtime embedder (FastEmbed) producing Redis-ready float32 vectors."""
import asyncio
import logging
from collections import OrderedDict

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


def _normalize_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


class OnnxEmbedder:
    """Thin wrapper around FastEmbed that returns normalized float32 bytes with an LRU cache."""

    def __init__(self, model_name: str, expected_dim: int) -> None:
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._model = TextEmbedding(model_name=model_name)
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    def _embed_raw(self, norm_text: str) -> bytes:
        vector: np.ndarray = next(iter(self._model.embed([norm_text]))).astype(np.float32)
        if vector.shape[0] != self._expected_dim:
            raise ValueError(f"Embedding dim {vector.shape[0]} != expected {self._expected_dim}")
        return vector.tobytes()

    def _get_cached(self, key: str) -> bytes | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _put_cached(self, key: str, vec: bytes) -> bytes:
        self._cache[key] = vec
        self._cache.move_to_end(key)
        if len(self._cache) > _LRU_MAXSIZE:
            self._cache.popitem(last=False)
        return vec

    def embed(self, text: str) -> bytes:
        """Synchronously embed one string (<0.05ms on LRU hit, ~1-3ms on cold ONNX inference)."""
        key = _normalize_key(text)
        hit = self._get_cached(key)
        return hit if hit is not None else self._put_cached(key, self._embed_raw(key))

    async def aembed(self, text: str) -> bytes:
        """Return immediately from memory on LRU hit; offload cold ONNX inference to a worker thread."""
        key = _normalize_key(text)
        hit = self._get_cached(key)
        if hit is not None:
            return hit
        vec = await asyncio.to_thread(self._embed_raw, key)
        return self._put_cached(key, vec)

    def warmup(self) -> SemanticCentroids:
        """Warm ONNX Runtime and return Complex/Simple semantic prototype centroids."""
        centroids = build_semantic_centroids(
            [self.embed(p) for p in COMPLEX_PROTOTYPES],
            [self.embed(p) for p in SIMPLE_PROTOTYPES],
        )
        logger.info("embedder_ready", extra={"model": self._model_name, "dim": self._expected_dim})
        return centroids
