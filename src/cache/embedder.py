"""ONNX Runtime embedder (FastEmbed) producing Redis-ready float32 vectors."""
import asyncio
import logging
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from src.routing.classifier import (
    COMPLEX_PROTOTYPES,
    SIMPLE_PROTOTYPES,
    register_semantic_prototypes,
)

logger = logging.getLogger(__name__)


def _normalize_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


class OnnxEmbedder:
    """Thin wrapper around FastEmbed that returns normalized float32 bytes with an LRU cache."""

    def __init__(self, model_name: str, expected_dim: int) -> None:
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._model = TextEmbedding(model_name=model_name)
        self._cached_embed = lru_cache(maxsize=1024)(self._embed_raw)

    def _embed_raw(self, norm_text: str) -> bytes:
        vector: np.ndarray = next(iter(self._model.embed([norm_text]))).astype(np.float32)
        if vector.shape[0] != self._expected_dim:
            raise ValueError(f"Embedding dim {vector.shape[0]} != expected {self._expected_dim}")
        return vector.tobytes()

    def embed(self, text: str) -> bytes:
        """Synchronously embed one string (<0.05ms on LRU hit, ~1-3ms on cold ONNX inference)."""
        return self._cached_embed(_normalize_key(text))

    async def aembed(self, text: str) -> bytes:
        """Embed in a worker thread on cache miss; return in <0.1ms on LRU hit."""
        return await asyncio.to_thread(self.embed, text)

    def warmup(self) -> None:
        """Warm ONNX Runtime and register Complex/Simple semantic prototype centroids."""
        register_semantic_prototypes(
            [self.embed(p) for p in COMPLEX_PROTOTYPES],
            [self.embed(p) for p in SIMPLE_PROTOTYPES],
        )
        logger.info("embedder_ready", extra={"model": self._model_name, "dim": self._expected_dim})
