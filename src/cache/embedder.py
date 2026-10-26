"""ONNX Runtime embedder (FastEmbed) producing Redis-ready float32 vectors."""
import asyncio
import logging

import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class OnnxEmbedder:
    """Thin wrapper around FastEmbed that returns normalized float32 bytes."""

    def __init__(self, model_name: str, expected_dim: int) -> None:
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> bytes:
        """Synchronously embed one string (CPU-bound, ~1-3ms)."""
        vector: np.ndarray = next(iter(self._model.embed([text]))).astype(np.float32)
        if vector.shape[0] != self._expected_dim:
            raise ValueError(f"Embedding dim {vector.shape[0]} != expected {self._expected_dim}")
        return vector.tobytes()

    async def aembed(self, text: str) -> bytes:
        """Embed in a worker thread so the event loop is never blocked."""
        return await asyncio.to_thread(self.embed, text)

    def warmup(self) -> None:
        """Run one inference at startup so the first real request avoids the cold-start penalty."""
        self.embed("warmup")
        logger.info("embedder_ready", extra={"model": self._model_name, "dim": self._expected_dim})
