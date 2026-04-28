"""
Local embedding engine for MemoryOS.

Uses ``sentence-transformers`` to run embeddings entirely on-device (CPU).
The underlying ``SentenceTransformer`` model is cached per model-name so
multiple ``Memory`` instances sharing the same model name don't reload the
weights from disk more than once per process.
"""

from __future__ import annotations

import threading
from typing import ClassVar, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .exceptions import EmbeddingError


class LocalEmbeddings:
    """
    On-device embedding engine backed by sentence-transformers.

    Model weights are loaded only once per model name per process and
    reused across all instances (process-level singleton cache with a lock
    for thread safety during the first load).

    Args:
        model_name: Any model compatible with ``SentenceTransformer``, e.g.
            ``"all-MiniLM-L6-v2"`` (fast, 22M params) or
            ``"all-mpnet-base-v2"`` (higher quality, 110M params).
        batch_size: Maximum number of texts to encode in a single forward pass.
    """

    _cache: ClassVar[dict[str, SentenceTransformer]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = self._load_model(model_name)

    # ------------------------------------------------------------------
    # Model loading (cached singleton per model name)
    # ------------------------------------------------------------------

    @classmethod
    def _load_model(cls, model_name: str) -> SentenceTransformer:
        """Return the cached model, loading it on first access."""
        if model_name not in cls._cache:
            with cls._lock:
                # Double-checked locking: another thread may have loaded it
                # while we were waiting for the lock.
                if model_name not in cls._cache:
                    try:
                        cls._cache[model_name] = SentenceTransformer(model_name)
                    except Exception as exc:
                        raise EmbeddingError(
                            f"Failed to load embedding model '{model_name}': {exc}\n"
                            "Ensure sentence-transformers is installed and the model name is valid."
                        ) from exc
        return cls._cache[model_name]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """
        Generate a dense vector for a single piece of text.

        Args:
            text: The input string to embed. Must be non-empty.

        Returns:
            A 1-D NumPy float32 array.

        Raises:
            EmbeddingError: If encoding fails.
            ValueError: If ``text`` is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty string.")
        try:
            return self._model.encode(text, convert_to_numpy=True)  # type: ignore[return-value]
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed for text '{text[:40]}…': {exc}") from exc

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings in optimised batches.

        Args:
            texts: A list of non-empty strings.

        Returns:
            A 2-D NumPy float32 array of shape ``(len(texts), embedding_dim)``.

        Raises:
            EmbeddingError: If encoding fails.
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts list must not be empty.")
        try:
            return self._model.encode(  # type: ignore[return-value]
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Batch embedding failed: {exc}") from exc
