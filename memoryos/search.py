"""
Cosine similarity vector search for MemoryOS.

Implemented with NumPy for fast, dependency-light retrieval over
in-memory arrays. For millions of vectors, consider integrating a
dedicated ANN library (FAISS, hnswlib) — the interface here stays stable.
"""

from __future__ import annotations

from typing import List

import numpy as np


def search_vectors(
    query_vector: List[float] | np.ndarray,
    document_vectors: List[List[float]] | np.ndarray,
    min_score: float = 0.0,
) -> np.ndarray:
    """
    Compute cosine similarity between a query and a corpus of document vectors.

    All vectors should be L2-normalised for best results (sentence-transformers
    returns normalised embeddings by default).

    Args:
        query_vector: 1-D array representing the search query.
        document_vectors: 2-D array of shape ``(n_docs, dim)``.
        min_score: Minimum similarity threshold. Scores below this value are
            set to ``-1.0`` so callers can filter them out easily.

    Returns:
        1-D NumPy array of cosine similarity scores in ``[-1, 1]``,
        clipped to ``[min_score, 1]`` (values below threshold become ``-1``).

    Raises:
        ValueError: If ``document_vectors`` is not 2-D or dimensions mismatch.
    """
    if len(document_vectors) == 0:
        return np.array([], dtype=np.float32)

    q = np.asarray(query_vector, dtype=np.float32)
    docs = np.asarray(document_vectors, dtype=np.float32)

    if docs.ndim != 2:
        raise ValueError(
            f"document_vectors must be 2-D, got shape {docs.shape}."
        )
    if q.shape[0] != docs.shape[1]:
        raise ValueError(
            f"Query dim ({q.shape[0]}) != document dim ({docs.shape[1]})."
        )

    # Cosine similarity: (docs @ q) / (||docs|| * ||q||)
    dot_products = docs @ q
    norms_docs = np.linalg.norm(docs, axis=1)
    norm_q = np.linalg.norm(q)

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(
            (norms_docs * norm_q) == 0.0,
            0.0,
            dot_products / (norms_docs * norm_q),
        )

    # Apply minimum-score threshold: suppress below-threshold results
    scores = np.where(scores >= min_score, scores, -1.0)
    return scores.astype(np.float32)
