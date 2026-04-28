"""
Centralized configuration for MemoryOS.

All tuneable defaults live here. Pass a ``Config`` instance to ``Memory``
to override any value — no scattered magic strings in core modules.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """
    Configuration container for MemoryOS.

    Attributes:
        db_path: Path to the SQLite database file.
        embedding_model: HuggingFace sentence-transformers model name.
            Must be a model compatible with ``SentenceTransformer``.
        llm_model: Ollama model tag used for summarization.
        embedding_batch_size: Number of texts to embed in a single batch.
        forget_threshold: Minimum cosine similarity score required before
            ``forget()`` will delete a memory. Prevents accidental deletion
            of unrelated memories.
        search_min_score: Minimum cosine similarity for ``search()`` results.
            Results below this value are silently filtered out.
        log_level: Python logging level string (``"DEBUG"``, ``"INFO"``, etc.).
    """

    db_path: Path = field(default_factory=lambda: Path("memoryos.db"))
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama3"
    embedding_batch_size: int = 64
    forget_threshold: float = 0.4
    search_min_score: float = 0.0
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate config values after construction."""
        self.db_path = Path(self.db_path)
        if not (0.0 <= self.forget_threshold <= 1.0):
            raise ValueError(
                f"forget_threshold must be between 0.0 and 1.0, got {self.forget_threshold}"
            )
        if not (0.0 <= self.search_min_score <= 1.0):
            raise ValueError(
                f"search_min_score must be between 0.0 and 1.0, got {self.search_min_score}"
            )
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be >= 1")
