"""
MemoryOS — Local-First AI Memory Layer.

Quickstart::

    from memoryos import Memory

    mem = Memory()
    mem.add("User prefers dark mode")
    results = mem.search("what does the user like?")
    for r in results:
        print(r.text, r.score)
"""

from .config import Config
from .exceptions import (
    EmbeddingError,
    MemoryNotFoundError,
    MemoryOSError,
    OllamaUnavailableError,
    StorageError,
    ValidationError,
)
from .memory import Memory, SearchResult

__version__ = "0.1.0"
__all__ = [
    "Memory",
    "SearchResult",
    "Config",
    # Exceptions
    "MemoryOSError",
    "StorageError",
    "EmbeddingError",
    "OllamaUnavailableError",
    "MemoryNotFoundError",
    "ValidationError",
]
