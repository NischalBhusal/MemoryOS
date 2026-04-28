"""
Custom exception classes for MemoryOS.

Provides a structured exception hierarchy so callers can catch specific
error categories without relying on broad bare `except Exception` blocks.
"""


class MemoryOSError(Exception):
    """Base exception for all MemoryOS errors."""


class StorageError(MemoryOSError):
    """Raised when the SQLite storage layer encounters an unrecoverable error."""


class EmbeddingError(MemoryOSError):
    """Raised when the local embedding engine fails to process text."""


class OllamaUnavailableError(MemoryOSError):
    """
    Raised when the Ollama service cannot be reached.

    This typically means the user hasn't started Ollama, or the requested
    model hasn't been pulled yet (e.g. `ollama pull llama3`).
    """


class MemoryNotFoundError(MemoryOSError):
    """Raised when a requested memory ID does not exist in storage."""


class ValidationError(MemoryOSError):
    """Raised when input validation fails (e.g. empty text, invalid top_k)."""
