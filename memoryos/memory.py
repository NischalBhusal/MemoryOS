"""
Core Memory class — the primary public API for MemoryOS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import ollama

from .config import Config
from .embeddings import LocalEmbeddings
from .exceptions import MemoryNotFoundError, OllamaUnavailableError, ValidationError
from .search import search_vectors
from .storage import Storage
from .utils import get_logger


@dataclass(frozen=True)
class SearchResult:
    """
    An immutable result returned by :py:meth:`Memory.search`.

    Attributes:
        text: The full plaintext of the matched memory.
        score: Cosine similarity score in the range ``[0, 1]``.
        memory_id: The primary-key ID in the SQLite database.
        session_id: The session tag this memory belongs to, if any.
    """

    text: str
    score: float
    memory_id: int
    session_id: Optional[str] = field(default=None)

    def __str__(self) -> str:
        tag = f" [{self.session_id}]" if self.session_id else ""
        return f"({self.score:.3f}){tag} {self.text}"


class Memory:
    """
    Local-first, persistent, semantic memory layer for LLM applications.

    Stores memories as text + embedding vector pairs in a local SQLite database.
    Retrieval is done via cosine similarity search, entirely on-device.
    Summarisation delegates to a locally running Ollama model.

    Args:
        config: A :class:`~memoryos.config.Config` instance. All defaults are
            sensible out-of-the-box — ``Memory()`` with zero arguments works
            immediately.

    Example::

        from memoryos import Memory

        mem = Memory()
        mem.add("User prefers dark mode")
        mem.add("User is learning Python")

        for r in mem.search("what does the user like?", top_k=3):
            print(r.text, r.score)

        mem.forget("dark mode")
        mem.clear()
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._cfg = config or Config()
        self._logger = get_logger(__name__, level=self._cfg.log_level)
        self._storage = Storage(self._cfg.db_path)
        self._embeddings = LocalEmbeddings(
            model_name=self._cfg.embedding_model,
            batch_size=self._cfg.embedding_batch_size,
        )
        self._logger.info(
            "MemoryOS ready — db=%s  model=%s  llm=%s",
            self._cfg.db_path,
            self._cfg.embedding_model,
            self._cfg.llm_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, text: str, session_id: Optional[str] = None) -> int:
        """
        Store a new memory.

        Args:
            text: The memory content. Must be non-empty after stripping
                whitespace.
            session_id: Optional tag for grouping related memories into
                logical sessions.

        Returns:
            The integer ID of the newly stored memory.

        Raises:
            ValidationError: If ``text`` is blank.
        """
        text = text.strip()
        if not text:
            raise ValidationError("Memory text must not be empty.")

        vector = self._embeddings.embed(text)
        memory_id = self._storage.insert(text, vector.tolist(), session_id)
        self._logger.debug("add() id=%d session=%s text='%s'", memory_id, session_id, _preview(text))
        return memory_id

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Semantically search stored memories.

        Args:
            query: Natural-language search string.
            top_k: Maximum number of results to return.
            min_score: Override the config ``search_min_score`` for this call.
            session_id: If set, restrict the search to this session only.

        Returns:
            Ranked list of :class:`SearchResult` objects (highest score first).

        Raises:
            ValidationError: If ``query`` is blank or ``top_k`` < 1.
        """
        query = query.strip()
        if not query:
            raise ValidationError("Search query must not be empty.")
        if top_k < 1:
            raise ValidationError(f"top_k must be >= 1, got {top_k}.")

        threshold = min_score if min_score is not None else self._cfg.search_min_score
        all_memories = (
            self._storage.get_by_session(session_id)
            if session_id
            else self._storage.get_all()
        )

        if not all_memories:
            return []

        query_vec = self._embeddings.embed(query)
        doc_vecs = [m["vector"] for m in all_memories]
        scores = search_vectors(query_vec, doc_vecs, min_score=threshold)

        results: List[SearchResult] = []
        for mem, score in zip(all_memories, scores):
            if score < 0:   # sentinel for below-threshold
                continue
            results.append(
                SearchResult(
                    text=mem["text"],
                    score=float(score),
                    memory_id=mem["id"],
                    session_id=mem["session_id"],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def remember(self, session_id: str) -> Optional[int]:
        """
        Summarise a session's messages with Ollama and store the result.

        The compressed summary is stored as a normal memory (no session_id)
        so it is always reachable by future :py:meth:`search` calls.

        Args:
            session_id: The session to summarise.

        Returns:
            The memory ID of the stored summary, or ``None`` if the session
            has no memories.

        Raises:
            OllamaUnavailableError: If the Ollama service is unreachable or
                the model is not available.
        """
        memories = self._storage.get_by_session(session_id)
        if not memories:
            self._logger.warning("remember(): no memories for session '%s'", session_id)
            return None

        texts = "\n".join(f"- {m['text']}" for m in memories)
        prompt = (
            "Summarise the following conversation session into a concise paragraph "
            "capturing all important facts, user preferences, and context. "
            "Be specific and dense — this summary will be retrieved later as long-term memory.\n\n"
            f"{texts}"
        )

        summary = self._call_ollama(prompt, context=f"remember(session='{session_id}')")
        stored_id = self.add(f"[Session {session_id} summary] {summary}")
        self._logger.info("remember(): session '%s' summarised → memory id=%d", session_id, stored_id)
        return stored_id

    def summarize(self) -> str:
        """
        Generate an AI narrative summary of **all** stored memories.

        Returns:
            A human-readable summary string produced by the local LLM.
            Returns a plain message if there are no memories stored.

        Raises:
            OllamaUnavailableError: If the Ollama service is unreachable.
        """
        memories = self._storage.get_all()
        if not memories:
            return "No memories stored yet."

        texts = "\n".join(f"- {m['text']}" for m in memories)
        prompt = (
            "Given the following list of memories about a user, write a clear, "
            "cohesive summary highlighting their key preferences, ongoing projects, "
            "and important context:\n\n"
            f"{texts}"
        )
        return self._call_ollama(prompt, context="summarize()")

    def forget(self, query: str, threshold: Optional[float] = None) -> bool:
        """
        Delete the single best-matching memory.

        A confidence threshold prevents accidentally deleting an unrelated
        memory when no close match exists.

        Args:
            query: Description of the memory to delete.
            threshold: Minimum cosine similarity required to delete. Overrides
                the config ``forget_threshold`` for this call.

        Returns:
            ``True`` if a memory was deleted, ``False`` if no match met the
            threshold.

        Raises:
            ValidationError: If ``query`` is blank.
        """
        min_sim = threshold if threshold is not None else self._cfg.forget_threshold
        results = self.search(query, top_k=1, min_score=min_sim)

        if not results:
            self._logger.warning(
                "forget(): no memory matched '%s' above threshold %.2f", _preview(query), min_sim
            )
            return False

        match = results[0]
        self._storage.delete(match.memory_id)
        self._logger.info(
            "forget(): deleted id=%d score=%.3f text='%s'",
            match.memory_id,
            match.score,
            _preview(match.text),
        )
        return True

    def clear(self) -> int:
        """
        Permanently delete all stored memories.

        Returns:
            Number of memories deleted.
        """
        deleted = self._storage.clear()
        self._logger.info("clear(): removed %d memories.", deleted)
        return deleted

    def list_all(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Return all memories, optionally filtered by session.

        Args:
            session_id: If set, only return memories tagged with this session.
            limit: Maximum number of results to return (most recent first).

        Returns:
            List of :class:`SearchResult` objects with ``score=1.0``.
        """
        records = (
            self._storage.get_by_session(session_id)
            if session_id
            else self._storage.get_all()
        )
        results = [
            SearchResult(text=r["text"], score=1.0, memory_id=r["id"], session_id=r["session_id"])
            for r in records
        ]
        if limit:
            results = results[-limit:]
        return results

    def count(self) -> int:
        """Return the total number of memories currently stored."""
        return self._storage.count()

    def close(self) -> None:
        """Release the database connection for the calling thread."""
        self._storage.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, context: str = "") -> str:
        """
        Send a prompt to Ollama and return the response text.

        Args:
            prompt: The full prompt string.
            context: Short label used in log/error messages.

        Returns:
            The model's response content as a string.

        Raises:
            OllamaUnavailableError: If Ollama is unreachable or returns an error.
        """
        try:
            self._logger.debug("ollama %s → model=%s", context, self._cfg.llm_model)
            response = ollama.chat(
                model=self._cfg.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            # ollama >= 0.2 returns a typed object; older versions return a dict.
            if hasattr(response, "message"):
                return response.message.content  # type: ignore[union-attr]
            return response["message"]["content"]  # type: ignore[index]
        except Exception as exc:
            hint = (
                f"Is Ollama running? Try: ollama serve\n"
                f"Is the model available? Try: ollama pull {self._cfg.llm_model}"
            )
            raise OllamaUnavailableError(
                f"Ollama call failed during {context}: {exc}\n{hint}"
            ) from exc


def _preview(text: str, max_len: int = 50) -> str:
    """Return a truncated preview of ``text`` for log messages."""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."
