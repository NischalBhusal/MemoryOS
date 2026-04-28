"""
Comprehensive test suite for MemoryOS.

Tests are isolated: each test gets a fresh in-memory (`:memory:`) SQLite
database and never touches the filesystem.
"""

from __future__ import annotations

import pytest

from memoryos import Config, Memory, SearchResult
from memoryos.exceptions import OllamaUnavailableError, ValidationError
from memoryos.search import search_vectors
from memoryos.storage import Storage
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config pointing at a temporary SQLite file that is deleted after each test."""
    return Config(db_path=tmp_path / "test.db")


@pytest.fixture()
def mem(cfg: Config) -> Memory:
    """Fresh Memory instance for each test."""
    return Memory(config=cfg)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.llm_model == "llama3"
    assert 0.0 <= cfg.forget_threshold <= 1.0


def test_config_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="forget_threshold"):
        Config(forget_threshold=1.5)


def test_config_invalid_min_score() -> None:
    with pytest.raises(ValueError, match="search_min_score"):
        Config(search_min_score=-0.1)


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


def test_storage_insert_and_retrieve(cfg: Config) -> None:
    store = Storage(cfg.db_path)
    rid = store.insert("hello world", [0.1, 0.2, 0.3])
    assert isinstance(rid, int)
    assert rid >= 1
    rows = store.get_all()
    assert len(rows) == 1
    assert rows[0]["text"] == "hello world"
    assert rows[0]["vector"] == [0.1, 0.2, 0.3]


def test_storage_count(cfg: Config) -> None:
    store = Storage(cfg.db_path)
    assert store.count() == 0
    store.insert("a", [1.0])
    store.insert("b", [2.0])
    assert store.count() == 2


def test_storage_delete_returns_bool(cfg: Config) -> None:
    store = Storage(cfg.db_path)
    rid = store.insert("to delete", [0.0])
    assert store.delete(rid) is True
    assert store.delete(rid) is False  # already gone


def test_storage_clear_returns_count(cfg: Config) -> None:
    store = Storage(cfg.db_path)
    store.insert("a", [1.0])
    store.insert("b", [2.0])
    deleted = store.clear()
    assert deleted == 2
    assert store.count() == 0


def test_storage_session_isolation(cfg: Config) -> None:
    store = Storage(cfg.db_path)
    store.insert("msg A", [0.1], session_id="s1")
    store.insert("msg B", [0.2], session_id="s2")
    assert len(store.get_by_session("s1")) == 1
    assert len(store.get_by_session("s2")) == 1
    assert len(store.get_by_session("s3")) == 0


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


def test_search_vectors_empty() -> None:
    result = search_vectors([1.0, 0.0], [])
    assert result.shape == (0,)


def test_search_vectors_identical() -> None:
    v = [1.0, 0.0, 0.0]
    scores = search_vectors(v, [v])
    assert abs(scores[0] - 1.0) < 1e-5


def test_search_vectors_orthogonal() -> None:
    scores = search_vectors([1.0, 0.0], [[0.0, 1.0]])
    assert abs(scores[0]) < 1e-5


def test_search_vectors_min_score_filters() -> None:
    v = [1.0, 0.0]
    orthogonal = [0.0, 1.0]
    scores = search_vectors(v, [orthogonal], min_score=0.5)
    assert scores[0] < 0  # sentinel -1.0


def test_search_vectors_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dim"):
        search_vectors([1.0, 0.0], [[1.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Memory — add / search
# ---------------------------------------------------------------------------


def test_add_returns_id(mem: Memory) -> None:
    rid = mem.add("User loves hiking")
    assert isinstance(rid, int)
    assert rid >= 1


def test_add_empty_raises(mem: Memory) -> None:
    with pytest.raises(ValidationError):
        mem.add("   ")


def test_search_returns_results(mem: Memory) -> None:
    mem.add("Python is a high-level programming language")
    mem.add("The Eiffel Tower is in Paris")
    results = mem.search("programming language", top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert "Python" in results[0].text
    assert results[0].score > 0.5


def test_search_empty_query_raises(mem: Memory) -> None:
    with pytest.raises(ValidationError):
        mem.search("")


def test_search_top_k_invalid_raises(mem: Memory) -> None:
    with pytest.raises(ValidationError):
        mem.search("anything", top_k=0)


def test_search_empty_store_returns_empty(mem: Memory) -> None:
    assert mem.search("anything") == []


def test_search_top_k_respected(mem: Memory) -> None:
    for i in range(10):
        mem.add(f"Memory number {i}")
    results = mem.search("memory", top_k=3)
    assert len(results) <= 3


def test_search_session_filter(mem: Memory) -> None:
    mem.add("Session A fact", session_id="s_a")
    mem.add("Session B fact", session_id="s_b")
    results = mem.search("fact", top_k=5, session_id="s_a")
    assert all(r.session_id == "s_a" for r in results)


# ---------------------------------------------------------------------------
# Memory — forget / clear / count / list_all
# ---------------------------------------------------------------------------


def test_forget_removes_best_match(mem: Memory) -> None:
    mem.add("I enjoy playing chess")
    mem.add("I love hiking in the mountains")
    deleted = mem.forget("chess", threshold=0.3)
    assert deleted is True
    results = mem.search("chess", top_k=1, min_score=0.5)
    assert not any("chess" in r.text for r in results)


def test_forget_below_threshold_returns_false(mem: Memory) -> None:
    mem.add("I enjoy playing chess")
    # Query is totally unrelated → should not cross the threshold
    deleted = mem.forget("quantum physics nuclear reactor", threshold=0.95)
    assert deleted is False
    assert mem.count() == 1  # nothing removed


def test_clear_removes_all(mem: Memory) -> None:
    mem.add("A")
    mem.add("B")
    n = mem.clear()
    assert n == 2
    assert mem.count() == 0


def test_count(mem: Memory) -> None:
    assert mem.count() == 0
    mem.add("one")
    mem.add("two")
    assert mem.count() == 2


def test_list_all(mem: Memory) -> None:
    mem.add("alpha")
    mem.add("beta")
    items = mem.list_all()
    assert len(items) == 2
    assert all(isinstance(r, SearchResult) for r in items)


def test_list_all_limit(mem: Memory) -> None:
    for i in range(5):
        mem.add(f"item {i}")
    assert len(mem.list_all(limit=3)) == 3


# ---------------------------------------------------------------------------
# Memory — context manager
# ---------------------------------------------------------------------------


def test_context_manager(cfg: Config) -> None:
    with Memory(config=cfg) as m:
        m.add("inside context manager")
        assert m.count() == 1
    # After __exit__, connection is closed; no exception should be raised.


# ---------------------------------------------------------------------------
# Memory — Ollama (mocked)
# ---------------------------------------------------------------------------


def test_summarize_no_memories(mem: Memory) -> None:
    result = mem.summarize()
    assert "No memories" in result


def test_remember_no_session(mem: Memory) -> None:
    result = mem.remember(session_id="nonexistent")
    assert result is None


def test_ollama_error_raises_custom_exception(mem: Memory, monkeypatch: pytest.MonkeyPatch) -> None:
    """summarize() should raise OllamaUnavailableError, not a raw ollama exception."""
    import ollama as _ollama

    def _fail(*args: object, **kwargs: object) -> None:
        raise ConnectionRefusedError("Ollama is not running")

    monkeypatch.setattr(_ollama, "chat", _fail)
    mem.add("some memory")

    with pytest.raises(OllamaUnavailableError, match="Ollama call failed"):
        mem.summarize()
