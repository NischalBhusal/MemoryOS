# MemoryOS 🧠

<div align="center">

[![PyPI version](https://badge.fury.io/py/memoryos.svg)](https://badge.fury.io/py/memoryos)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Persistent, semantic, long-term memory for any LLM app — 100% local, 100% free, zero setup.**

[Quickstart](#-quickstart) · [How it works](#-how-it-works) · [API Reference](#-api-reference) · [Contributing](#-contributing)

</div>

---

## Why MemoryOS?

LLMs forget everything between sessions. MemoryOS solves this by giving your app a persistent, searchable memory layer that:

- **Stores memories** in a local SQLite database (zero external dependencies)
- **Retrieves relevant context** via cosine similarity over local embeddings
- **Summarises sessions** with a locally-running Ollama model
- **Runs offline** on an 8 GB laptop — no API keys, no cloud, no cost

---

## ⚡ Quickstart

### 1. Install

```bash
pip install memoryos
```

> **Prerequisites**: [Ollama](https://ollama.com/) installed and running with a model pulled:
> ```bash
> ollama pull llama3
> ```
> The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first use.

### 2. Use it

```python
from memoryos import Memory

mem = Memory()                           # zero config — works instantly

mem.add("User prefers dark mode")        # store a memory
mem.add("User is learning Python")

results = mem.search("what does the user like?", top_k=3)
for r in results:
    print(r.text, r.score)

mem.remember(session_id="chat_001")      # compress & store a session
summary = mem.summarize()                # AI summary of all memories
mem.forget("dark mode")                  # delete by semantic match
count = mem.clear()                      # wipe all; returns count deleted
```

### 3. Custom config

```python
from memoryos import Memory, Config

mem = Memory(config=Config(
    db_path="my_app.db",
    embedding_model="all-mpnet-base-v2",   # higher quality embeddings
    llm_model="phi3",                       # use a lighter Ollama model
    forget_threshold=0.6,                   # stricter forget matching
    log_level="DEBUG",
))
```

---

## 🛠️ How it Works

```
mem.add("User likes Python")
       │
       ▼
┌─────────────────────┐      ┌──────────────────────────────────┐
│  LocalEmbeddings    │ ───► │  Storage (SQLite / WAL mode)     │
│  sentence-          │      │  id │ text          │ vector      │
│  transformers       │      │   1 │ User likes …  │ [0.12, …]  │
│  (CPU, offline)     │      └──────────────────────────────────┘
└─────────────────────┘

mem.search("what language does user prefer?")
       │
       ▼
┌──────────────────────────┐
│  Embed query → vector    │
│  cosine_similarity(      │
│    query_vec, all_vecs)  │  → ranked SearchResult list
└──────────────────────────┘

mem.summarize()
       │
       ▼
┌──────────────────────────┐
│  All memory texts        │
│  → Ollama prompt         │  → narrative summary string
│  (local LLM, offline)   │
└──────────────────────────┘
```

| Step | Module | Technology |
|------|--------|------------|
| Embed text | `embeddings.py` | `sentence-transformers` (CPU) |
| Store vector | `storage.py` | SQLite + WAL mode |
| Retrieve context | `search.py` | NumPy cosine similarity |
| Summarise | `memory.py` | Ollama local LLM |

---

## 📖 API Reference

### `Memory(config=None)`

| Method | Description | Returns |
|--------|-------------|---------|
| `add(text, session_id=None)` | Store a memory | `int` (memory ID) |
| `search(query, top_k=3, min_score=None, session_id=None)` | Semantic search | `List[SearchResult]` |
| `forget(query, threshold=None)` | Delete best-matching memory | `bool` |
| `remember(session_id)` | Summarise & store a session | `Optional[int]` |
| `summarize()` | AI narrative of all memories | `str` |
| `list_all(session_id=None, limit=None)` | List all memories | `List[SearchResult]` |
| `count()` | Total memory count | `int` |
| `clear()` | Wipe everything | `int` (count deleted) |
| `close()` | Release DB connection | `None` |

### `SearchResult`

```python
@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float          # cosine similarity, 0–1
    memory_id: int
    session_id: Optional[str]
```

### Exceptions

| Exception | When raised |
|-----------|-------------|
| `MemoryOSError` | Base class for all MemoryOS errors |
| `StorageError` | SQLite backend failure |
| `EmbeddingError` | Embedding model failure |
| `OllamaUnavailableError` | Ollama not running or model not found |
| `ValidationError` | Empty text, invalid `top_k`, etc. |

---

## 🆚 Comparison

| Feature | **MemoryOS** | Chroma / Weaviate | LangChain Memory | OpenAI Memory |
|---------|-------------|-------------------|------------------|---------------|
| **Cost** | ✅ Free | ✅ / ❌ Paid tiers | ✅ Free | ❌ Paid |
| **Offline** | ✅ 100% | ⚠️ Self-host needed | ✅ / ❌ depends | ❌ Cloud |
| **Setup** | ✅ `pip install` | ❌ Docker / daemon | ⚠️ Complex | ❌ API key |
| **Dependencies** | Minimal | Heavy | Very heavy | HTTP only |
| **Privacy** | ✅ Data stays local | ⚠️ Self-host | Varies | ❌ Cloud |
| **RAG-ready** | ✅ | ✅ | ✅ | ❌ |

---

## 🤝 Contributing

We love contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

```bash
git clone https://github.com/NischalBhusal/MemoryOS
cd MemoryOS
pip install -e ".[dev]"
pytest                  # run tests
ruff check memoryos/    # lint
mypy memoryos/          # type check
```

Please update [CHANGELOG.md](CHANGELOG.md) with your changes.

---

## 📄 License

[MIT](LICENSE) © MemoryOS Contributors
