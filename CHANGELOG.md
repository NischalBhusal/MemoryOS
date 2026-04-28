# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-28

### Added
- Initial release of MemoryOS
- Core `Memory` class with `add`, `search`, `remember`, `summarize`, `forget`, `clear`, `list_all`, and `count` methods
- `Storage` class: thread-safe SQLite backend with schema versioning and WAL mode
- `LocalEmbeddings` class: singleton-cached `sentence-transformers` engine with batch support
- `search_vectors`: NumPy cosine similarity with min-score threshold filtering
- `MemoryOSError`, `OllamaUnavailableError`, `MemoryNotFoundError` custom exceptions
- `Config` dataclass for centralized, type-safe configuration
- Example scripts: `basic_usage.py`, `chatbot_with_memory.py`, `ollama_integration.py`
- Comprehensive pytest test suite
- GitHub Actions CI workflow
- Full type hints and docstrings throughout
