# Contributing to MemoryOS

Thank you for considering contributing! MemoryOS is an open-source project and we welcome all kinds of contributions — bug reports, documentation improvements, new features, and more.

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/your-username/memoryos.git
cd memoryos
```

### 2. Set Up Development Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 3. Run the Test Suite

```bash
pytest
```

All tests must pass before submitting a PR.

## Code Standards

- **Type hints** on every function signature
- **Docstrings** on every public class and method (Google style)
- **No magic strings** — use constants or config values
- Format code with `ruff format .` before committing
- Run `mypy memoryos/` and fix all warnings

## Submitting a Pull Request

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes with clean, focused commits
3. Update `CHANGELOG.md` under the `[Unreleased]` section
4. Open a PR against `main` with a clear description of what changed and why

## Reporting Bugs

Open a GitHub Issue with:
- Your OS and Python version
- A minimal reproduction script
- The full error traceback

## Feature Requests

Open a GitHub Discussion with your use case. We love hearing how people are using MemoryOS!
