"""
Thread-safe SQLite storage backend for MemoryOS.

Uses WAL (Write-Ahead Logging) mode for concurrent reads, and a single
persistent connection per thread via ``threading.local()`` to avoid the
overhead of reconnecting on every call while remaining thread-safe.
"""

import json
import sqlite3
import threading
import datetime
from pathlib import Path
from typing import Any, Optional

from .exceptions import StorageError


_SCHEMA_VERSION = 1

_CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER  PRIMARY KEY AUTOINCREMENT,
    text       TEXT     NOT NULL,
    vector     TEXT     NOT NULL,
    timestamp  TEXT     NOT NULL,
    session_id TEXT
)
"""

_CREATE_SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
)
"""

_CREATE_TEXT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories (session_id)
"""


class Storage:
    """
    Thread-safe SQLite backend for MemoryOS.

    Each calling thread gets its own SQLite connection (via ``threading.local``),
    so the object can be safely shared across threads. WAL mode is enabled for
    better concurrent read performance.

    Args:
        db_path: Path to the ``.db`` file. The parent directory will be created
            automatically if it does not exist.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Eagerly initialise on the calling thread so errors surface early.
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        """Return this thread's cached SQLite connection, creating it if needed."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def close(self) -> None:
        """Close this thread's SQLite connection (call from the same thread)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Schema initialisation & migration
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and apply any pending schema migrations."""
        try:
            conn = self._connection()
            with conn:
                conn.execute(_CREATE_SCHEMA_TABLE)
                conn.execute(_CREATE_MEMORIES_TABLE)
                conn.execute(_CREATE_TEXT_INDEX)

                row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                if row is None:
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
                # Future migrations: elif row["version"] < 2: …
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to initialise database at '{self.db_path}': {exc}") from exc

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def insert(
        self,
        text: str,
        vector: list[float],
        session_id: Optional[str] = None,
    ) -> int:
        """
        Insert a new memory record.

        Args:
            text: The plaintext memory string.
            vector: The embedding vector as a Python list of floats.
            session_id: Optional session tag for grouping related memories.

        Returns:
            The ``id`` of the newly inserted row.

        Raises:
            StorageError: On any SQLite failure.
        """
        try:
            conn = self._connection()
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with conn:
                cursor = conn.execute(
                    "INSERT INTO memories (text, vector, timestamp, session_id) VALUES (?, ?, ?, ?)",
                    (text, json.dumps(vector), timestamp, session_id),
                )
                return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to insert memory: {exc}") from exc

    def get_all(self) -> list[Dict[str, Any]]:
        """
        Retrieve every memory record.

        Returns:
            A list of dicts with keys ``id``, ``text``, ``vector``,
            ``timestamp``, and ``session_id``.

        Raises:
            StorageError: On any SQLite failure.
        """
        try:
            conn = self._connection()
            rows = conn.execute(
                "SELECT id, text, vector, timestamp, session_id FROM memories ORDER BY id"
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to retrieve memories: {exc}") from exc

    def get_by_session(self, session_id: str) -> list[Dict[str, Any]]:
        """
        Retrieve all memories belonging to a specific session.

        Args:
            session_id: The session tag to filter on.

        Returns:
            Ordered list of memory dicts for the given session.

        Raises:
            StorageError: On any SQLite failure.
        """
        try:
            conn = self._connection()
            rows = conn.execute(
                "SELECT id, text, vector, timestamp, session_id FROM memories "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to retrieve session memories: {exc}") from exc

    def delete(self, memory_id: int) -> bool:
        """
        Delete a specific memory by primary key.

        Args:
            memory_id: The ``id`` of the row to delete.

        Returns:
            ``True`` if a row was deleted, ``False`` if it didn't exist.

        Raises:
            StorageError: On any SQLite failure.
        """
        try:
            conn = self._connection()
            with conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to delete memory id={memory_id}: {exc}") from exc

    def clear(self) -> int:
        """
        Delete all memory records.

        Returns:
            The number of rows deleted.

        Raises:
            StorageError: On any SQLite failure.
        """
        try:
            conn = self._connection()
            with conn:
                cursor = conn.execute("DELETE FROM memories")
                return cursor.rowcount
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to clear memories: {exc}") from exc

    def count(self) -> int:
        """
        Return the total number of stored memories.

        Raises:
            StorageError: On any SQLite failure.
        """
        try:
            conn = self._connection()
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return row[0]
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to count memories: {exc}") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a ``sqlite3.Row`` into a plain dict with deserialized vector."""
        return {
            "id": row["id"],
            "text": row["text"],
            "vector": json.loads(row["vector"]),
            "timestamp": row["timestamp"],
            "session_id": row["session_id"],
        }
