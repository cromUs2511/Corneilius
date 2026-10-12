from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ConversationStore:
    """SQLite-backed conversations with one persistent default session."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA foreign_keys = ON")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'Conversation',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT OR IGNORE INTO sessions(id, title) VALUES (1, 'Conversation');
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_message(self, role: str, content: str, session_id: int = 1) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            return int(cursor.lastrowid)

    def get_messages(self, session_id: int = 1, limit: int = 200) -> list[dict[str, str]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT role, content, created_at FROM ("
                "SELECT role, content, created_at, id FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
