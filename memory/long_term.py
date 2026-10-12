"""Explicitly saved user facts, preferences, and project notes."""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from jarvis.memory.database import ConversationStore

MemoryEntry = dict[str, int | str]
VALID_CATEGORIES = {"fact", "preference", "project"}


class LongTermMemory:
    def __init__(self, database_path: Path) -> None:
        self.path = database_path
        # Reuse the same connection policy and ensure the base schema exists.
        ConversationStore(database_path)
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY,
                    category TEXT NOT NULL CHECK(category IN ('fact', 'preference', 'project')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _validate_category(category: str) -> None:
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of: {', '.join(sorted(VALID_CATEGORIES))}")

    def save(self, category: str, content: str) -> int:
        self._validate_category(category)
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO long_term_memory(category, content) VALUES (?, ?)",
                (category, content),
            )
            return int(cursor.lastrowid)

    def list_entries(self, category: str | None = None) -> list[MemoryEntry]:
        if category is not None:
            self._validate_category(category)
        with self._connect() as db:
            if category is None:
                rows = db.execute(
                    "SELECT id, category, content, created_at, updated_at FROM long_term_memory ORDER BY id"
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id, category, content, created_at, updated_at FROM long_term_memory "
                    "WHERE category = ? ORDER BY id", (category,),
                ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query: str, category: str | None = None, limit: int = 10) -> list[MemoryEntry]:
        if category is not None:
            self._validate_category(category)
        terms = [term for term in re.findall(r"[\w'-]+", query.casefold()) if len(term) >= 2]
        if not terms:
            return []
        clauses = " OR ".join("LOWER(content) LIKE ?" for _ in terms)
        parameters: list[object] = [f"%{term}%" for term in terms]
        category_clause = ""
        if category is not None:
            category_clause = " AND category = ?"
            parameters.append(category)
        parameters.append(max(1, min(limit, 100)))
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, category, content, created_at, updated_at FROM long_term_memory WHERE (" +
                clauses + ")" + category_clause + " ORDER BY updated_at DESC, id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def retrieve_relevant(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Return deduplicated matching entries, with multi-term matches first."""
        terms = [term for term in re.findall(r"[\w'-]+", query.casefold()) if len(term) >= 3]
        if not terms:
            return []
        matches = self.search(" ".join(terms), limit=100)
        terms_by_id = {
            int(entry["id"]): sum(term in str(entry["content"]).casefold() for term in terms)
            for entry in matches
        }
        return sorted(matches, key=lambda entry: (-terms_by_id[int(entry["id"])], -int(entry["id"])))[:limit]

    def update(self, entry_id: int, content: str, category: str | None = None) -> None:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")
        if category is not None:
            self._validate_category(category)
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS memory_corrections (
                    id INTEGER PRIMARY KEY,
                    entry_id INTEGER NOT NULL,
                    old_content TEXT NOT NULL DEFAULT '',
                    new_content TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            row = db.execute(
                "SELECT content FROM long_term_memory WHERE id = ?", (entry_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"No saved memory with ID {entry_id}.")
            old_content = str(row["content"])
            if category is None:
                cursor = db.execute(
                    "UPDATE long_term_memory SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (content, entry_id),
                )
            else:
                cursor = db.execute(
                    "UPDATE long_term_memory SET category = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (category, content, entry_id),
                )
            if cursor.rowcount == 0:
                raise KeyError(f"No saved memory with ID {entry_id}.")
            if old_content != content:
                db.execute(
                    "INSERT INTO memory_corrections(entry_id, old_content, new_content)"
                    " VALUES (?, ?, ?)",
                    (entry_id, old_content[:2000], content[:2000]),
                )

    def delete(self, entry_id: int) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM long_term_memory WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0
