"""Tool audit records stored in the existing SQLite database."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _sanitize_args(args: dict[str, Any], limit: int = 200) -> str:
    """Serialize tool arguments without storing large or private payloads verbatim."""
    redacted: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > limit:
            redacted[key] = value[:limit] + f"...<truncated {len(value)} chars>"
        else:
            redacted[key] = value
    return json.dumps(redacted, ensure_ascii=False)


class ToolAuditLog:
    """Append-only audit trail for tool executions."""

    def __init__(self, database_path: Path) -> None:
        self.path = database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS tool_audit (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    tool TEXT NOT NULL,
                    risk TEXT NOT NULL DEFAULT 'read-only',
                    args TEXT NOT NULL DEFAULT '{}',
                    ok INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    elapsed_ms INTEGER NOT NULL DEFAULT 0
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

    def record(
        self,
        tool: str,
        risk: str,
        args: dict[str, Any],
        ok: bool,
        error: str = "",
        elapsed_ms: int = 0,
    ) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO tool_audit(tool, risk, args, ok, error, elapsed_ms)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (tool, risk, _sanitize_args(args), 1 if ok else 0, error[:500], int(elapsed_ms)),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, created_at, tool, risk, args, ok, error, elapsed_ms"
                " FROM tool_audit ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
