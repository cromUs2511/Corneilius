"""Persistent agent task runs: resumable state that survives restarts.

The orchestrator keeps pause/resume state in memory for speed, but every
agent run is also recorded here so a paused or interrupted task can be
listed and resumed after a restart. Steps are stored as plain JSON dicts
so old rows stay readable even if the AgentStep dataclass changes.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

VALID_STATUSES = frozenset({
    "running", "paused", "finished", "cancelled", "failed",
    "timeout", "model-error", "max-iterations",
})


class TaskRunStore:
    def __init__(self, database_path: Path) -> None:
        self.path = database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS task_runs (
                    id INTEGER PRIMARY KEY,
                    goal TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    steps_json TEXT NOT NULL DEFAULT '[]',
                    pending_json TEXT NOT NULL DEFAULT '{}',
                    flags_json TEXT NOT NULL DEFAULT '{}',
                    result TEXT NOT NULL DEFAULT '',
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

    def create(self, goal: str, flags: dict[str, Any] | None = None) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO task_runs(goal, status, flags_json) VALUES (?, 'running', ?)",
                (goal[:2000], json.dumps(flags or {}, ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def save_pause(
        self,
        task_id: int,
        steps: list[dict[str, Any]],
        pending: dict[str, Any] | None,
        flags: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE task_runs SET status = 'paused', steps_json = ?, pending_json = ?, "
                "flags_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (
                    json.dumps(steps, ensure_ascii=False),
                    json.dumps(pending or {}, ensure_ascii=False),
                    json.dumps(flags or {}, ensure_ascii=False),
                    task_id,
                ),
            )

    def finish(
        self,
        task_id: int,
        status: str,
        steps: list[dict[str, Any]],
        result: str = "",
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Unknown task status: {status!r}.")
        with self._connect() as db:
            db.execute(
                "UPDATE task_runs SET status = ?, steps_json = ?, result = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, json.dumps(steps, ensure_ascii=False), result[:4000], task_id),
            )

    def get(self, task_id: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, goal, status, steps_json, pending_json, flags_json, result,"
                " created_at, updated_at FROM task_runs WHERE id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        for key in ("steps_json", "pending_json", "flags_json"):
            try:
                data[key.replace("_json", "")] = json.loads(data[key] or ("[]" if key == "steps_json" else "{}"))
            except (json.JSONDecodeError, TypeError):
                data[key.replace("_json", "")] = [] if key == "steps_json" else {}
        return data

    def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, goal, status, created_at, updated_at FROM task_runs"
                " ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_resumable(self, task_id: int) -> dict[str, Any] | None:
        """Return a paused/running row with decoded steps, else None."""
        data = self.get(task_id)
        if data is None or data["status"] not in {"paused", "running"}:
            return None
        return data
