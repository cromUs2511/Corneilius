"""Versioned, reusable skills plus a correction log for saved memory.

A skill is a named workflow: an ordered list of {tool, args} steps captured
from a successful task trace. New skills are always saved as drafts, must
pass schema validation (``test``) before they can be enabled, and are never
executed automatically -- the orchestrator only lists and runs them on
explicit user request. This keeps Day 3's "never automatically trust
untested generated code" guarantee structural, not just advisory.

Corrections records every user-driven fix to a long-term memory entry so
later training can prefer corrected content over the original.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

VALID_SKILL_STATUSES = frozenset({"draft", "tested", "enabled", "disabled"})
MAX_WORKFLOW_STEPS = 20


class SkillStore:
    def __init__(self, database_path: Path) -> None:
        self.path = database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    workflow_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'draft',
                    source_task_id INTEGER,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, version)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS memory_corrections (
                    id INTEGER PRIMARY KEY,
                    entry_id INTEGER NOT NULL,
                    old_content TEXT NOT NULL DEFAULT '',
                    new_content TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
    def validate_workflow(workflow: Any, registry: Any = None) -> str | None:
        """Return an error string, or None when the workflow is acceptable.

        Checks structure and -- when a registry is given -- that every tool
        exists and its args pass schema validation. Never executes anything.
        """
        if not isinstance(workflow, list) or not workflow:
            return "Workflow must be a non-empty list of {tool, args} steps."
        if len(workflow) > MAX_WORKFLOW_STEPS:
            return f"Workflow exceeds the {MAX_WORKFLOW_STEPS}-step limit."
        for index, step in enumerate(workflow, start=1):
            if not isinstance(step, dict):
                return f"Step {index} must be an object with 'tool' and 'args'."
            tool = step.get("tool")
            args = step.get("args", {})
            if not isinstance(tool, str) or not tool:
                return f"Step {index} has no tool name."
            if not isinstance(args, dict):
                return f"Step {index} ('{tool}') args must be an object."
            if registry is not None:
                specs = {spec.name: spec for spec in registry.specs()}
                spec = specs.get(tool)
                if spec is None:
                    return f"Step {index} uses unknown tool '{tool}'."
                from jarvis.tools.registry import validate_args

                try:
                    validate_args(spec, args)
                except ValueError as exc:
                    return f"Step {index} ('{tool}') invalid: {exc}."
        return None

    def save_draft(
        self,
        name: str,
        workflow: list[dict[str, Any]],
        source_task_id: int | None = None,
        note: str = "",
    ) -> tuple[int, int]:
        name = name.strip()[:100]
        if not name:
            raise ValueError("Skill name cannot be empty.")
        error = self.validate_workflow(workflow)
        if error is not None:
            raise ValueError(f"Invalid workflow: {error}")
        with self._connect() as db:
            row = db.execute(
                "SELECT COALESCE(MAX(version), 0) FROM skills WHERE name = ?", (name,)
            ).fetchone()
            version = int(row[0]) + 1
            cursor = db.execute(
                "INSERT INTO skills(name, version, workflow_json, status, source_task_id, note)"
                " VALUES (?, ?, ?, 'draft', ?, ?)",
                (name, version, json.dumps(workflow, ensure_ascii=False), source_task_id, note[:500]),
            )
            return int(cursor.lastrowid), version

    def mark_tested(self, name: str, version: int | None = None) -> None:
        skill = self.get_latest(name) if version is None else self.get(name, version)
        if skill is None:
            raise KeyError(f"No skill named '{name}'.")
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE skills SET status = 'tested', updated_at = CURRENT_TIMESTAMP"
                " WHERE name = ? AND version = ? AND status IN ('draft', 'tested')",
                (skill["name"], skill["version"]),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Skill '{name}' v{skill['version']} cannot be marked tested from its current state.")

    def enable(self, name: str, version: int | None = None) -> None:
        """Enable only a previously tested workflow; drafts can never be enabled."""
        skill = self.get_latest(name) if version is None else self.get(name, version)
        if skill is None:
            raise KeyError(f"No skill named '{name}'.")
        if skill["status"] != "tested":
            raise ValueError(
                f"Skill '{name}' v{skill['version']} must be tested before enabling"
                f" (current status: {skill['status']})."
            )
        with self._connect() as db:
            db.execute(
                "UPDATE skills SET status = 'tested', updated_at = CURRENT_TIMESTAMP"
                " WHERE name = ? AND status = 'enabled'",
                (skill["name"],),
            )
            db.execute(
                "UPDATE skills SET status = 'enabled', updated_at = CURRENT_TIMESTAMP"
                " WHERE name = ? AND version = ?",
                (skill["name"], skill["version"]),
            )

    def disable(self, name: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE skills SET status = 'disabled', updated_at = CURRENT_TIMESTAMP"
                " WHERE name = ? AND status = 'enabled'",
                (name,),
            )

    def get(self, name: str, version: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, name, version, workflow_json, status, source_task_id, note,"
                " created_at, updated_at FROM skills WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
        if row is None:
            return None
        return self._decode(dict(row))

    def get_latest(self, name: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, name, version, workflow_json, status, source_task_id, note,"
                " created_at, updated_at FROM skills WHERE name = ? ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return self._decode(dict(row))

    def list_skills(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, name, version, status, source_task_id, note, created_at, updated_at,"
                " LENGTH(workflow_json) AS workflow_chars FROM skills ORDER BY name, version"
            ).fetchall()
        return [dict(row) for row in rows]

    def record_correction(self, entry_id: int, old_content: str, new_content: str) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO memory_corrections(entry_id, old_content, new_content) VALUES (?, ?, ?)",
                (entry_id, old_content[:2000], new_content[:2000]),
            )
            return int(cursor.lastrowid)

    def list_corrections(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, entry_id, old_content, new_content, created_at FROM memory_corrections"
                " ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        try:
            row["workflow"] = json.loads(row.pop("workflow_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            row["workflow"] = []
        return row
