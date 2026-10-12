"""Phase 5 coding-assistant tools: search, precise edits, tests, git awareness.

Everything stays inside the authorized workspace. Edits require exact unique
matches; test runs execute only a fixed unittest command; git commands are a
fixed read-only allowlist (status/diff). Commits, pushes, checkouts, and
arbitrary shell commands are intentionally not exposed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from jarvis.tools.registry import CONFIRMATION_REQUIRED, READ_ONLY, ToolRegistry, ToolSpec
from jarvis.tools.workspace import resolve_within

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".testdata",
             ".testdata-tools", ".testdata-agent", ".testdata-natural",
             ".testdata-stream", ".testdata-web", ".testdata-code"}
MAX_SEARCH_BYTES = 500_000
MAX_MATCHES = 50
MAX_EDIT_CHARS = 500_000
TEST_TIMEOUT = 90.0
GIT_TIMEOUT = 15.0


def _iter_text_files(root: Path, base: Path):  # type: ignore[no-untyped-def]
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            rel = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        try:
            if path.stat().st_size > MAX_SEARCH_BYTES:
                continue
        except OSError:
            continue
        yield path, rel


def _is_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\x00" not in handle.read(4096)
    except OSError:
        return False


def register_code_tools(registry: ToolRegistry, workspace_root: Path) -> None:
    root = workspace_root.expanduser()

    def search_code(args: dict[str, Any]) -> dict[str, Any]:
        try:
            pattern = re.compile(str(args.get("pattern", "")), re.MULTILINE)
        except re.error as exc:
            raise ValueError(f"Invalid search pattern: {exc}.")
        if not str(args.get("pattern", "")).strip():
            raise ValueError("Search pattern cannot be empty.")
        context = max(0, min(int(args.get("context", 1)), 5))
        target = resolve_within(root, str(args.get("path", ".")))
        if not target.exists():
            raise ValueError(f"Path does not exist: {args.get('path', '.')!r}.")
        bases = [target] if target.is_dir() else [target]
        if not target.is_dir() and not _is_text(target):
            raise ValueError(f"Not a searchable text file: {args.get('path')!r}.")
        matches: list[str] = []
        files_hit = 0
        for base in bases:
            files = [(base, Path(str(args.get("path", "."))))] if base.is_file() else list(_iter_text_files(root, base))
            for path, rel in files:
                if len(matches) >= MAX_MATCHES:
                    break
                if not _is_text(path):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                lines = text.splitlines()
                for number, line in enumerate(lines, start=1):
                    if len(matches) >= MAX_MATCHES:
                        break
                    if pattern.search(line):
                        files_hit += 1
                        start = max(1, number - context)
                        end = min(len(lines), number + context)
                        snippet = "\n".join(f"  {n}:{lines[n - 1][:160]}" for n in range(start, end + 1))
                        matches.append(f"{rel.as_posix()}:{number}:\n{snippet}")
        if not matches:
            return {"output": f"No matches for pattern in {args.get('path', '.')!r}.", "matches": 0}
        capped = f" (capped at {MAX_MATCHES})" if len(matches) >= MAX_MATCHES else ""
        return {"output": f"{len(matches)} match(es){capped}:\n\n" + "\n\n".join(matches),
                "matches": len(matches)}

    def edit_code(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_within(root, str(args["path"]))
        old = str(args["old"])
        new = str(args["new"])
        if not old:
            raise ValueError("The 'old' text cannot be empty.")
        if len(new) > MAX_EDIT_CHARS:
            raise ValueError(f"Replacement exceeds the {MAX_EDIT_CHARS}-character safety limit.")
        if not target.exists() or not target.is_file():
            raise ValueError(f"File does not exist: {args['path']!r}.")
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Could not read {args['path']!r}: {exc}.")
        occurrences = text.count(old)
        if occurrences == 0:
            raise ValueError(f"Exact text not found in {args['path']!r}; nothing changed.")
        if occurrences > 1:
            raise ValueError(
                f"Text matches {occurrences} times in {args['path']!r}; "
                "include more context so it is unique. Nothing changed."
            )
        lineno = text[: text.index(old)].count("\n") + 1
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return {"output": f"Edited {args['path']!r} at line {lineno} "
                          f"({len(new)} chars replace {len(old)} chars).",
                "path": str(args["path"]), "line": lineno}

    def run_tests(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_within(root, str(args.get("path", ".")))
        if not target.exists():
            raise ValueError(f"Path does not exist: {args.get('path', '.')!r}.")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", str(target)],
                cwd=str(root.resolve()), capture_output=True, text=True, timeout=TEST_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"Tests exceeded the {TEST_TIMEOUT:g}s limit.") from exc
        except OSError as exc:
            raise ValueError(f"Could not start tests: {exc}.") from exc
        tail = (completed.stdout + "\n" + completed.stderr).strip()[-4000:]
        verdict = "passed" if completed.returncode == 0 else "FAILED"
        return {"output": f"Tests {verdict} (exit {completed.returncode}):\n{tail}",
                "returncode": completed.returncode,
                "passed": completed.returncode == 0}

    def git_status(args: dict[str, Any]) -> dict[str, Any]:
        if shutil.which("git") is None:
            raise ValueError("git is not installed.")
        target = resolve_within(root, str(args.get("path", ".")))
        directory = target if target.is_dir() else target.parent
        try:
            completed = subprocess.run(
                ["git", "status", "--porcelain"], cwd=str(directory),
                capture_output=True, text=True, timeout=GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("git status timed out.") from exc
        except OSError as exc:
            raise ValueError(f"Could not run git: {exc}.") from exc
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()[:300]
            if "not a git repository" in details:
                raise ValueError(f"Not a git repository: {args.get('path', '.')!r}.")
            raise ValueError(f"git status failed: {details}")
        lines = completed.stdout.strip().splitlines()
        # Skip the branch header line (starts with "##")
        porcelain_lines = [l for l in lines if not l.startswith("##")]
        body = "\n".join(porcelain_lines) if porcelain_lines else ""
        return {"output": body or "Clean working tree.", "clean": not bool(porcelain_lines)}

    def git_diff(args: dict[str, Any]) -> dict[str, Any]:
        if shutil.which("git") is None:
            raise ValueError("git is not installed.")
        target = resolve_within(root, str(args.get("path", ".")))
        directory = target if target.is_dir() else target.parent
        stat_only = bool(args.get("stat_only", True))
        command = ["git", "diff", "--stat"] if stat_only else ["git", "diff", "--", "."]
        try:
            completed = subprocess.run(
                command, cwd=str(directory), capture_output=True, text=True, timeout=GIT_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("git diff timed out.") from exc
        except OSError as exc:
            raise ValueError(f"Could not run git: {exc}.") from exc
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()[:300]
            if "not a git repository" in details:
                raise ValueError(f"Not a git repository: {args.get('path', '.')!r}.")
            raise ValueError(f"git diff failed: {details}")
        body = completed.stdout.strip()[:6000]
        return {"output": body or "No uncommitted changes.", "empty": not bool(body)}

    registry.register(
        ToolSpec(
            name="repo.search",
            description="Regex-search text files in the workspace with context lines.",
            risk=READ_ONLY,
            parameters={
                "pattern": {"type": "string", "required": True, "max_length": 500},
                "path": {"type": "string", "required": False, "default": ".", "max_length": 1000},
                "context": {"type": "integer", "required": False, "default": 1},
            },
        ),
        search_code,
    )
    registry.register(
        ToolSpec(
            name="repo.edit",
            description="Replace one exact unique text block in a workspace file. Fails unless unique.",
            risk=CONFIRMATION_REQUIRED,
            parameters={
                "path": {"type": "string", "required": True, "max_length": 1000},
                "old": {"type": "string", "required": True, "max_length": MAX_EDIT_CHARS},
                "new": {"type": "string", "required": True, "max_length": MAX_EDIT_CHARS},
            },
        ),
        edit_code,
    )
    registry.register(
        ToolSpec(
            name="repo.test",
            description="Run the fixed unittest discovery inside the workspace (no custom commands).",
            risk=CONFIRMATION_REQUIRED,
            parameters={
                "path": {"type": "string", "required": False, "default": ".", "max_length": 1000},
            },
        ),
        run_tests,
    )
    registry.register(
        ToolSpec(
            name="repo.git_status",
            description="Show git branch and short status for a workspace path (read-only).",
            risk=READ_ONLY,
            parameters={
                "path": {"type": "string", "required": False, "default": ".", "max_length": 1000},
            },
        ),
        git_status,
    )
    registry.register(
        ToolSpec(
            name="repo.git_diff",
            description="Show uncommitted changes for a workspace path (read-only).",
            risk=READ_ONLY,
            parameters={
                "path": {"type": "string", "required": False, "default": ".", "max_length": 1000},
                "stat_only": {"type": "boolean", "required": False, "default": True},
            },
        ),
        git_diff,
    )
