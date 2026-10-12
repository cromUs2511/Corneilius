"""Authorized workspace file operations. All paths stay inside the configured root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.tools.registry import CONFIRMATION_REQUIRED, READ_ONLY, ToolRegistry, ToolSpec

MAX_LIST_ENTRIES = 200
MAX_READ_CHARS = 100_000
MAX_WRITE_CHARS = 500_000


def resolve_within(root: Path, user_path: str) -> Path:
    """Resolve a user-supplied path and ensure it stays inside root."""
    cleaned = (user_path or ".").strip().replace("\\", "/")
    if not cleaned or cleaned == ".":
        return root.resolve()
    candidate = (root / cleaned).resolve()
    base = root.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Path escapes the authorized workspace: {user_path!r}.") from exc
    return candidate


def _relative_display(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def register_workspace_tools(registry: ToolRegistry, workspace_root: Path) -> None:
    root = workspace_root.expanduser()

    def list_dir(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_within(root, str(args.get("path", ".")))
        recursive = bool(args.get("recursive", False))
        if not target.exists():
            raise ValueError(f"Path does not exist: {args.get('path', '.')!r}.")
        if not target.is_dir():
            raise ValueError(f"Not a directory: {args.get('path')!r}.")
        if recursive:
            entries = sorted(p for p in target.rglob("*") if p != target)
        else:
            entries = sorted(target.iterdir())
        entries = entries[:MAX_LIST_ENTRIES]
        lines = []
        for entry in entries:
            try:
                rel = entry.resolve().relative_to(root.resolve())
            except ValueError:
                continue  # skip symlinks pointing outside the workspace
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{rel.as_posix()}{suffix}")
        output = "\n".join(lines) if lines else "(empty directory)"
        return {"output": output, "count": len(lines), "truncated": len(entries) >= MAX_LIST_ENTRIES}

    def read_file(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_within(root, str(args["path"]))
        max_chars = int(args.get("max_chars", 20_000))
        max_chars = max(1, min(max_chars, MAX_READ_CHARS))
        if not target.exists():
            raise ValueError(f"File does not exist: {args['path']!r}.")
        if not target.is_file():
            raise ValueError(f"Not a file: {args['path']!r}.")
        if target.stat().st_size > MAX_READ_CHARS * 4:
            raise ValueError(f"File is too large to read safely: {args['path']!r}.")
        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        return {
            "output": text[:max_chars],
            "path": _relative_display(root, target),
            "size_chars": len(text),
            "truncated": truncated,
        }

    def write_file(args: dict[str, Any]) -> dict[str, Any]:
        root.mkdir(parents=True, exist_ok=True)
        target = resolve_within(root, str(args["path"]))
        content = str(args["content"])
        if len(content) > MAX_WRITE_CHARS:
            raise ValueError(f"Content exceeds the {MAX_WRITE_CHARS}-character safety limit.")
        create_dirs = bool(args.get("create_dirs", False))
        if target.exists() and target.is_dir():
            raise ValueError(f"Refusing to overwrite directory: {args['path']!r}.")
        if not target.parent.exists():
            if not create_dirs:
                raise ValueError(
                    f"Parent directory does not exist: {target.parent.name!r}. Use create_dirs=true."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding="utf-8")
        action = "Overwrote" if existed else "Created"
        return {
            "output": f"{action} {_relative_display(root, target)} ({len(content)} characters).",
            "path": _relative_display(root, target),
            "bytes": len(content.encode('utf-8')),
            "overwrote": existed,
        }

    registry.register(
        ToolSpec(
            name="workspace.list",
            description="List files inside the authorized workspace.",
            risk=READ_ONLY,
            parameters={
                "path": {"type": "string", "required": False, "default": ".", "max_length": 1000},
                "recursive": {"type": "boolean", "required": False, "default": False},
            },
        ),
        list_dir,
    )
    registry.register(
        ToolSpec(
            name="workspace.read",
            description="Read a text file inside the authorized workspace.",
            risk=READ_ONLY,
            parameters={
                "path": {"type": "string", "required": True, "max_length": 1000},
                "max_chars": {"type": "integer", "required": False, "default": 20000},
            },
        ),
        read_file,
    )
    registry.register(
        ToolSpec(
            name="workspace.write",
            description="Create or overwrite a text file inside the authorized workspace.",
            risk=CONFIRMATION_REQUIRED,
            parameters={
                "path": {"type": "string", "required": True, "max_length": 1000},
                "content": {"type": "string", "required": True, "max_length": MAX_WRITE_CHARS},
                "create_dirs": {"type": "boolean", "required": False, "default": False},
            },
        ),
        write_file,
    )
