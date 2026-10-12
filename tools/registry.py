"""Tool registry with schema validation, timeouts, and audit records."""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jarvis.tools.audit import ToolAuditLog

logger = logging.getLogger(__name__)

READ_ONLY = "read-only"
REVERSIBLE = "reversible"
CONFIRMATION_REQUIRED = "confirmation-required"
RESTRICTED = "restricted"

RISKS = frozenset({READ_ONLY, REVERSIBLE, CONFIRMATION_REQUIRED, RESTRICTED})

ParamSchema = dict[str, Any]
Handler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    risk: str
    parameters: dict[str, ParamSchema] = field(default_factory=dict)

    def usage(self) -> str:
        parts = []
        for key, schema in self.parameters.items():
            marker = "required" if schema.get("required") else "optional"
            parts.append(f"{key} ({schema.get('type', 'string')}, {marker})")
        params = ", ".join(parts) if parts else "no arguments"
        return f"{self.name} [{self.risk}]: {self.description} Args: {params}"


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    name: str
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_ms: int = 0
    confirmation_required: bool = False


def _check_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def validate_args(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    """Validate raw arguments against a spec; return cleaned args or raise ValueError."""
    if not isinstance(args, dict):
        raise ValueError(f"Arguments for '{spec.name}' must be a JSON object.")
    unknown = sorted(set(args) - set(spec.parameters))
    if unknown:
        raise ValueError(f"Unknown argument(s) for '{spec.name}': {', '.join(unknown)}.")
    cleaned: dict[str, Any] = {}
    for key, schema in spec.parameters.items():
        if key in args:
            value = args[key]
            expected = str(schema.get("type", "string"))
            if not _check_type(value, expected):
                raise ValueError(f"Argument '{key}' for '{spec.name}' must be {expected}.")
            if isinstance(value, str) and "max_length" in schema and len(value) > int(schema["max_length"]):
                raise ValueError(f"Argument '{key}' exceeds maximum length {schema['max_length']}.")
            cleaned[key] = value
        elif schema.get("required"):
            raise ValueError(f"Missing required argument '{key}' for '{spec.name}'.")
        elif "default" in schema:
            cleaned[key] = schema["default"]
    return cleaned


class ToolRegistry:
    """Registry of safe tools. No shell access is exposed through this registry."""

    def __init__(self, audit: ToolAuditLog, default_timeout: float = 10.0) -> None:
        self._audit = audit
        self._default_timeout = max(0.5, float(default_timeout))
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Handler] = {}

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        if not spec.name or not spec.name.replace(".", "").replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid tool name: {spec.name!r}.")
        if spec.risk not in RISKS:
            raise ValueError(f"Unknown risk level: {spec.risk!r}.")
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}.")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        confirmed: bool = False,
        timeout: float | None = None,
    ) -> ToolResult:
        started = time.monotonic()
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(ok=False, name=name, output="", error=f"Unknown tool: {name}.")

        if spec.risk == RESTRICTED:
            self._audit.record(name, spec.risk, args if isinstance(args, dict) else {}, False, "restricted", 0)
            return ToolResult(ok=False, name=name, output="", error=f"Tool '{name}' is restricted and disabled.")

        try:
            cleaned = validate_args(spec, args)
        except ValueError as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            self._audit.record(name, spec.risk, args if isinstance(args, dict) else {}, False, str(exc), elapsed)
            return ToolResult(ok=False, name=name, output="", error=str(exc), elapsed_ms=elapsed)

        if spec.risk == CONFIRMATION_REQUIRED and not confirmed:
            elapsed = int((time.monotonic() - started) * 1000)
            self._audit.record(name, spec.risk, cleaned, False, "confirmation-required", elapsed)
            return ToolResult(
                ok=False,
                name=name,
                output="",
                error=f"Tool '{name}' requires confirmation. Reply /confirm to proceed or /cancel to abort.",
                elapsed_ms=elapsed,
                confirmation_required=True,
            )

        limit = self._default_timeout if timeout is None else max(0.5, float(timeout))
        handler = self._handlers[name]
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(handler, cleaned)
            try:
                raw = future.result(timeout=limit)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        except concurrent.futures.TimeoutError as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            self._audit.record(name, spec.risk, cleaned, False, f"timed out after {limit:g}s", elapsed)
            logger.warning("Tool %s timed out after %ss", name, limit)
            return ToolResult(ok=False, name=name, output="", error=f"Tool '{name}' timed out after {limit:g}s.", elapsed_ms=elapsed)
        except Exception as exc:  # handler errors become structured failures, never raised
            elapsed = int((time.monotonic() - started) * 1000)
            message = str(exc) or type(exc).__name__
            self._audit.record(name, spec.risk, cleaned, False, message, elapsed)
            logger.warning("Tool %s failed: %s", name, message)
            return ToolResult(ok=False, name=name, output="", error=message, elapsed_ms=elapsed)

        elapsed = int((time.monotonic() - started) * 1000)
        if not isinstance(raw, dict) or "output" not in raw:
            self._audit.record(name, spec.risk, cleaned, False, "malformed tool result", elapsed)
            return ToolResult(ok=False, name=name, output="", error=f"Tool '{name}' returned a malformed result.", elapsed_ms=elapsed)
        output = str(raw.get("output", ""))
        data = {k: v for k, v in raw.items() if k != "output"}
        self._audit.record(name, spec.risk, cleaned, True, "", elapsed)
        logger.info("Tool %s ok in %sms", name, elapsed)
        return ToolResult(ok=True, name=name, output=output, data=data, elapsed_ms=elapsed)


def build_default_registry(
    audit: ToolAuditLog,
    workspace_root: Path,
    default_timeout: float = 10.0,
) -> ToolRegistry:
    """Create the toolset: workspace ops, system info, web research, code tools."""
    from jarvis.tools.code import register_code_tools
    from jarvis.tools.system_info import register_system_tools
    from jarvis.tools.web import register_web_tools
    from jarvis.tools.workspace import register_workspace_tools

    registry = ToolRegistry(audit, default_timeout=default_timeout)
    register_workspace_tools(registry, workspace_root)
    register_system_tools(registry)
    register_web_tools(registry)
    register_code_tools(registry, workspace_root)
    return registry
