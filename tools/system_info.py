"""Read-only system information tool (standard library only)."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

from jarvis.tools.registry import READ_ONLY, ToolRegistry, ToolSpec


def register_system_tools(registry: ToolRegistry) -> None:
    def system_info(args: dict[str, Any]) -> dict[str, Any]:
        del args
        info = (
            f"OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
            f"Python: {sys.version.split()[0]}\n"
            f"CPUs: {os.cpu_count() or 'unknown'}"
        )
        return {
            "output": info,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cpus": os.cpu_count(),
        }

    registry.register(
        ToolSpec(
            name="system.info",
            description="Report basic OS and Python facts without sensitive details.",
            risk=READ_ONLY,
            parameters={},
        ),
        system_info,
    )
