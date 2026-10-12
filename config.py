"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("JARVIS_DATA_DIR", Path.home() / ".jarvis"))
    model_provider: str = os.getenv("JARVIS_MODEL_PROVIDER", "mock").lower()
    ollama_url: str = os.getenv("JARVIS_OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("JARVIS_OLLAMA_MODEL", "qwen2.5:3b")
    request_timeout: float = float(os.getenv("JARVIS_REQUEST_TIMEOUT", "60"))
    workspace_root: Path = Path(os.getenv("JARVIS_WORKSPACE", Path.home() / "jarvis-workspace"))
    tool_timeout: float = float(os.getenv("JARVIS_TOOL_TIMEOUT", "10"))
    agent_max_iterations: int = int(os.getenv("JARVIS_AGENT_MAX_ITERATIONS", "6"))
    agent_timeout: float = float(os.getenv("JARVIS_AGENT_TIMEOUT", "120"))
    agent_retry: int = int(os.getenv("JARVIS_AGENT_RETRY", "1"))

    @property
    def database_path(self) -> Path:
        return self.data_dir / "jarvis.sqlite3"
