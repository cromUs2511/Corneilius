from __future__ import annotations

import logging

try:
    import customtkinter as ctk
except ImportError as exc:
    raise SystemExit(
        "CustomTkinter is required for the Corneilius interface. Install it with:\n"
        "  pip install -r jarvis/requirements.txt"
    ) from exc

from jarvis.config import Settings
from jarvis.core.orchestrator import Orchestrator
from jarvis.interface.desktop import ChatWindow
from jarvis.memory.database import ConversationStore
from jarvis.models.local_model import OllamaModel
from jarvis.models.mock_model import MockModel


def build_app(settings: Settings | None = None) -> tuple[ctk.CTk, ChatWindow]:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.workspace_root.expanduser().mkdir(parents=True, exist_ok=True)
    memory = ConversationStore(settings.database_path)
    if settings.model_provider == "mock":
        model = MockModel()
    elif settings.model_provider == "ollama":
        model = OllamaModel(settings.ollama_url, settings.ollama_model, settings.request_timeout)
    else:
        raise ValueError("JARVIS_MODEL_PROVIDER must be 'mock' or 'ollama'.")
    root = ctk.CTk()
    return root, ChatWindow(root, Orchestrator(model, memory, settings), memory)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler()])
    root, _app = build_app()
    root.mainloop()


if __name__ == "__main__":
    main()
