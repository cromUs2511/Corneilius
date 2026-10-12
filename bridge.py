"""JSON-RPC bridge between the Electron UI and the Corneilius backend.

A persistent Python process speaking line-delimited JSON over stdio, so the
desktop app needs no ports and works fully offline:

    --> {"id": 1, "method": "chat.send", "params": {"message": "hello"}}
    <-- {"id": 1, "ok": true, "result": {"reply": "..."}}
    <-- {"event": "progress", "data": {"text": "Agent step 1: ..."}}
    <-- {"id": 2, "ok": false, "error": "..."}

Run:  python -m jarvis.bridge
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from jarvis.config import Settings
from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore
from jarvis.models.local_model import OllamaModel
from jarvis.models.mock_model import MockModel
from jarvis.tools.audit import ToolAuditLog

logger = logging.getLogger(__name__)


def _model_for(settings: Settings):  # type: ignore[no-untyped-def]
    if settings.model_provider == "mock":
        return MockModel()
    if settings.model_provider == "ollama":
        return OllamaModel(settings.ollama_url, settings.ollama_model, settings.request_timeout)
    raise ValueError("JARVIS_MODEL_PROVIDER must be 'mock' or 'ollama'.")


class Bridge:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.workspace_root.expanduser().mkdir(parents=True, exist_ok=True)
        memory = ConversationStore(self.settings.database_path)
        self.memory = memory
        self.orchestrator = Orchestrator(_model_for(self.settings), memory, self.settings)
        self.orchestrator.on_progress = self.emit_progress
        self._write_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bridge")

    # -- transport ---------------------------------------------------------

    def emit(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self._write_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def emit_progress(self, text: str) -> None:
        try:
            self.emit({"event": "progress", "data": {"text": text}})
        except Exception:
            logger.warning("Progress emit failed", exc_info=True)

    def respond(self, request_id: Any, ok: bool, result: Any = None, error: str = "") -> None:
        payload: dict[str, Any] = {"id": request_id, "ok": ok}
        if ok:
            payload["result"] = result
        else:
            payload["error"] = error
        self.emit(payload)

    # -- methods -----------------------------------------------------------

    def handle(self, method: str, params: dict[str, Any]) -> Any:
        orch = self.orchestrator
        if method == "health":
            return {
                "provider": self.settings.model_provider,
                "model": self.settings.ollama_model,
                "workspace": str(self.settings.workspace_root),
                "data_dir": str(self.settings.data_dir),
            }
        if method == "settings.get":
            return {
                "provider": self.settings.model_provider,
                "model": self.settings.ollama_model,
                "ollama_url": self.settings.ollama_url,
                "data_dir": str(self.settings.data_dir),
                "workspace": str(self.settings.workspace_root),
                "request_timeout": self.settings.request_timeout,
                "tool_timeout": self.settings.tool_timeout,
                "agent_max_iterations": self.settings.agent_max_iterations,
                "agent_timeout": self.settings.agent_timeout,
            }
        if method == "chat.send":
            message = str(params.get("message", "")).strip()
            if not message:
                raise ValueError("Message cannot be empty.")
            return {"reply": orch.submit(message)}
        if method == "chat.cancel":
            orch.cancel_agent()
            return {"cancelled": True}
        if method == "memory.list":
            category = params.get("category")
            entries = orch.long_term.list_entries(category) if category else orch.long_term.list_entries()
            return {"entries": [dict(e) for e in entries]}
        if method == "memory.search":
            return {"entries": [dict(e) for e in orch.long_term.search(str(params.get("query", "")))]}
        if method == "memory.delete":
            deleted = orch.long_term.delete(int(params["id"]))
            return {"deleted": deleted}
        if method == "history.list":
            limit = max(1, min(int(params.get("limit", 200)), 500))
            return {"messages": self.memory.get_messages(limit=limit)}
        if method == "tools.list":
            return {"tools": [
                {"name": spec.name, "description": spec.description, "risk": spec.risk}
                for spec in orch.tools.specs()
            ]}
        if method == "tools.run":
            result = orch.tools.execute(
                str(params.get("name", "")),
                dict(params.get("args", {})),
                confirmed=bool(params.get("confirmed", False)),
            )
            return {
                "ok": result.ok,
                "output": result.output,
                "error": result.error,
                "elapsed_ms": result.elapsed_ms,
                "confirmation_required": result.confirmation_required,
            }
        if method == "audit.recent":
            limit = max(1, min(int(params.get("limit", 20)), 100))
            return {"entries": ToolAuditLog(self.memory.path).recent(limit)}
        raise ValueError(f"Unknown method: {method}.")

    def _dispatch(self, line: str) -> None:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            self.respond(None, False, error="Invalid JSON request.")
            return
        if not isinstance(request, dict):
            self.respond(None, False, error="Request must be a JSON object.")
            return
        if request.get("method") == "chat.send_stream":
            self._stream_chat(request.get("id"), dict(request.get("params", {})))
            return
        try:
            request_id = request.get("id")
            result = self.handle(str(request.get("method", "")), dict(request.get("params", {})))
        except Exception as exc:
            logger.exception("Bridge request failed")
            try:
                request_id = json.loads(line).get("id")
            except Exception:
                request_id = None
            self.respond(request_id, False, error=str(exc) or type(exc).__name__)
            return
        self.respond(request_id, True, result=result)

    def _stream_chat(self, request_id: Any, params: dict[str, Any]) -> None:
        """Yield reply chunks as chat_chunk events, then one final response."""
        message = str(params.get("message", "")).strip()
        if not message:
            self.respond(request_id, False, error="Message cannot be empty.")
            return
        try:
            stream = self.orchestrator.submit_stream(message)
            try:
                for chunk in stream:
                    self.emit({"event": "chat_chunk",
                               "data": {"request_id": request_id, "text": chunk}})
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            logger.exception("Bridge streaming request failed")
            self.respond(request_id, False, error=str(exc) or type(exc).__name__)
            return
        self.respond(request_id, True, result={
            "reply": self.orchestrator._last_final,
            "streamed": not self.orchestrator._last_replaced,
            "cancelled": self.orchestrator._agent_cancel.is_set(),
        })

    def serve_forever(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            self._pool.submit(self._dispatch, line)
        self._pool.shutdown(wait=True)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr)])
    # Windows defaults stdio to cp1252, which crashes on model output like → or emoji.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        logger.warning("Could not set UTF-8 stdio", exc_info=True)
    Bridge().serve_forever()


if __name__ == "__main__":
    main()
