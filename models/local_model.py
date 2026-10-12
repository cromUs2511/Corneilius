from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

from .base import ModelAdapter, ModelError

KEEP_ALIVE = "30m"  # keep weights in VRAM between chats so replies start fast


class OllamaModel(ModelAdapter):
    """Optional adapter for Ollama's local /api/chat endpoint."""

    def __init__(self, base_url: str, model: str, timeout: float = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _payload(self, message: str, history: list[dict[str, str]], stream: bool) -> bytes:
        messages = [{"role": row["role"], "content": row["content"]} for row in history]
        messages.append({"role": "user", "content": message})
        return json.dumps({
            "model": self.model, "messages": messages, "stream": stream, "keep_alive": KEEP_ALIVE,
        }).encode()

    def _request(self, payload: bytes) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )

    def respond(self, message: str, history: list[dict[str, str]]) -> str:
        try:
            with urllib.request.urlopen(self._request(self._payload(message, history, False)),
                                        timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("message", {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise ModelError("Ollama returned an empty or malformed response.")
            return content.strip()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and check the configured model ({self.model})."
            ) from exc
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            raise ModelError("Ollama returned an invalid response.") from exc

    def stream_respond(self, message: str, history: list[dict[str, str]]) -> Iterator[str]:
        """Yield reply deltas as they arrive. Raises ModelError on connection failure."""
        try:
            stream = urllib.request.urlopen(self._request(self._payload(message, history, True)),
                                            timeout=self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and check the configured model ({self.model})."
            ) from exc
        try:
            with stream:
                saw_content = False
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue  # skip malformed keep-alive frames
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        raise ModelError(f"Ollama reported an error: {event['error']}")
                    content = event.get("message", {}).get("content")
                    if isinstance(content, str) and content:
                        saw_content = True
                        yield content
                    if event.get("done"):
                        break
                if not saw_content:
                    raise ModelError("Ollama returned an empty response.")
        except ModelError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError("Ollama connection failed mid-response.") from exc
