from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class ModelError(RuntimeError):
    """A model could not produce a response."""


class ModelAdapter(ABC):
    @abstractmethod
    def respond(self, message: str, history: list[dict[str, str]]) -> str:
        """Return a response for a user message and recent conversation."""

    def stream_respond(self, message: str, history: list[dict[str, str]]) -> Iterator[str]:
        """Yield a response in chunks. Defaults to one chunk (non-streaming models)."""
        yield self.respond(message, history)
