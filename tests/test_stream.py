from __future__ import annotations

import shutil
import unittest
from collections.abc import Iterator
from pathlib import Path

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore
from jarvis.models.base import ModelAdapter
from jarvis.tools.audit import ToolAuditLog
from jarvis.tools.registry import build_default_registry
from jarvis.tests.test_agent import fence


class ChunkModel(ModelAdapter):
    """Model double with controllable streamed chunks and full replies."""

    def __init__(self, chunks: list[str], full: str | None = None) -> None:
        self.chunks = list(chunks)
        self.full = full if full is not None else "".join(chunks)

    def respond(self, message: str, history: list[dict[str, str]]) -> str:
        del message, history
        return self.full

    def stream_respond(self, message: str, history: list[dict[str, str]]) -> Iterator[str]:
        del message, history
        yield from self.chunks


class StreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(__file__).parent / ".testdata-stream"
        self.temp.mkdir(exist_ok=True)
        self.db_path = self.temp / "stream.sqlite3"
        self.db_path.unlink(missing_ok=True)
        self.workspace = self.temp / "workspace"
        shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def make_assistant(self, model: ModelAdapter) -> Orchestrator:
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(model, store)
        assistant.tools = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        return assistant

    def test_plain_chat_streams_chunks_and_stores(self) -> None:
        assistant = self.make_assistant(ChunkModel(["Hel", "lo", " there"]))
        chunks = list(assistant.submit_stream("hi"))
        self.assertEqual(chunks, ["Hel", "lo", " there"])
        self.assertEqual(assistant._last_final, "Hello there")
        self.assertFalse(assistant._last_replaced)
        stored = ConversationStore(self.db_path).get_messages()
        self.assertEqual(stored[-1]["content"], "Hello there")

    def test_tool_path_marks_replaced(self) -> None:
        (self.workspace / "todo.txt").write_text("buy milk", encoding="utf-8")
        model = ChunkModel(
            [fence("workspace.read", {"path": "todo.txt"})],
            full=fence("workspace.read", {"path": "todo.txt"}),
        )
        assistant = self.make_assistant(model)
        chunks = list(assistant.submit_stream("what does my todo say?"))
        self.assertTrue(assistant._last_replaced)
        self.assertIn("buy milk", assistant._last_final)
        self.assertTrue(any("buy milk" in chunk for chunk in chunks))

    def test_commands_yield_once(self) -> None:
        assistant = self.make_assistant(ChunkModel(["unused"]))
        chunks = list(assistant.submit_stream("remember that my cat is Pixel"))
        self.assertEqual(len(chunks), 1)
        self.assertIn("saved as fact", chunks[0])
        self.assertFalse(assistant._last_replaced)

    def test_empty_stream_raises(self) -> None:
        assistant = self.make_assistant(ChunkModel([]))
        with self.assertRaises(RuntimeError):
            list(assistant.submit_stream("hi"))

    def test_cancel_mid_stream_stops(self) -> None:
        def slow_stream() -> Iterator[str]:
            yield "part one "
            assistant_holder["assistant"].cancel_agent()
            yield "part two (should never arrive)"

        class SlowModel(ModelAdapter):
            def respond(self, message: str, history: list[dict[str, str]]) -> str:
                return "part one part two"

            def stream_respond(self, message: str, history: list[dict[str, str]]) -> Iterator[str]:
                del message, history
                return slow_stream()

        assistant_holder: dict = {}
        assistant = self.make_assistant(SlowModel())
        assistant_holder["assistant"] = assistant
        chunks = list(assistant.submit_stream("go slow"))
        self.assertEqual(chunks, ["part one "])
        self.assertTrue(assistant._agent_cancel.is_set())

    def test_submit_still_behaves_after_refactor(self) -> None:
        assistant = self.make_assistant(ChunkModel(["all at once"]))
        self.assertEqual(assistant.submit("hi"), "all at once")
        stored = ConversationStore(self.db_path).get_messages()
        self.assertEqual([m["role"] for m in stored], ["user", "assistant"])
        self.assertEqual(stored[-1]["content"], "all at once")


if __name__ == "__main__":
    unittest.main()
