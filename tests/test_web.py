from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore
from jarvis.models.base import ModelAdapter
from jarvis.tools.audit import ToolAuditLog
from jarvis.tools.registry import build_default_registry
from jarvis.tools.web import ensure_public_http
from jarvis.tests.test_agent import ScriptModel, fence

DDG_FIXTURE = (
    '<a class="result__a" href="https://example.com/spidey">First Amazing Spider-Man</a>'
    '<a class="result__snippet" href="x">2012 film starring Andrew Garfield</a>'
    '<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo">Second</a>'
    '<a class="result__snippet" href="x">more info</a>'
)

PAGE_FIXTURE = (
    "<html><head><title>T</title><style>.x{color:red}</style></head>"
    "<body><script>evil()</script><h1>Spidey</h1><p>The film came out in 2012.</p></body></html>"
)


class FakeHeaders:
    def __init__(self, content_type: str = "text/html") -> None:
        self._content_type = content_type

    def get_content_type(self) -> str:
        return self._content_type


class FakeResponse:
    def __init__(self, data: bytes, content_type: str = "text/html") -> None:
        self._data = data
        self.headers = FakeHeaders(content_type)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._data if size is None or size < 0 else self._data[:size]


# example.com, documentation range — keeps fetch tests offline and deterministic.
FAKE_ADDRINFO = [(2, 1, 6, "", ("93.184.216.34", 443))]


class WebToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(__file__).parent / ".testdata-web"
        self.temp.mkdir(exist_ok=True)
        self.db_path = self.temp / "web.sqlite3"
        self.db_path.unlink(missing_ok=True)
        self.workspace = self.temp / "workspace"
        shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.registry = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def make_assistant(self, model: ModelAdapter) -> Orchestrator:
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(model, store)
        assistant.tools = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        return assistant

    def test_search_parses_results(self) -> None:
        with patch("urllib.request.urlopen", return_value=FakeResponse(DDG_FIXTURE.encode())):
            result = self.registry.execute("web.search", {"query": "spiderman"}, confirmed=True)
        self.assertTrue(result.ok, result.error)
        self.assertIn("First Amazing Spider-Man", result.output)
        self.assertIn("https://example.com/two", result.output)
        self.assertEqual(len(result.data["results"]), 2)

    def test_search_needs_confirmation(self) -> None:
        result = self.registry.execute("web.search", {"query": "spiderman"})
        self.assertFalse(result.ok)
        self.assertTrue(result.confirmation_required)

    def test_fetch_strips_tags_and_caps(self) -> None:
        with patch("urllib.request.urlopen", return_value=FakeResponse(PAGE_FIXTURE.encode())), \
             patch("socket.getaddrinfo", return_value=FAKE_ADDRINFO):
            result = self.registry.execute(
                "web.fetch", {"url": "https://example.com/t", "max_chars": 20}, confirmed=True
            )
        self.assertTrue(result.ok, result.error)
        self.assertNotIn("evil()", result.output)
        self.assertNotIn("<h1>", result.output)
        self.assertIn("Spidey", result.output)
        self.assertTrue(result.data["truncated"])
        self.assertEqual(result.data["size_chars"], 37)

    def test_fetch_rejects_local_and_private_hosts(self) -> None:
        for bad in ("file:///etc/passwd", "http://localhost/x", "http://127.0.0.1/x",
                    "http://192.168.1.1/x", "http://10.0.0.5/x", "ftp://example.com/x"):
            with patch("urllib.request.urlopen", return_value=FakeResponse(b"nope")):
                result = self.registry.execute("web.fetch", {"url": bad}, confirmed=True)
            self.assertFalse(result.ok, bad)
        with self.assertRaises(ValueError):
            ensure_public_http("http://169.254.169.254/latest")

    def test_fetch_rejects_non_text(self) -> None:
        fake = FakeResponse(b"\x89PNG", content_type="image/png")
        with patch("urllib.request.urlopen", return_value=fake), \
             patch("socket.getaddrinfo", return_value=FAKE_ADDRINFO):
            result = self.registry.execute("web.fetch", {"url": "https://example.com/i.png"}, confirmed=True)
        self.assertFalse(result.ok)
        self.assertIn("non-text", result.error)

    def test_agent_pauses_for_web_without_flag(self) -> None:
        model = ScriptModel([fence("web.search", {"query": "spiderman"})])
        assistant = self.make_assistant(model)
        paused = assistant.submit("/agent when was the first spiderman movie")
        self.assertIn("yes", paused.lower())
        self.assertIn("needs approval", paused)

    def test_agent_allow_web_runs_search(self) -> None:
        model = ScriptModel([
            fence("web.search", {"query": "spiderman"}),
            "The Amazing Spider-Man came out in 2012.",
        ])
        assistant = self.make_assistant(model)
        with patch("urllib.request.urlopen", return_value=FakeResponse(DDG_FIXTURE.encode())):
            answer = assistant.submit("/agent --allow-web when was the first spiderman movie")
        self.assertEqual(answer, "The Amazing Spider-Man came out in 2012.")

    def test_yes_resumes_web_step(self) -> None:
        model = ScriptModel([
            fence("web.search", {"query": "spiderman"}),
            "Found it: 2012.",
        ])
        assistant = self.make_assistant(model)
        with patch("urllib.request.urlopen", return_value=FakeResponse(DDG_FIXTURE.encode())):
            assistant.submit("/agent when was the first spiderman movie")
            done = assistant.submit("yes")
        self.assertEqual(done, "Found it: 2012.")


if __name__ == "__main__":
    unittest.main()
