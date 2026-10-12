from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore
from jarvis.models.mock_model import MockModel
from jarvis.tools.audit import ToolAuditLog
from jarvis.tools.registry import build_default_registry

HAS_GIT = shutil.which("git") is not None


class CodeToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(__file__).parent / ".testdata-code"
        self.temp.mkdir(exist_ok=True)
        self.db_path = self.temp / "code.sqlite3"
        self.db_path.unlink(missing_ok=True)
        self.workspace = self.temp / "workspace"
        shutil.rmtree(self.workspace, ignore_errors=True)
        (self.workspace / "pkg").mkdir(parents=True, exist_ok=True)
        (self.workspace / "pkg" / "mod.py").write_text(
            "def greet():\n    return 'hello'\n", encoding="utf-8"
        )
        (self.workspace / "pkg" / "test_mod.py").write_text(
            "import unittest\nfrom mod import greet\n\n"
            "class T(unittest.TestCase):\n    def test_greet(self):\n"
            "        self.assertEqual(greet(), 'hello')\n",
            encoding="utf-8",
        )
        (self.workspace / "blob.bin").write_bytes(b"\x00\x01binary")
        self.registry = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 30.0)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_search_finds_code_and_skips_binary(self) -> None:
        result = self.registry.execute("repo.search", {"pattern": "def greet"})
        self.assertTrue(result.ok, result.error)
        self.assertIn("mod.py:1", result.output)
        self.assertNotIn("blob.bin", result.output)
        self.assertEqual(result.data["matches"], 1)

    def test_search_rejects_bad_patterns_and_escapes(self) -> None:
        bad = self.registry.execute("repo.search", {"pattern": "([invalid"})
        self.assertFalse(bad.ok)
        outside = self.registry.execute("repo.search", {"pattern": "x", "path": ".."})
        self.assertFalse(outside.ok)

    def test_edit_requires_unique_match(self) -> None:
        ok = self.registry.execute(
            "repo.edit",
            {"path": "pkg/mod.py", "old": "return 'hello'", "new": "return 'hi'"},
            confirmed=True,
        )
        self.assertTrue(ok.ok, ok.error)
        self.assertIn("line 2", ok.output)
        self.assertIn("return 'hi'", (self.workspace / "pkg" / "mod.py").read_text(encoding="utf-8"))
        missing = self.registry.execute(
            "repo.edit", {"path": "pkg/mod.py", "old": "nope", "new": "x"}, confirmed=True
        )
        self.assertFalse(missing.ok)
        (self.workspace / "pkg" / "dup.py").write_text("a = 1\na = 1\n", encoding="utf-8")
        ambiguous = self.registry.execute(
            "repo.edit", {"path": "pkg/dup.py", "old": "a = 1", "new": "a = 2"}, confirmed=True
        )
        self.assertFalse(ambiguous.ok)
        self.assertIn("2 times", ambiguous.error)
        self.assertIn("a = 1\na = 1", (self.workspace / "pkg" / "dup.py").read_text(encoding="utf-8"))

    def test_edit_needs_confirmation(self) -> None:
        result = self.registry.execute(
            "repo.edit", {"path": "pkg/mod.py", "old": "x", "new": "y"}
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.confirmation_required)

    def test_run_tests_executes_fixed_suite(self) -> None:
        result = self.registry.execute("repo.test", {"path": "pkg"}, confirmed=True)
        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.data["passed"])
        self.assertIn("OK", result.output)

    def test_run_tests_reports_failures_honestly(self) -> None:
        (self.workspace / "pkg" / "test_mod.py").write_text(
            "import unittest\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
            "        self.assertEqual(1, 2)\n",
            encoding="utf-8",
        )
        result = self.registry.execute("repo.test", {"path": "pkg"}, confirmed=True)
        self.assertTrue(result.ok, result.error)  # the tool ran fine
        self.assertFalse(result.data["passed"])
        self.assertIn("FAILED", result.output)

    def test_run_tests_needs_confirmation(self) -> None:
        result = self.registry.execute("repo.test", {"path": "pkg"})
        self.assertFalse(result.ok)
        self.assertTrue(result.confirmation_required)

    def test_orchestrator_repairs_with_tests(self) -> None:
        from jarvis.tests.test_agent import ScriptModel, fence

        (self.workspace / "pkg" / "mod.py").write_text(
            "def greet():\n    return 'bye'\n", encoding="utf-8"
        )
        model = ScriptModel([
            fence("repo.test", {"path": "pkg"}),
            fence("repo.edit", {"path": "pkg/mod.py", "old": "return 'bye'", "new": "return 'hello'"}),
            fence("repo.test", {"path": "pkg"}),
            "Fixed greet and tests pass.",
        ])
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(MockModel(), store)
        assistant.tools = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 30.0)
        paused = assistant.submit("/agent --allow-write fix greet")
        self.assertIn("yes", paused.lower())  # repo.test always needs explicit approval
        answer = assistant.submit("yes")
        self.assertIn("tests pass", answer)
        self.assertIn("return 'hello'", (self.workspace / "pkg" / "mod.py").read_text(encoding="utf-8"))

    @unittest.skipUnless(HAS_GIT, "git is not installed")
    def test_git_status_and_diff(self) -> None:
        repo = self.workspace / "gitproj"
        repo.mkdir(exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
        (repo / "a.txt").write_text("one\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
        clean = self.registry.execute("repo.git_status", {"path": "gitproj"})
        self.assertTrue(clean.ok, clean.error)
        self.assertTrue(clean.data["clean"])
        (repo / "a.txt").write_text("two\n", encoding="utf-8")
        dirty = self.registry.execute("repo.git_status", {"path": "gitproj"})
        self.assertTrue(dirty.ok, dirty.error)
        self.assertFalse(dirty.data["clean"])
        self.assertIn("a.txt", dirty.output)
        diff = self.registry.execute("repo.git_diff", {"path": "gitproj"})
        self.assertTrue(diff.ok, diff.error)
        self.assertIn("a.txt", diff.output)

    def test_git_tools_report_non_repos(self) -> None:
        status = self.registry.execute("repo.git_status", {"path": "pkg"})
        self.assertFalse(status.ok)
        self.assertIn("Not a git repository", status.error)

    def test_no_destructive_git_tools_exist(self) -> None:
        names = [spec.name for spec in self.registry.specs()]
        for forbidden in ("commit", "push", "checkout", "reset", "shell", "exec"):
            self.assertFalse(any(forbidden in name for name in names), names)


if __name__ == "__main__":
    unittest.main()
