from __future__ import annotations

import time
import unittest
from pathlib import Path

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore
from jarvis.models.mock_model import MockModel
from jarvis.tools.audit import ToolAuditLog
from jarvis.tools.registry import ToolRegistry, ToolSpec, build_default_registry
from jarvis.tools.registry import READ_ONLY


class SafeToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(__file__).parent / ".testdata-tools"
        self.temp.mkdir(exist_ok=True)
        self.db_path = self.temp / "tools.sqlite3"
        self.db_path.unlink(missing_ok=True)
        self.workspace = self.temp / "workspace"
        import shutil
        shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.registry = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)
        for child in sorted(self.workspace.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass

    def test_workspace_write_read_list_roundtrip(self) -> None:
        needs_confirm = self.registry.execute("workspace.write", {"path": "notes/hello.txt", "content": "hi"})
        self.assertFalse(needs_confirm.ok)
        self.assertTrue(needs_confirm.confirmation_required)
        created = self.registry.execute(
            "workspace.write", {"path": "notes/hello.txt", "content": "hi", "create_dirs": True}, confirmed=True
        )
        self.assertTrue(created.ok, created.error)
        self.assertIn("Created", created.output)
        read = self.registry.execute("workspace.read", {"path": "notes/hello.txt"})
        self.assertTrue(read.ok, read.error)
        self.assertEqual(read.output, "hi")
        listing = self.registry.execute("workspace.list", {"path": "notes"})
        self.assertTrue(listing.ok, listing.error)
        self.assertIn("hello.txt", listing.output)

    def test_path_escape_is_rejected(self) -> None:
        for tool in ("workspace.read", "workspace.list"):
            result = self.registry.execute(tool, {"path": "../outside.txt"})
            self.assertFalse(result.ok)
            self.assertIn("workspace", result.error)
        write = self.registry.execute(
            "workspace.write", {"path": "../../evil.txt", "content": "x"}, confirmed=True
        )
        self.assertFalse(write.ok)
        self.assertNotExists(Path(self.temp) / "evil.txt")

    def assertNotExists(self, path: Path) -> None:
        self.assertFalse(path.exists(), f"unexpected file created: {path}")

    def test_validation_rejects_bad_args(self) -> None:
        missing = self.registry.execute("workspace.read", {})
        self.assertFalse(missing.ok)
        self.assertIn("Missing required", missing.error)
        wrong_type = self.registry.execute("workspace.read", {"path": 123})
        self.assertFalse(wrong_type.ok)
        unknown_tool = self.registry.execute("nope.tool", {})
        self.assertFalse(unknown_tool.ok)
        unknown_arg = self.registry.execute("system.info", {"extra": 1})
        self.assertFalse(unknown_arg.ok)

    def test_timeout_returns_structured_failure(self) -> None:
        def slow(args: dict) -> dict:
            time.sleep(2.0)
            return {"output": "too late"}

        self.registry.register(
            ToolSpec(name="test.slow", description="slow", risk=READ_ONLY, parameters={}), slow
        )
        result = self.registry.execute("test.slow", {}, timeout=0.5)
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.error)

    def test_audit_records_do_not_store_full_content(self) -> None:
        big = "sensitive-content-" * 50
        result = self.registry.execute(
            "workspace.write", {"path": "secret.txt", "content": big}, confirmed=True
        )
        self.assertTrue(result.ok, result.error)
        entries = ToolAuditLog(self.db_path).recent(5)
        self.assertGreaterEqual(len(entries), 1)
        self.assertEqual(entries[0]["tool"], "workspace.write")
        self.assertNotIn(big, entries[0]["args"])
        self.assertIn("truncated", entries[0]["args"])

    def test_system_info_is_read_only_and_structured(self) -> None:
        result = self.registry.execute("system.info", {})
        self.assertTrue(result.ok, result.error)
        self.assertIn("Python", result.output)
        self.assertIn("python", result.data)

    def test_orchestrator_tool_commands_with_confirmation(self) -> None:
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(MockModel(), store)
        # Point the orchestrator's workspace at the test directory.
        assistant.tools = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        self.assertIn("workspace.list", assistant.submit("/tools"))
        answer = assistant.submit('{"path": ".", "recursive": false}'.join(["/tool workspace.list ", ""]))
        self.assertIn("succeeded", answer)
        pending = assistant.submit('/tool workspace.write {"path": "a.txt", "content": "hello"}')
        self.assertIn("yes", pending.lower())
        self.assertFalse((self.workspace / "a.txt").exists())
        done = assistant.submit("/confirm")
        self.assertIn("succeeded", done)
        self.assertEqual((self.workspace / "a.txt").read_text(encoding="utf-8"), "hello")
        audit = assistant.submit("/audit 5")
        self.assertIn("workspace.write", audit)
        # Failed validation is reported, not claimed as success.
        failed = assistant.submit('/tool workspace.read {"path": "missing.txt"}')
        self.assertIn("failed", failed)

    def test_orchestrator_cancel_discards_pending_write(self) -> None:
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(MockModel(), store)
        assistant.tools = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        assistant.submit('/tool workspace.write {"path": "b.txt", "content": "hello"}')
        self.assertIn("cancelled", assistant.submit("/cancel").lower())
        self.assertFalse((self.workspace / "b.txt").exists())

    def test_no_shell_tool_is_exposed(self) -> None:
        names = [spec.name for spec in self.registry.specs()]
        self.assertNotIn("shell.exec", names)
        self.assertNotIn("shell", "".join(names))
        for spec in self.registry.specs():
            self.assertNotEqual(spec.risk, "unrestricted")


if __name__ == "__main__":
    unittest.main()
