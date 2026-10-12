from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore
from jarvis.memory.skills import SkillStore
from jarvis.memory.tasks import TaskRunStore
from jarvis.models.mock_model import MockModel
from jarvis.tests.test_agent import ScriptModel, fence
from jarvis.tools.audit import ToolAuditLog
from jarvis.tools.registry import build_default_registry


class TaskSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(__file__).parent / ".testdata-tasks"
        self.temp.mkdir(exist_ok=True)
        self.db_path = self.temp / "tasks.sqlite3"
        self.db_path.unlink(missing_ok=True)
        self.workspace = self.temp / "workspace"
        shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def make_assistant(self, model) -> Orchestrator:  # type: ignore[no-untyped-def]
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(model, store)
        assistant.tools = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        return assistant

    def test_task_run_pause_and_resume_roundtrip(self) -> None:
        store = TaskRunStore(self.db_path)
        task_id = store.create("write a note", {"allow_write": True})
        steps = [{"index": 1, "tool": "workspace.write", "args": {"path": "n.txt"}}]
        store.save_pause(task_id, steps, {"tool": "workspace.write", "args": {"path": "n.txt"}})
        saved = store.load_resumable(task_id)
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(saved["status"], "paused")
        self.assertEqual(saved["steps"][0]["tool"], "workspace.write")
        store.finish(task_id, "finished", steps, "done")
        self.assertIsNone(store.load_resumable(task_id))
        self.assertEqual(len(store.list_recent()), 1)

    def test_skill_draft_test_enable_gating(self) -> None:
        skills = SkillStore(self.db_path)
        registry = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        workflow = [{"tool": "system.info", "args": {}}]
        _, v1 = skills.save_draft("syscheck", workflow, source_task_id=1)
        self.assertEqual(v1, 1)
        # Drafts cannot be enabled directly.
        with self.assertRaises(ValueError):
            skills.enable("syscheck")
        # Drafts accept structurally valid workflows; unknown tools are caught
        # at test time against the live registry (never executed).
        bad_id, _ = skills.save_draft("bad", [{"tool": "nope.tool", "args": {}}])
        self.assertGreater(bad_id, 0)
        self.assertIsNotNone(SkillStore.validate_workflow([{"tool": "nope.tool", "args": {}}], registry))
        self.assertIsNone(SkillStore.validate_workflow(workflow, registry))
        skills.mark_tested("syscheck", 1)
        skills.enable("syscheck")
        self.assertEqual(skills.get_latest("syscheck")["status"], "enabled")
        # Saving again versions up.
        _, v2 = skills.save_draft("syscheck", workflow)
        self.assertEqual(v2, 2)

    def test_memory_update_records_correction(self) -> None:
        from jarvis.memory.long_term import LongTermMemory

        memory = LongTermMemory(self.db_path)
        entry_id = memory.save("fact", "The cat is named Pixle")
        memory.update(entry_id, "The cat is named Pixel")
        corrections = SkillStore(self.db_path).list_corrections()
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["entry_id"], entry_id)
        self.assertIn("Pixle", corrections[0]["old_content"])

    def test_agent_pause_persists_and_resumes_after_restart(self) -> None:
        model = ScriptModel([
            fence("workspace.write", {"path": "note.txt", "content": "hi"}),
            "Saved note.txt.",
        ])
        assistant = self.make_assistant(model)
        paused = assistant.submit("/agent save a note")
        self.assertIn("task #", paused)
        task_id = int(assistant._last_task_id or 0)
        self.assertGreater(task_id, 0)
        self.assertIn("task", assistant.submit("/tasks").lower())
        # Simulate a restart: a fresh orchestrator on the same database.
        # The pending write was already approved by "yes"; the new instance's
        # model only supplies the final summary.
        store2 = ConversationStore(self.db_path)
        assistant2 = Orchestrator(ScriptModel(["Saved note.txt."]), store2)
        assistant2.tools = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        resumed = assistant2.submit(f"/resume {task_id}")
        self.assertIn("yes", resumed.lower())
        done = assistant2.submit("yes")
        self.assertEqual(done, "Saved note.txt.")
        self.assertEqual((self.workspace / "note.txt").read_text(encoding="utf-8"), "hi")

    def test_skill_save_test_enable_flow(self) -> None:
        model = ScriptModel([
            fence("workspace.write", {"path": "greet.txt", "content": "hello", "create_dirs": True}),
            "Created greet.txt.",
        ])
        assistant = self.make_assistant(model)
        assistant.submit("/agent --allow-write create a greeting file")
        saved = assistant.submit("/skill save greet-file")
        self.assertIn("draft", saved)
        tested = assistant.submit("/skill test greet-file")
        self.assertIn("tested", tested)
        enabled = assistant.submit("/skill enable greet-file")
        self.assertIn("enabled", enabled)
        self.assertIn("greet-file", assistant.submit("/skills"))

    def test_skill_save_refuses_without_success(self) -> None:
        assistant = self.make_assistant(MockModel())
        self.assertIn("No successful steps", assistant.submit("/skill save empty-skill"))


if __name__ == "__main__":
    unittest.main()
