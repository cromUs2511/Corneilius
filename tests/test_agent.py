from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from jarvis.core.agent import parse_tool_request, verify_write
from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore
from jarvis.models.base import ModelAdapter
from jarvis.tools.audit import ToolAuditLog
from jarvis.tools.registry import READ_ONLY, ToolRegistry, ToolSpec, build_default_registry


def fence(tool: str, args: dict) -> str:
    import json
    return "```tool\n" + json.dumps({"tool": tool, "args": args}) + "\n```"


class ScriptModel(ModelAdapter):
    """Scripted model double that emits planned tool requests then a final answer."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def respond(self, message: str, history: list[dict[str, str]]) -> str:
        del message, history
        self.calls += 1
        if self.replies:
            return self.replies.pop(0)
        return "Done."


class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(__file__).parent / ".testdata-agent"
        self.temp.mkdir(exist_ok=True)
        self.db_path = self.temp / "agent.sqlite3"
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

    def test_multi_step_run_verifies_writes(self) -> None:
        model = ScriptModel([
            fence("workspace.write", {"path": "greet.txt", "content": "hello", "create_dirs": True}),
            fence("workspace.read", {"path": "greet.txt"}),
            "Created and confirmed greet.txt holds 'hello'.",
        ])
        assistant = self.make_assistant(model)
        answer = assistant.submit("/agent --allow-write create a greeting file")
        self.assertEqual(answer, "Created and confirmed greet.txt holds 'hello'.")
        self.assertNotIn("failed", answer)
        self.assertEqual((self.workspace / "greet.txt").read_text(encoding="utf-8"), "hello")

    def test_write_pauses_for_confirmation_then_resumes(self) -> None:
        model = ScriptModel([
            fence("workspace.write", {"path": "note.txt", "content": "remember this"}),
            "Saved note.txt.",
        ])
        assistant = self.make_assistant(model)
        paused = assistant.submit("/agent save a note")
        self.assertIn("yes", paused.lower())
        self.assertFalse((self.workspace / "note.txt").exists())
        done = assistant.submit("/confirm")
        self.assertEqual(done, "Saved note.txt.")
        self.assertEqual((self.workspace / "note.txt").read_text(encoding="utf-8"), "remember this")

    def test_max_iterations_caps_chatty_models(self) -> None:
        model = ScriptModel([fence("system.info", {})] * 50)
        assistant = self.make_assistant(model)
        answer = assistant.submit("/agent inspect the system repeatedly")
        self.assertIn("max iterations", answer)
        self.assertLessEqual(model.calls, 6)

    def test_cancel_before_run_aborts_without_acting(self) -> None:
        calls = {"n": 0}
        assistant_holder: dict[str, Any] = {}

        class CancellingModel(ModelAdapter):
            def respond(self, message: str, history: list[dict[str, str]]) -> str:
                del message, history
                calls["n"] += 1
                # Simulate the user pressing Cancel while the first step runs.
                assistant_holder["assistant"].cancel_agent()
                return fence("system.info", {})

        store = ConversationStore(self.db_path)
        assistant = Orchestrator(CancellingModel(), store)
        assistant.tools = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        assistant_holder["assistant"] = assistant
        answer = assistant.submit("/agent --allow-write do something")
        self.assertIn("cancelled", answer)
        self.assertEqual(calls["n"], 1)

    def test_transient_timeout_is_retried_once(self) -> None:
        state = {"calls": 0}

        def flaky(args: dict) -> dict:
            state["calls"] += 1
            if state["calls"] == 1:
                raise TimeoutError("timed out after 5s")
            return {"output": "steady now"}

        registry = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        registry.register(
            ToolSpec(name="test.flaky", description="flaky", risk=READ_ONLY, parameters={}), flaky
        )
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(ScriptModel([fence("test.flaky", {}), "Recovered."]), store)
        assistant.tools = registry
        answer = assistant.submit("/agent --allow-write try the flaky tool")
        self.assertEqual(answer, "Recovered.")
        self.assertEqual(state["calls"], 2)

    def test_failed_steps_are_reported_not_claimed(self) -> None:
        model = ScriptModel([
            fence("workspace.read", {"path": "missing.txt"}),
            "I could not read the file.",
        ])
        assistant = self.make_assistant(model)
        answer = assistant.submit("/agent read the missing file")
        self.assertIn("failed", answer)
        self.assertIn("Warning", answer)

    def test_parse_protocol(self) -> None:
        self.assertIsNone(parse_tool_request("Just a final answer."))
        name, args = parse_tool_request(fence("workspace.read", {"path": "a.txt"})) or ("", {})
        self.assertEqual(name, "workspace.read")
        self.assertEqual(args, {"path": "a.txt"})
        with self.assertRaises(ValueError):
            parse_tool_request("```tool\nnot json\n```")

    def test_parse_tolerates_single_quoted_requests(self) -> None:
        text = "```tool\n{'tool': 'workspace.read', 'args': {'path': 'a.txt'}}\n```"
        name, args = parse_tool_request(text) or ("", {})
        self.assertEqual(name, "workspace.read")
        self.assertEqual(args, {"path": "a.txt"})

    def test_prompts_prefer_knowledge_over_tools(self) -> None:
        from jarvis.core.agent import build_agent_system, build_chat_system
        chat = build_chat_system([], ["workspace.read [read-only]: R. Args: path (string, required)"])
        agent = build_agent_system("goal", ["workspace.read [read-only]: R. Args: path (string, required)"])
        for prompt in (chat, agent):
            self.assertIn("own knowledge", prompt)
            self.assertIn("double quotes", prompt)

    def test_verify_write_detects_tampering(self) -> None:
        registry = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        ok = registry.execute("workspace.write", {"path": "v.txt", "content": "original"}, confirmed=True)
        self.assertTrue(ok.ok, ok.error)
        verified, _note = verify_write(registry, "v.txt", "original")
        self.assertTrue(verified)
        (self.workspace / "v.txt").write_text("tampered", encoding="utf-8")
        verified_after, note = verify_write(registry, "v.txt", "original")
        self.assertFalse(verified_after)
        self.assertIn("mismatch", note)

    def test_saved_guidance_reaches_agent_runs(self) -> None:
        seen: list[list[dict[str, str]]] = []

        class CapturingModel(ModelAdapter):
            def respond(self, message: str, history: list[dict[str, str]]) -> str:
                del message
                seen.append(list(history))
                return "All done, following your guidance."

        assistant = self.make_assistant(CapturingModel())
        assistant.submit("/remember preference | Always greet with Ahoy")
        answer = assistant.submit("/agent --allow-write greet the crew")
        self.assertEqual(answer, "All done, following your guidance.")
        self.assertTrue(seen, "agent should call the model")
        self.assertIn("Ahoy", seen[0][0]["content"])

    def test_verbose_flag_shows_steps_on_request(self) -> None:
        model = ScriptModel([
            fence("system.info", {}),
            "Checked once.",
        ])
        assistant = self.make_assistant(model)
        plain = assistant.submit("/agent check once")
        self.assertEqual(plain, "Checked once.")
        model2 = ScriptModel([
            fence("system.info", {}),
            "Checked once.",
        ])
        assistant2 = self.make_assistant(model2)
        detailed = assistant2.submit("/agent --verbose check once")
        self.assertIn("Step 1", detailed)
        self.assertIn("Checked once.", detailed)

    def test_progress_hook_reports_each_step(self) -> None:
        updates: list[str] = []
        model = ScriptModel([
            fence("system.info", {}),
            fence("system.info", {}),
            "Checked twice.",
        ])
        assistant = self.make_assistant(model)
        assistant.on_progress = updates.append
        answer = assistant.submit("/agent --allow-write check twice")
        self.assertEqual(answer, "Checked twice.")
        joined = "\n".join(updates)
        self.assertIn("Agent started", joined)
        self.assertIn("Agent step 1", joined)
        self.assertIn("Agent step 2", joined)


if __name__ == "__main__":
    unittest.main()
