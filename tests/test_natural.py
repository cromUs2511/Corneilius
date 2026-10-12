from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from jarvis.core.natural import (
    is_affirmative,
    is_negative,
    parse_forget,
    parse_recall,
    parse_remember,
)
from jarvis.memory.database import ConversationStore
from jarvis.memory.long_term import LongTermMemory
from jarvis.models.mock_model import MockModel
from jarvis.tools.audit import ToolAuditLog
from jarvis.tools.registry import build_default_registry
from jarvis.core.orchestrator import Orchestrator
from jarvis.tests.test_agent import ScriptModel, fence


class NaturalLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(__file__).parent / ".testdata-natural"
        self.temp.mkdir(exist_ok=True)
        self.db_path = self.temp / "natural.sqlite3"
        self.db_path.unlink(missing_ok=True)
        self.workspace = self.temp / "workspace"
        shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def make_assistant(self, model) -> Orchestrator:
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(model, store)
        assistant.tools = build_default_registry(ToolAuditLog(self.db_path), self.workspace, 5.0)
        return assistant

    def test_plain_chat_is_unchanged_without_commands(self) -> None:
        assistant = self.make_assistant(ScriptModel(["Just chatting, no tools needed."]))
        self.assertEqual(assistant.submit("hello there"), "Just chatting, no tools needed.")

    def test_mock_echo_still_passes_through_verbatim(self) -> None:
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(MockModel(), store)
        self.assertEqual(assistant.submit("Testing one two three"), "I received: Testing one two three")

    def test_natural_tool_use_without_commands(self) -> None:
        (self.workspace / "todo.txt").write_text("buy milk", encoding="utf-8")
        model = ScriptModel([
            fence("workspace.read", {"path": "todo.txt"}),
            "Your todo says: buy milk.",
        ])
        assistant = self.make_assistant(model)
        answer = assistant.submit("what does my todo say?")
        self.assertEqual(answer, "Your todo says: buy milk.")

    def test_natural_write_pauses_and_yes_resumes(self) -> None:
        model = ScriptModel([
            fence("workspace.write", {"path": "hi.txt", "content": "hi"}),
            "Wrote hi.txt.",
        ])
        assistant = self.make_assistant(model)
        paused = assistant.submit("write hi.txt containing hi")
        self.assertIn("yes", paused.lower())
        self.assertFalse((self.workspace / "hi.txt").exists())
        done = assistant.submit("yes")
        self.assertEqual(done, "Wrote hi.txt.")
        self.assertEqual((self.workspace / "hi.txt").read_text(encoding="utf-8"), "hi")

    def test_natural_no_aborts_paused_run(self) -> None:
        model = ScriptModel([fence("workspace.write", {"path": "x.txt", "content": "x"})])
        assistant = self.make_assistant(model)
        assistant.submit("write x.txt")
        answer = assistant.submit("no, never mind")
        self.assertIn("dropped", answer.lower())
        self.assertFalse((self.workspace / "x.txt").exists())

    def test_remember_intent_saves_without_command(self) -> None:
        assistant = self.make_assistant(ScriptModel([]))
        answer = assistant.submit("remember that my cat is named Pixel")
        self.assertIn("saved as fact", answer)
        entries = LongTermMemory(self.db_path).list_entries()
        self.assertEqual(len(entries), 1)
        self.assertIn("Pixel", entries[0]["content"])

    def test_preference_wording_saves_as_preference(self) -> None:
        assistant = self.make_assistant(ScriptModel([]))
        answer = assistant.submit("remember that I prefer short answers")
        self.assertIn("preference", answer)

    def test_ordinary_statements_are_never_auto_saved(self) -> None:
        assistant = self.make_assistant(ScriptModel(["Cute!"]))
        assistant.submit("My cat is named Pixel")
        assistant.submit("I remember when we first met")
        self.assertEqual(LongTermMemory(self.db_path).list_entries(), [])

    def test_recall_intents(self) -> None:
        assistant = self.make_assistant(ScriptModel([]))
        assistant.submit("remember that my cat is named Pixel")
        self.assertIn("Pixel", assistant.submit("what do you remember about my cat"))
        self.assertIn("Pixel", assistant.submit("do you remember my cat?"))
        self.assertIn("Pixel", assistant.submit("what do you remember?"))
        self.assertIn("don't have anything saved", assistant.submit("what do you remember about dogs?"))

    def test_forget_unique_match_deletes(self) -> None:
        assistant = self.make_assistant(ScriptModel([]))
        assistant.submit("remember that my cat is named Pixel")
        answer = assistant.submit("forget my cat")
        self.assertIn("Forgot", answer)
        self.assertEqual(LongTermMemory(self.db_path).list_entries(), [])

    def test_forget_ambiguous_deletes_nothing(self) -> None:
        assistant = self.make_assistant(ScriptModel([]))
        assistant.submit("remember that Pixel likes naps")
        assistant.submit("remember that Pixel likes tuna")
        answer = assistant.submit("forget Pixel")
        self.assertIn("deleted nothing", answer)
        self.assertEqual(len(LongTermMemory(self.db_path).list_entries()), 2)

    def test_first_call_model_error_still_raises(self) -> None:
        class DownModel(ScriptModel):
            def respond(self, message, history):
                raise RuntimeError("Could not reach Ollama")

        assistant = self.make_assistant(DownModel([]))
        with self.assertRaises(RuntimeError):
            assistant.submit("hello?")

    def test_intent_parsing(self) -> None:
        self.assertEqual(parse_remember("remember that my cat is Pixel"), "my cat is Pixel")
        self.assertEqual(parse_remember("Please remember: call mom"), "call mom")
        self.assertEqual(parse_remember("don't forget to water plants"), "to water plants")
        self.assertIsNone(parse_remember("I remember when we met"))
        self.assertIsNone(parse_remember("remember"))
        self.assertEqual(parse_recall("what do you remember?"), ("all", None))
        self.assertEqual(parse_recall("do you remember my cat?"), ("about", "my cat"))
        self.assertIsNone(parse_recall("hello"))
        self.assertEqual(parse_forget("forget my cat"), "my cat")
        self.assertIsNone(parse_forget("I forget names easily"))
        self.assertTrue(is_affirmative("yes"))
        self.assertTrue(is_affirmative(" Yes, please! ") or is_affirmative("go ahead"))
        self.assertTrue(is_negative("no"))
        self.assertTrue(is_negative("never mind"))
        self.assertFalse(is_affirmative("yes, but later"))
        self.assertFalse(is_negative("not really"))


if __name__ == "__main__":
    unittest.main()
