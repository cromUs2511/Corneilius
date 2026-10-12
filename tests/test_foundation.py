from __future__ import annotations

import unittest
from pathlib import Path

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore
from jarvis.memory.long_term import LongTermMemory
from jarvis.models.mock_model import MockModel


class FoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(__file__).parent / ".testdata"
        self.temp.mkdir(exist_ok=True)
        self.db_path = self.temp / "memory.sqlite3"
        self.db_path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_mock_conversation_is_persisted_and_reloaded(self) -> None:
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(MockModel(), store)
        self.assertEqual(assistant.submit("Testing one two three"), "I received: Testing one two three")
        reloaded = ConversationStore(self.db_path)
        self.assertEqual([m["role"] for m in reloaded.get_messages()], ["user", "assistant"])
        self.assertEqual(reloaded.get_messages()[1]["content"], "I received: Testing one two three")

    def test_mock_answers_greetings_help_and_time(self) -> None:
        store = ConversationStore(self.db_path)
        assistant = Orchestrator(MockModel(), store)
        self.assertIn("offline", assistant.submit("hello").lower())
        self.assertIn("remember", assistant.submit("what can you do?").lower())
        self.assertIn("20", assistant.submit("what is the time?"))
        self.assertEqual(assistant.submit("Testing one two three"), "I received: Testing one two three")

    def test_blank_message_is_rejected_without_saving(self) -> None:
        store = ConversationStore(self.db_path)
        with self.assertRaises(ValueError):
            Orchestrator(MockModel(), store).submit("   ")
        self.assertEqual(store.get_messages(), [])

    def test_store_rejects_unknown_role(self) -> None:
        store = ConversationStore(self.db_path)
        with self.assertRaises(ValueError):
            store.add_message("system", "should not be stored")

    def test_long_term_memory_is_explicit_searchable_editable_and_persistent(self) -> None:
        memory = LongTermMemory(self.db_path)
        fact_id = memory.save("fact", "My cat is named Pixel")
        memory.save("preference", "I prefer concise answers")

        reloaded = LongTermMemory(self.db_path)
        self.assertEqual(reloaded.search("Pixel")[0]["id"], fact_id)
        self.assertEqual(len(reloaded.list_entries("preference")), 1)
        reloaded.update(fact_id, "My cat is named Miso")
        self.assertEqual(reloaded.search("Pixel"), [])
        self.assertEqual(reloaded.search("Miso")[0]["content"], "My cat is named Miso")
        self.assertTrue(reloaded.delete(fact_id))
        self.assertEqual(reloaded.list_entries("fact"), [])

    def test_memory_rejects_invalid_category_and_does_not_mix_categories(self) -> None:
        memory = LongTermMemory(self.db_path)
        memory.save("project", "JARVIS prototype notes")
        self.assertEqual(memory.search("JARVIS", category="project")[0]["category"], "project")
        self.assertEqual(memory.search("JARVIS", category="fact"), [])
        with self.assertRaises(ValueError):
            memory.save("secret", "never accepted")

    def test_chat_memory_commands_save_list_search_edit_and_delete(self) -> None:
        memory = ConversationStore(self.db_path)
        assistant = Orchestrator(MockModel(), memory)
        self.assertIn("Saved memory #", assistant.submit("/remember preference | Use short answers"))
        saved_id = LongTermMemory(self.db_path).list_entries()[0]["id"]
        self.assertIn("Use short answers", assistant.submit("/memory"))
        self.assertIn("Use short answers", assistant.submit("/memory search short"))
        self.assertIn("Updated memory", assistant.submit(f"/memory edit {saved_id} | Use bullet points"))
        self.assertIn("Deleted memory", assistant.submit(f"/forget {saved_id}"))
        self.assertEqual(LongTermMemory(self.db_path).list_entries(), [])


if __name__ == "__main__":
    unittest.main()
