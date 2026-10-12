"""Sidebar panels, each wired to a real backend. No placeholders."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from jarvis.config import Settings
from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore

logger = logging.getLogger(__name__)


class MemoryPanel(QWidget):
    """Live saved memories with search, refresh, and confirmed delete."""

    def __init__(self, orchestrator: Orchestrator, parent: Any = None) -> None:
        super().__init__(parent)
        self.orchestrator = orchestrator
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search memories…")
        self.search.textChanged.connect(lambda _text: self.refresh())
        layout.addWidget(self.search)

        self.listing = QListWidget(self)
        self.listing.setFrameStyle(QFrame.Shape.NoFrame)
        layout.addWidget(self.listing, stretch=1)

        row = QHBoxLayout()
        refresh = QPushButton("Refresh", self)
        refresh.clicked.connect(self.refresh)
        self.delete_button = QPushButton("Delete", self)
        self.delete_button.setObjectName("secondary")
        self.delete_button.clicked.connect(self.delete_selected)
        row.addWidget(refresh)
        row.addWidget(self.delete_button)
        layout.addLayout(row)
        self.refresh()

    def current_entries(self) -> list[dict[str, Any]]:
        query = self.search.text().strip()
        store = self.orchestrator.long_term
        entries = store.search(query) if query else store.list_entries()
        return [dict(entry) for entry in entries]

    def refresh(self) -> None:
        self.listing.clear()
        try:
            entries = self.current_entries()
        except Exception as exc:
            logger.warning("Memory refresh failed: %s", exc)
            self.listing.addItem(f"Could not load memories: {exc}")
            return
        if not entries:
            self.listing.addItem("No saved memories yet.")
            return
        for entry in entries:
            content = str(entry.get("content", ""))[:80]
            self.listing.addItem(f"#{entry['id']} [{entry['category']}] {content}")

    def delete_selected(self) -> None:
        row = self.listing.currentRow()
        entries = self.current_entries()
        if row < 0 or row >= len(entries):
            QMessageBox.information(self, "Memory", "Select a memory first.")
            return
        entry = entries[row]
        confirm = QMessageBox.question(
            self, "Delete memory",
            f"Delete #{entry['id']}: {entry['content']}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.orchestrator.long_term.delete(int(entry["id"]))
        self.refresh()


class HistoryPanel(QWidget):
    """Read-only conversation history with refresh."""

    def __init__(self, memory: ConversationStore, parent: Any = None) -> None:
        super().__init__(parent)
        self.memory = memory
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.view = QTextBrowser(self)
        self.view.setFrameStyle(QFrame.Shape.NoFrame)
        layout.addWidget(self.view, stretch=1)
        refresh = QPushButton("Refresh", self)
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        self.refresh()

    def refresh(self) -> None:
        try:
            messages = self.memory.get_messages()
        except Exception as exc:
            logger.warning("History refresh failed: %s", exc)
            self.view.setPlainText(f"Could not load history: {exc}")
            return
        if not messages:
            self.view.setPlainText("No conversation yet.")
            return
        lines = []
        for message in messages[-100:]:
            who = "You" if message["role"] == "user" else "Corneilius"
            lines.append(f"{who}: {message['content']}")
        self.view.setPlainText("\n\n".join(lines))


class ToolsPanel(QWidget):
    """Run real read-only tools and inspect their output."""

    def __init__(self, orchestrator: Orchestrator, run_tool, parent: Any = None) -> None:
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.run_tool = run_tool  # (name, args) -> runs on a worker, shows output
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.path_input = QLineEdit(self)
        self.path_input.setPlaceholderText("Workspace path (default: .)")
        layout.addWidget(self.path_input)

        row = QHBoxLayout()
        self.info_button = QPushButton("System info", self)
        self.info_button.clicked.connect(lambda: self.run_tool("system.info", {}))
        self.list_button = QPushButton("List workspace", self)
        self.list_button.setObjectName("secondary")
        self.list_button.clicked.connect(self.list_workspace)
        row.addWidget(self.info_button)
        row.addWidget(self.list_button)
        layout.addLayout(row)

        self.output = QTextBrowser(self)
        self.output.setObjectName("output")
        self.output.setFrameStyle(QFrame.Shape.NoFrame)
        self.output.setPlaceholderText("Tool output appears here.")
        layout.addWidget(self.output, stretch=1)

    def list_workspace(self) -> None:
        path = self.path_input.text().strip() or "."
        self.run_tool("workspace.list", {"path": path})


class SettingsPanel(QWidget):
    """Live configuration values. Editing needs an app restart."""

    def __init__(self, settings: Settings, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        form = QFormLayout()
        form.addRow("Provider:", QLabel(settings.model_provider, self))
        form.addRow("Model:", QLabel(settings.ollama_model, self))
        form.addRow("Ollama URL:", QLabel(settings.ollama_url, self))
        form.addRow("Data dir:", QLabel(str(settings.data_dir), self))
        form.addRow("Workspace:", QLabel(str(settings.workspace_root), self))
        form.addRow("Request timeout:", QLabel(f"{settings.request_timeout:g}s", self))
        form.addRow("Tool timeout:", QLabel(f"{settings.tool_timeout:g}s", self))
        form.addRow("Agent turns:", QLabel(str(settings.agent_max_iterations), self))
        layout.addLayout(form)
        note = QLabel("Change these via environment variables, then restart the app.", self)
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
