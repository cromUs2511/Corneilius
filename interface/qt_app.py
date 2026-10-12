"""Corneilius Qt main window: minimalist chat with a collapsible backend-wired sidebar."""

from __future__ import annotations

import html
import logging
from typing import Any

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextBrowser,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from jarvis.config import Settings
from jarvis.core.orchestrator import Orchestrator
from jarvis.interface.qt_panels import HistoryPanel, MemoryPanel, SettingsPanel, ToolsPanel
from jarvis.interface.qt_style import ASSISTANT_NAME, STATUS_STYLES, STYLESHEET, TEXT, USER_NAME
from jarvis.interface.qt_work import ChatInput, Worker
from jarvis.memory.database import ConversationStore

logger = logging.getLogger(__name__)

SIDEBAR_WIDTH = 300


class CorneiliusWindow(QMainWindow):
    def __init__(
        self,
        orchestrator: Orchestrator,
        memory: ConversationStore,
        settings: Settings,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.memory = memory
        self.settings = settings
        self.worker: Worker | None = None
        self.busy = False
        self.setWindowTitle("Corneilius")
        self.resize(1020, 700)
        self.setMinimumSize(720, 500)
        self.setStyleSheet(STYLESHEET)
        self._build()
        self._load_history()

    # -- layout ------------------------------------------------------------

    def _build(self) -> None:
        central = QWidget(self)
        central.setObjectName("central")
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        self.sidebar = QWidget(central)
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setMaximumWidth(SIDEBAR_WIDTH)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbox = QToolBox(self.sidebar)
        self.memory_panel = MemoryPanel(self.orchestrator, self.sidebar)
        self.history_panel = HistoryPanel(self.memory, self.sidebar)
        self.tools_panel = ToolsPanel(self.orchestrator, self.run_tool, self.sidebar)
        self.settings_panel = SettingsPanel(self.settings, self.sidebar)
        self.toolbox.addItem(self.memory_panel, "Memory")
        self.toolbox.addItem(self.history_panel, "History")
        self.toolbox.addItem(self.tools_panel, "Tools")
        self.toolbox.addItem(self.settings_panel, "Settings")
        side_layout.addWidget(self.toolbox)
        outer.addWidget(self.sidebar)

        chat = QWidget(central)
        chat_layout = QVBoxLayout(chat)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(8)

        header = QWidget(chat)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 4, 0)
        self.toggle_button = QPushButton("☰", header)
        self.toggle_button.setObjectName("secondary")
        self.toggle_button.setFixedWidth(40)
        self.toggle_button.setToolTip("Show or hide the sidebar")
        self.toggle_button.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.toggle_button)
        title = QLabel("Corneilius", header)
        title.setObjectName("title")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        self.status = QLabel("● Idle", header)
        self.status.setObjectName("status")
        self.set_status("idle", "Idle")
        header_layout.addWidget(self.status)
        chat_layout.addWidget(header)

        self.transcript = QTextBrowser(chat)
        self.transcript.setObjectName("transcript")
        self.transcript.setFrameStyle(QFrame.Shape.NoFrame)
        self.transcript.setOpenExternalLinks(False)
        chat_layout.addWidget(self.transcript, stretch=1)

        entry = QWidget(chat)
        entry_layout = QHBoxLayout(entry)
        entry_layout.setContentsMargins(0, 0, 0, 0)
        entry_layout.setSpacing(8)
        self.input = ChatInput(entry)
        self.input.setFixedHeight(72)
        self.input.setFrameStyle(QFrame.Shape.NoFrame)
        self.input.send_requested.connect(self.send_message)
        entry_layout.addWidget(self.input, stretch=1)
        self.send_button = QPushButton("Send", entry)
        self.send_button.clicked.connect(self.send_message)
        entry_layout.addWidget(self.send_button, alignment=Qt.AlignmentFlag.AlignBottom)
        self.cancel_button = QPushButton("Cancel", entry)
        self.cancel_button.setObjectName("secondary")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_work)
        entry_layout.addWidget(self.cancel_button, alignment=Qt.AlignmentFlag.AlignBottom)
        chat_layout.addWidget(entry)

        hint = QLabel("Just talk — no commands needed · Ctrl + Enter to send", chat)
        hint.setObjectName("muted")
        hint.setAlignment(Qt.AlignmentFlag.AlignRight)
        chat_layout.addWidget(hint)

        outer.addWidget(chat, stretch=1)

    # -- chat --------------------------------------------------------------

    def _load_history(self) -> None:
        for message in self.memory.get_messages():
            self._append(message["role"], message["content"])

    @staticmethod
    def _bubble(role: str, content: str) -> str:
        name = "You" if role == "user" else "Corneilius"
        color = USER_NAME if role == "user" else ASSISTANT_NAME
        align = "right" if role == "user" else "left"
        # <p align> is the alignment Qt rich text honors; each message is two
        # separate paragraphs so names never merge into adjacent messages.
        safe = html.escape(content).replace("\n", "<br>")
        return (
            f"<p align='{align}' style='color:{color}; margin:0;'><b>{name}</b></p>"
            f"<p align='{align}' style='color:{TEXT}; margin:0 0 12px 0;'>{safe}</p>"
        )

    def _append(self, role: str, content: str) -> None:
        self.transcript.insertHtml(self._bubble(role, content))
        scrollbar = self.transcript.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_status(self, state: str, text: str) -> None:
        dot = {"idle": "●", "thinking": "◐", "error": "●"}.get(state, "●")
        self.status.setText(f"{dot} {text[:60]}")
        self.status.setStyleSheet(STATUS_STYLES.get(state, STATUS_STYLES["idle"]))

    def send_message(self) -> None:
        message = self.input.toPlainText().strip()
        if not message or self.busy:
            return
        self.input.clear()
        self._append("user", message)
        self.set_busy(True)
        self.set_status("thinking", "Thinking…")
        self.start_worker(lambda: self.orchestrator.submit(message), self.on_answer)

    def on_answer(self, answer: str) -> None:
        self._append("assistant", answer)
        self.set_busy(False)
        self.set_status("idle", "Idle")
        self.memory_panel.refresh()
        self.history_panel.refresh()

    def on_worker_error(self, error: str) -> None:
        self._append("assistant", f"I couldn't complete that request: {error}")
        self.set_busy(False)
        self.set_status("error", "Error")

    def on_progress(self, text: str) -> None:
        self.set_status("thinking", text)

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.send_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def cancel_work(self) -> None:
        if not self.busy:
            return
        self.orchestrator.cancel_agent()
        self.set_status("thinking", "Cancelling…")

    # -- workers and tools -------------------------------------------------

    def start_worker(self, fn, on_ok) -> None:  # type: ignore[no-untyped-def]
        worker = Worker(fn, self)
        worker.progressed.connect(self.on_progress)
        worker.finished_ok.connect(on_ok)
        worker.finished_err.connect(self.on_worker_error)
        worker.finished.connect(worker.deleteLater)
        self.orchestrator.on_progress = worker.progressed.emit
        self.worker = worker
        worker.start()

    def run_tool(self, name: str, args: dict) -> None:
        """Run a read-only panel tool on a worker. Writes stay in chat, where confirmation lives."""

        def execute() -> str:
            result = self.orchestrator.tools.execute(name, args)
            if result.confirmation_required:
                return (
                    f"Tool '{name}' can change files, so run it from chat "
                    "where you can reply yes or no."
                )
            if result.ok:
                return f"Tool '{result.name}' succeeded in {result.elapsed_ms}ms:\n{result.output}"
            return f"Tool '{result.name}' failed: {result.error}"

        self.tools_panel.output.setPlainText("Running…")
        self.start_worker(execute, self.tools_panel.output.setPlainText)

    # -- sidebar -----------------------------------------------------------

    def toggle_sidebar(self) -> None:
        collapsed = self.sidebar.maximumWidth() > 0
        animation = QPropertyAnimation(self.sidebar, b"maximumWidth", self)
        animation.setDuration(180)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        animation.setStartValue(self.sidebar.maximumWidth())
        animation.setEndValue(0 if collapsed else SIDEBAR_WIDTH)
        animation.start()
        self._sidebar_animation = animation  # keep alive until finished


def build_qt_app(settings: Settings | None = None) -> tuple[QApplication, CorneiliusWindow]:
    from jarvis.memory.database import ConversationStore
    from jarvis.models.local_model import OllamaModel
    from jarvis.models.mock_model import MockModel

    app = QApplication.instance() or QApplication([])
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.workspace_root.expanduser().mkdir(parents=True, exist_ok=True)
    memory = ConversationStore(settings.database_path)
    if settings.model_provider == "mock":
        model = MockModel()
    elif settings.model_provider == "ollama":
        model = OllamaModel(settings.ollama_url, settings.ollama_model, settings.request_timeout)
    else:
        raise ValueError("JARVIS_MODEL_PROVIDER must be 'mock' or 'ollama'.")
    window = CorneiliusWindow(Orchestrator(model, memory, settings), memory, settings)
    return app, window
