"""Qt workers and input widgets: keep blocking work off the GUI thread."""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QTextEdit

logger = logging.getLogger(__name__)


class Worker(QThread):
    """Run one blocking callable; results come back as signals on the GUI thread."""

    finished_ok = Signal(str)
    finished_err = Signal(str)
    progressed = Signal(str)

    def __init__(self, fn: Callable[[], str], parent: Any = None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # runs on the worker thread
        try:
            self.finished_ok.emit(self._fn())
        except Exception as exc:  # structured failure, never a crash
            logger.exception("Background task failed")
            self.finished_err.emit(str(exc))


class ChatInput(QTextEdit):
    """Multiline input where Ctrl+Enter sends."""

    send_requested = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("input")
        self.setPlaceholderText("Talk to Corneilius…")
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.send_requested.emit)
