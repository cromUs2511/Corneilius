"""Corneilius Qt theme: dark palette, cyan accents, rounded panels."""

from __future__ import annotations

BG = "#0d1117"
PANEL = "#161b22"
PANEL_BORDER = "#2a3139"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#35c4ff"
ACCENT_DARK = "#1f8fc4"
USER_NAME = "#7aa8ff"
ASSISTANT_NAME = "#5fd6a4"
ERROR = "#ff8181"
FIELD = "#0d1117"

FONT_FAMILY = "Segoe UI"

STYLESHEET = f"""
QMainWindow, QWidget#central, QWidget#sidebar {{
    background-color: {BG};
    color: {TEXT};
    font-family: "{FONT_FAMILY}";
}}
QFrame#card {{
    background-color: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 12px;
}}
QLabel#title {{
    font-size: 20px;
    font-weight: bold;
}}
QLabel#muted {{
    color: {MUTED};
    font-size: 11px;
}}
QLabel#status {{
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 10px;
}}
QTextBrowser#transcript, QTextBrowser#output, QTextEdit#input, QPlainTextEdit {{
    background-color: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 12px;
    padding: 8px;
    font-size: 14px;
    selection-background-color: {ACCENT_DARK};
}}
QLineEdit {{
    background-color: {FIELD};
    border: 1px solid {PANEL_BORDER};
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 13px;
}}
QPushButton {{
    background-color: {ACCENT_DARK};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {ACCENT};
}}
QPushButton:disabled {{
    background-color: #2a3139;
    color: {MUTED};
}}
QPushButton#secondary {{
    background-color: #2a3139;
    color: {TEXT};
}}
QPushButton#secondary:hover {{
    background-color: #3a424c;
}}
QListWidget {{
    background-color: {FIELD};
    border: 1px solid {PANEL_BORDER};
    border-radius: 8px;
    font-size: 12px;
}}
QToolBox::tab {{
    background-color: {PANEL};
    border-radius: 8px;
    padding: 6px;
    font-weight: bold;
}}
QToolBox::tab:selected {{
    color: {ACCENT};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: #2a3139;
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""

STATUS_STYLES = {
    "idle": f"background-color: #1c2620; color: {MUTED};",
    "thinking": f"background-color: #12232e; color: {ACCENT};",
    "error": "background-color: #2e1a1a; color: #ff8181;",
}
