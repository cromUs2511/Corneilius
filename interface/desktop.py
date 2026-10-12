from __future__ import annotations

import logging
import queue
import threading

import customtkinter as ctk

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.database import ConversationStore

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ChatWindow:
    """Modern desktop UI; worker results are delivered through a queue on the UI thread."""

    def __init__(self, root: ctk.CTk, orchestrator: Orchestrator, memory: ConversationStore) -> None:
        self.root = root
        self.orchestrator = orchestrator
        self.memory = memory
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.busy = False
        root.title("Corneilius")
        root.geometry("860x680")
        root.minsize(560, 440)
        self._build()
        self._load_history()
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Corneilius", font=ctk.CTkFont(size=20, weight="bold")).grid(
            row=0, column=0, sticky="w")
        self.status = ctk.CTkLabel(header, text="Ready", font=ctk.CTkFont(size=12),
                                   text_color="gray70")
        self.status.grid(row=0, column=1, sticky="e")

        body = ctk.CTkFrame(self.root, corner_radius=12)
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self.transcript = ctk.CTkTextbox(body, wrap="word", state="disabled", corner_radius=12,
                                         font=ctk.CTkFont(size=14), spacing1=4, spacing3=10)
        scrollbar = ctk.CTkScrollbar(body, command=self.transcript.yview)
        self.transcript.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        self.transcript.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        self._text = self.transcript._textbox  # underlying Tk Text for tags/scroll
        self._text.tag_configure("user", foreground="#7aa8ff", font=ctk.CTkFont(size=12, weight="bold"))
        self._text.tag_configure("assistant", foreground="#5fd6a4", font=ctk.CTkFont(size=12, weight="bold"))

        footer = ctk.CTkFrame(self.root, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 4))
        footer.grid_columnconfigure(0, weight=1)
        self.input = ctk.CTkTextbox(footer, height=72, wrap="word", corner_radius=12,
                                    font=ctk.CTkFont(size=14))
        self.input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input.bind("<Control-Return>", self._send_event)
        buttons = ctk.CTkFrame(footer, fg_color="transparent")
        buttons.grid(row=0, column=1, sticky="s")
        self.send_button = ctk.CTkButton(buttons, text="Send", command=self.send, width=90,
                                         font=ctk.CTkFont(size=13, weight="bold"))
        self.send_button.pack(side="left")
        self.cancel_button = ctk.CTkButton(buttons, text="Cancel", command=self.cancel, width=90,
                                           fg_color="gray30", hover_color="gray40",
                                           font=ctk.CTkFont(size=13, weight="bold"), state="disabled")
        self.cancel_button.pack(side="left", padx=(8, 0))
        ctk.CTkLabel(self.root, text="Just talk — no commands needed · Ctrl + Enter to send",
                     font=ctk.CTkFont(size=11), text_color="gray60").grid(
            row=3, column=0, sticky="e", padx=18, pady=(0, 10))

    def _append(self, role: str, content: str) -> None:
        self.transcript.configure(state="normal")
        label = "You" if role == "user" else "Corneilius"
        self.transcript.insert("end", f"{label}\n", role)
        self.transcript.insert("end", f"{content}\n\n")
        self.transcript.configure(state="disabled")
        self._text.see("end")

    def _load_history(self) -> None:
        for message in self.memory.get_messages():
            self._append(message["role"], message["content"])

    def _send_event(self, _event) -> str:  # type: ignore[no-untyped-def]
        self.send()
        return "break"

    def send(self) -> None:
        message = self.input.get("1.0", "end").strip()
        if not message or self.busy:
            return
        self.input.delete("1.0", "end")
        self._append("user", message)
        self.busy = True
        self.send_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status.configure(text="Thinking…", text_color="#7aa8ff")
        self.orchestrator.on_progress = lambda text: self.events.put(("status", text))

        def work() -> None:
            try:
                self.events.put(("answer", self.orchestrator.submit(message)))
            except Exception as exc:
                logger.exception("Chat request failed")
                self.events.put(("error", str(exc)))

        threading.Thread(target=work, name="jarvis-request", daemon=True).start()

    def cancel(self) -> None:
        """Request cancellation of in-flight work (cooperative: the agent loop stops after its current step)."""
        if not self.busy:
            return
        self.orchestrator.cancel_agent()
        self.status.configure(text="Cancelling…", text_color="#ff8181")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    # Live progress from the worker; the run is still in flight.
                    self.status.configure(text=value[:80], text_color="#7aa8ff")
                    continue
                self.busy = False
                self.send_button.configure(state="normal")
                self.cancel_button.configure(state="disabled")
                if kind == "answer":
                    self.status.configure(text="Ready", text_color="gray70")
                    self._append("assistant", value)
                else:
                    self.status.configure(text="Error", text_color="#ff8181")
                    self._append("assistant", f"I couldn't complete that request: {value}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)
