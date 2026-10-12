from __future__ import annotations

import re
from datetime import datetime

from .base import ModelAdapter

_GREETING_RE = re.compile(r"^(hi|hello|hey|yo|good\s?(morning|afternoon|evening))\b", re.IGNORECASE)
_HELP_RE = re.compile(r"\b(help|what can you do|commands|abilities)\b", re.IGNORECASE)
_WHO_RE = re.compile(r"\b(who are you|your name|about yourself|what are you)\b", re.IGNORECASE)
_THANKS_RE = re.compile(r"\b(thank|thanks|thx)\b", re.IGNORECASE)
_BYE_RE = re.compile(r"^(bye|goodbye|good ?night|see you)\b", re.IGNORECASE)


class MockModel(ModelAdapter):
    """Offline deterministic provider.

    More conversational than a plain echo so the app feels alive without a
    model, but it cannot reason. Anything it does not recognize is echoed
    back verbatim. Connect a real local model (Ollama) for actual answers.
    """

    def respond(self, message: str, history: list[dict[str, str]]) -> str:
        del history
        cleaned = message.strip()
        lowered = cleaned.casefold()
        if _GREETING_RE.search(cleaned) and len(cleaned) < 60:
            return (
                "Hello! I'm Corneilius, running offline in mock mode, so my replies are "
                "simple. I can still remember things you tell me to remember, and show "
                "saved memories. For full answers, connect a local model via Ollama."
            )
        if _HELP_RE.search(cleaned):
            return (
                "Here's what I can do right now:\n"
                "- Chat (offline replies are simple; a local model makes me smarter)\n"
                "- 'remember that ...' to save a fact, preference, or project note\n"
                "- 'what do you remember about ...?' to recall\n"
                "- 'forget ...' to delete a saved memory\n"
                "- Ask about files in my workspace (needs a real model to act)"
            )
        if _WHO_RE.search(cleaned):
            return (
                "I'm Corneilius, a local-first personal assistant. I'm software — not "
                "conscious or sentient — running on your machine with your data."
            )
        if re.search(r"\bwhat('| i)s the (time|date)\b", lowered) or (
            re.search(r"\b(time|date|today)\b", lowered) and len(cleaned.split()) <= 4
        ):
            now = datetime.now()
            return f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."
        if _THANKS_RE.search(cleaned) and len(cleaned) < 60:
            return "You're welcome!"
        if _BYE_RE.search(cleaned):
            return "Goodbye! I'll be here when you need me."
        return f"I received: {cleaned}"
