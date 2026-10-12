"""Natural-language intents so users can talk plainly instead of using commands.

Only explicit phrasing ("remember ...", "do you remember ...", "forget ...",
"yes"/"no" while a run is paused) triggers actions. Ordinary statements are
never silently persisted.
"""

from __future__ import annotations

import re

REMEMBER_RE = re.compile(
    r"^(?:please\s+)?(?:remember(?:\s+that)?|note\s+down|make\s+a\s+note\s+(?:that\s+)?"
    r"|do\s*n['\u2019]t\s+forget(?:\s+that)?|save\s+to\s+memory|commit\s+to\s+memory)"
    r"\b[\s,:]+(.+)$",
    re.IGNORECASE | re.DOTALL,
)

RECALL_ALL_RE = re.compile(
    r"^(?:what\s+do\s+you\s+remember|what\s+have\s+i\s+(?:told|asked)\s+you\s+to\s+remember"
    r"|(?:list|show\s+me)\s+(?:what\s+)?you\s+remember)\s*\??$",
    re.IGNORECASE,
)

RECALL_ABOUT_RE = re.compile(
    r"^(?:what\s+do\s+you\s+remember\s+about|do\s+you\s+remember)\s+(.+?)\s*\??$",
    re.IGNORECASE | re.DOTALL,
)

FORGET_RE = re.compile(
    r"^(?:please\s+)?(?:forget(?:\s+that)?|delete\s+from\s+memory|remove\s+from\s+memory)"
    r"\b\s+(.+?)\s*\??$",
    re.IGNORECASE | re.DOTALL,
)

PREFERENCE_HINT_RE = re.compile(r"\bprefer\w*\b", re.IGNORECASE)

AFFIRMATIVE = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead", "do it",
    "proceed", "confirm", "yes please", "please do", "sounds good",
    "lets do it", "let's do it", "do it please",
})

NEGATIVE = frozenset({
    "no", "nope", "nah", "cancel", "abort", "stop", "dont", "don't",
    "never mind", "nevermind", "no thanks", "no thank you", "drop it",
})


def _normalize(text: str) -> str:
    return text.strip().casefold().replace("\u2019", "'").rstrip(".!?").strip()


def is_affirmative(text: str) -> bool:
    return _normalize(text) in AFFIRMATIVE


def is_negative(text: str) -> bool:
    return _normalize(text) in NEGATIVE


def parse_remember(text: str) -> str | None:
    """Return the content to save, or None when this is not a remember request."""
    match = REMEMBER_RE.match(text.strip())
    if match is None:
        return None
    content = match.group(1).strip().rstrip(".!?").strip()
    return content or None


def parse_recall(text: str) -> tuple[str, str | None] | None:
    """Return ("all", None) or ("about", query), or None when not a recall request."""
    cleaned = text.strip()
    if RECALL_ALL_RE.match(cleaned):
        return ("all", None)
    match = RECALL_ABOUT_RE.match(cleaned)
    if match is None:
        return None
    query = match.group(1).strip()
    return ("about", query) if query else None


def parse_forget(text: str) -> str | None:
    """Return the forget query, or None when this is not a forget request."""
    match = FORGET_RE.match(text.strip())
    if match is None:
        return None
    query = match.group(1).strip()
    return query or None


def guess_category(content: str) -> str:
    """Default to fact; explicit preference wording becomes a preference."""
    return "preference" if PREFERENCE_HINT_RE.search(content) else "fact"
