"""Optional web research tools: search (DuckDuckGo, no key) and page fetch.

Both are confirmation-required: contacting external sites discloses query
content, so agent runs pause for approval unless started with --allow-web.
Direct /tool calls are explicit approval. SSRF guards restrict fetching to
public http(s) hosts.
"""

from __future__ import annotations

import html as html_module
import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from jarvis.tools.registry import CONFIRMATION_REQUIRED, ToolRegistry, ToolSpec

DDG_SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Corneilius/0.1 (local research assistant; +offline)"
FETCH_TIMEOUT = 15.0
FETCH_MAX_BYTES = 300_000
MAX_RESULTS = 5


def _reject_private_host(host: str) -> str:
    """Return the host if it is safe to contact, else raise ValueError."""
    hostname = host.strip().lower().rstrip(".")
    if not hostname:
        raise ValueError("URL has no host.")
    if hostname in {"localhost"} or hostname.endswith((".localhost", ".local", ".invalid", ".example", ".test")):
        raise ValueError(f"Refusing non-public host: {host!r}.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError(f"Refusing non-public IP: {host!r}.")
    if address is None:
        try:
            infos = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        except (socket.gaierror, TimeoutError, OSError) as exc:
            raise ValueError(f"Could not resolve host {host!r}; refusing.") from exc
        resolved = {info[4][0] for info in infos}
        for raw in resolved:
            try:
                if not ipaddress.ip_address(raw).is_global:
                    raise ValueError(f"Host {host!r} resolves to a non-public address; refusing.")
            except ValueError as exc:
                if "refusing" in str(exc):
                    raise
                raise ValueError(f"Could not verify host {host!r}; refusing.") from exc
    return hostname


def ensure_public_http(url: str) -> str:
    """Validate an absolute public http(s) URL or raise ValueError."""
    cleaned = (url or "").strip()
    parts = urllib.parse.urlsplit(cleaned)
    if parts.scheme not in {"http", "https"}:
        raise ValueError(f"Only http(s) URLs may be fetched, got: {url!r}.")
    if "@" in parts.netloc or not parts.hostname:
        raise ValueError(f"Invalid URL: {url!r}.")
    _reject_private_host(parts.hostname)
    return cleaned


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.chunks.append(data.strip() + " ")

    def text(self) -> str:
        collapsed = re.sub(r"[ \t]+", " ", "".join(self.chunks))
        return re.sub(r"\n\s*\n+", "\n\n", collapsed).strip()


def _ddg_href(raw_href: str) -> str:
    """Unwrap DuckDuckGo redirect links to the real destination."""
    unescaped = html_module.unescape(raw_href)
    parts = urllib.parse.urlsplit(unescaped)
    query = urllib.parse.parse_qs(parts.query)
    if "uddg" in query and ("duckduckgo.com" in (parts.hostname or "") or parts.path.startswith("/l")):
        target = query.get("uddg", [""])[0]
        if target:
            return target
    return unescaped


def search_web(query: str, count: int = MAX_RESULTS) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    count = max(1, min(int(count), 10))
    body = urllib.parse.urlencode({"q": query}).encode()
    request = urllib.request.Request(
        DDG_SEARCH_URL, data=body, headers={"User-Agent": USER_AGENT}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
            page = response.read(FETCH_MAX_BYTES).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"Web search failed: {exc}.") from exc
    links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.DOTALL)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.DOTALL)
    results = []
    for index, (href, title_html) in enumerate(links[:count]):
        title = re.sub(r"<[^>]+>", "", html_module.unescape(title_html)).strip()
        snippet = ""
        if index < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", html_module.unescape(snippets[index])).strip()
        results.append({"title": title or "(no title)", "url": _ddg_href(href), "snippet": snippet})
    if not results:
        return {"output": f"No results found for '{query}'.", "results": []}
    lines = [f"{i + 1}. {item['title']}\n   {item['url']}" + (f"\n   {item['snippet']}" if item["snippet"] else "")
             for i, item in enumerate(results)]
    return {"output": "\n".join(lines), "results": results}


def fetch_page(url: str, max_chars: int = 8000) -> dict[str, Any]:
    target = ensure_public_http(url)
    max_chars = max(1, min(int(max_chars), 100_000))
    request = urllib.request.Request(target, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
            content_type = ""
            try:
                content_type = response.headers.get_content_type()
            except AttributeError:
                content_type = ""
            if content_type and not content_type.startswith("text/"):
                raise ValueError(f"Refusing non-text content ({content_type}): {url!r}.")
            raw = response.read(FETCH_MAX_BYTES).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"Could not fetch {url!r}: {exc}.") from exc
    extractor = _TextExtractor()
    extractor.feed(raw)
    text = extractor.text()
    truncated = len(text) > max_chars
    return {"output": text[:max_chars], "url": target, "size_chars": len(text), "truncated": truncated}


def register_web_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="web.search",
            description="Search the web (DuckDuckGo, no key). Needs approval: queries leave this machine.",
            risk=CONFIRMATION_REQUIRED,
            parameters={
                "query": {"type": "string", "required": True, "max_length": 500},
                "count": {"type": "integer", "required": False, "default": 5},
            },
        ),
        lambda args: search_web(str(args["query"]), int(args.get("count", MAX_RESULTS))),
    )
    registry.register(
        ToolSpec(
            name="web.fetch",
            description="Fetch a public http(s) page as text. Needs approval; local/private hosts refused.",
            risk=CONFIRMATION_REQUIRED,
            parameters={
                "url": {"type": "string", "required": True, "max_length": 2000},
                "max_chars": {"type": "integer", "required": False, "default": 8000},
            },
        ),
        lambda args: fetch_page(str(args["url"]), int(args.get("max_chars", 8000))),
    )
