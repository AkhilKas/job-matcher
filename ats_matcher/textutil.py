"""Small text helpers. Stdlib only, no bs4 dependency."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

_BLOCK_TAGS = {
    "p", "br", "div", "li", "ul", "ol", "tr", "table",
    "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
}


class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data):
        self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def html_to_text(raw: str) -> str:
    """Turn (possibly entity-encoded) HTML into clean plain text.

    Greenhouse returns descriptions HTML-entity-encoded, so we unescape first,
    then strip tags. Lever and Ashby already expose a plain-text field, but this
    is a safe fallback for any HTML we get.
    """
    if not raw:
        return ""
    # Some feeds double-encode; unescape twice is safe (idempotent once clean).
    unescaped = html.unescape(raw)
    parser = _Stripper()
    try:
        parser.feed(unescaped)
        out = parser.text()
    except Exception:
        # If the HTML is malformed enough to break the parser, fall back to a
        # blunt tag-strip rather than losing the description entirely.
        out = re.sub(r"<[^>]+>", " ", unescaped)
    return normalize_ws(out)


def normalize_ws(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


_WORD = re.compile(r"[A-Za-z0-9+#.\-]+")


def tokenize(s: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(s or "")]


def contains_term(haystack: str, term: str) -> bool:
    """Case-insensitive whole-token match (so 'go' won't match 'google')."""
    if not term:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(term.lower()) + r"(?![A-Za-z0-9])"
    return re.search(pattern, (haystack or "").lower()) is not None
