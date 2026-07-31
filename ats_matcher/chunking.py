"""Best-effort chunking of a job description into labeled sections.

The pipeline treats a JD as a mixture of requirements, responsibilities, and
nice-to-haves. A key requirement and a throwaway "we have unlimited PTO" line
should not count equally, so we split by section headers and let the ranker
weight them (see SECTION_WEIGHTS). When no headers are found the whole JD
falls through as 'other', which reproduces the old flat-embedding behavior.
"""

from __future__ import annotations

import re

# Weight applied to each section when averaging per-chunk similarities.
# 'title' isn't produced by chunk_jd -- it's the job title/department/team
# prefix that matching.py stitches on separately -- but it lives here so both
# callers pull weights from one place.
SECTION_WEIGHTS: dict[str, float] = {
    "title": 2.0,
    "requirements": 1.5,
    "responsibilities": 1.0,
    "other": 0.75,
    "nice_to_have": 0.4,
}

# Case- and punctuation-insensitive header phrases -> section label. Compared
# against a normalized version of each line (see _normalize_header).
_HEADER_MAP: dict[str, str] = {
    # requirements-flavored
    "requirements": "requirements",
    "requirement": "requirements",
    "qualifications": "requirements",
    "minimum qualifications": "requirements",
    "basic qualifications": "requirements",
    "required qualifications": "requirements",
    "must haves": "requirements",
    "must have": "requirements",
    "what were looking for": "requirements",
    "what we are looking for": "requirements",
    "who you are": "requirements",
    "about you": "requirements",
    "you have": "requirements",
    "skills": "requirements",
    # responsibilities-flavored
    "responsibilities": "responsibilities",
    "what youll do": "responsibilities",
    "what you will do": "responsibilities",
    "the role": "responsibilities",
    "role overview": "responsibilities",
    "day to day": "responsibilities",
    "you will": "responsibilities",
    "about the role": "responsibilities",
    # nice-to-have flavored
    "nice to have": "nice_to_have",
    "nice to haves": "nice_to_have",
    "bonus": "nice_to_have",
    "bonus points": "nice_to_have",
    "preferred qualifications": "nice_to_have",
    "preferred": "nice_to_have",
    "pluses": "nice_to_have",
    "extra credit": "nice_to_have",
}

_HEADER_MAX_LEN = 60


def _normalize_header(line: str) -> str:
    """Lowercase, strip punctuation and whitespace. Return '' if the line is
    too long or obviously a sentence rather than a heading."""
    s = line.strip().rstrip(":").rstrip("-").strip()
    if not s or len(s) > _HEADER_MAX_LEN:
        return ""
    # Section headers don't typically contain internal punctuation like commas
    # or semicolons -- those suggest a sentence, which we skip.
    if any(c in s for c in ",;"):
        return ""
    s = re.sub(r"[^a-zA-Z\s]", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def chunk_jd(text: str) -> list[tuple[str, str]]:
    """Split `text` into (label, chunk) pairs in document order.

    Text before the first recognized header is labeled 'other'; text after
    each header takes that section's label until the next header. If no
    header is found, returns a single ('other', text) entry so callers can
    treat the output uniformly.
    """
    if not text.strip():
        return []

    current_label = "other"
    current_buf: list[str] = []
    chunks: list[tuple[str, str]] = []

    for line in text.split("\n"):
        header_key = _normalize_header(line)
        if header_key in _HEADER_MAP:
            content = "\n".join(current_buf).strip()
            if content:
                chunks.append((current_label, content))
            current_label = _HEADER_MAP[header_key]
            current_buf = []
        else:
            current_buf.append(line)

    tail = "\n".join(current_buf).strip()
    if tail:
        chunks.append((current_label, tail))

    return chunks
