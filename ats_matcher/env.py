"""Tiny stdlib-only .env loader. Sets keys in os.environ without overwriting
anything already present. Comments (#) and blank lines are ignored; optional
surrounding quotes on values are stripped.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike | None = None) -> bool:
    """Load .env from `path` (or the project root, if unset). Returns True if
    a file was found and read."""
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not p.is_file():
        return False
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)
    return True
