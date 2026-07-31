"""Fetch and normalize job postings from public ATS boards.

All three endpoints are public JSON, no auth, no anti-bot:
  Greenhouse: https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true
  Lever:      https://api.lever.co/v0/postings/{token}?mode=json
  Ashby:      https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true

The `token` is the board slug you can read off a company's careers URL, e.g.
boards.greenhouse.io/stripe -> "stripe", jobs.lever.co/spotify -> "spotify",
jobs.ashbyhq.com/ramp -> "ramp".

Field mappings below were checked against the Greenhouse/Lever public responses and
the official Ashby docs (developers.ashbyhq.com/docs/public-job-posting-api). ATS
feeds only carry what the employer publishes, so any field can be missing; every
parser degrades gracefully to "".
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import requests

from .models import Job
from .textutil import html_to_text, normalize_ws

USER_AGENT = "ats-resume-matcher/0.1 (personal job search tool)"
TIMEOUT = 20
RETRIES = 2

VALID_PROVIDERS = ("greenhouse", "lever", "ashby")


# --------------------------------------------------------------------------- #
# date parsing
# --------------------------------------------------------------------------- #


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        s = s.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None


def parse_epoch_ms(n) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(n) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OSError):
        return None


def _looks_remote(*texts: str) -> bool | None:
    joined = " ".join(t for t in texts if t).lower()
    if "remote" in joined:
        return True
    return None


# --------------------------------------------------------------------------- #
# parsers  (pure functions: JSON in, list[Job] out -- unit-testable offline)
# --------------------------------------------------------------------------- #


def parse_greenhouse(data: dict, token: str) -> list[Job]:
    jobs: list[Job] = []
    for j in (data or {}).get("jobs", []) or []:
        loc = (j.get("location") or {}).get("name", "") or ""
        depts = j.get("departments") or []
        dept = depts[0].get("name", "") if depts else ""
        desc = html_to_text(j.get("content", "") or "")
        jobs.append(
            Job(
                source="greenhouse",
                company=token,
                title=normalize_ws(j.get("title", "") or ""),
                url=j.get("absolute_url", "") or "",
                location=normalize_ws(loc),
                department=normalize_ws(dept),
                team="",
                employment_type="",
                remote=_looks_remote(loc, j.get("title", "")),
                description=desc,
                posted_at=parse_iso(j.get("updated_at")),
                external_id=str(j.get("id", "")),
                raw=j,
            )
        )
    return jobs


def parse_lever(data: list, token: str) -> list[Job]:
    jobs: list[Job] = []
    for j in data or []:
        cats = j.get("categories") or {}
        wtype = (j.get("workplaceType") or "").strip().lower()
        if wtype == "remote":
            remote: bool | None = True
        elif wtype in ("on-site", "onsite", "hybrid"):
            remote = False
        else:
            remote = _looks_remote(cats.get("location", ""))
        desc = j.get("descriptionPlain") or html_to_text(j.get("description", "") or "")
        jobs.append(
            Job(
                source="lever",
                company=token,
                title=normalize_ws(j.get("text", "") or ""),
                url=j.get("hostedUrl") or j.get("applyUrl") or "",
                location=normalize_ws(cats.get("location", "") or ""),
                department=normalize_ws(cats.get("department", "") or ""),
                team=normalize_ws(cats.get("team", "") or ""),
                employment_type=normalize_ws(cats.get("commitment", "") or ""),
                remote=remote,
                description=normalize_ws(desc),
                posted_at=parse_epoch_ms(j.get("createdAt")),
                external_id=str(j.get("id", "")),
                raw=j,
            )
        )
    return jobs


def parse_ashby(data: dict, token: str) -> list[Job]:
    jobs: list[Job] = []
    for j in (data or {}).get("jobs", []) or []:
        # isListed=False means "direct link only", skip from a discovery feed.
        if j.get("isListed") is False:
            continue
        comp = ""
        comp_obj = j.get("compensation") or {}
        if isinstance(comp_obj, dict):
            comp = comp_obj.get("compensationTierSummary", "") or ""
        desc = j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml", "") or "")
        jobs.append(
            Job(
                source="ashby",
                company=token,
                title=normalize_ws(j.get("title", "") or ""),
                url=j.get("applyUrl") or j.get("jobUrl") or "",
                location=normalize_ws(j.get("location", "") or ""),
                department=normalize_ws(j.get("department", "") or ""),
                team=normalize_ws(j.get("team", "") or ""),
                employment_type=normalize_ws(j.get("employmentType", "") or ""),
                remote=j.get("isRemote") if isinstance(j.get("isRemote"), bool) else None,
                description=normalize_ws(desc),
                posted_at=parse_iso(j.get("publishedAt")),
                compensation=normalize_ws(comp),
                raw=j,
            )
        )
    return jobs


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, tuple[Callable[[str], str], Callable]] = {
    "greenhouse": (
        lambda t: f"https://boards-api.greenhouse.io/v1/boards/{t}/jobs?content=true",
        parse_greenhouse,
    ),
    "lever": (
        lambda t: f"https://api.lever.co/v0/postings/{t}?mode=json",
        parse_lever,
    ),
    "ashby": (
        lambda t: f"https://api.ashbyhq.com/posting-api/job-board/{t}?includeCompensation=true",
        parse_ashby,
    ),
}


def _get_json(url: str):
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 -- we want to retry on anything transient
            last_err = e
            if attempt < RETRIES:
                time.sleep(0.6 * (attempt + 1))
    raise last_err  # type: ignore[misc]


def parse_spec(spec: str) -> tuple[str, str]:
    """'greenhouse:stripe' -> ('greenhouse', 'stripe')."""
    if ":" not in spec:
        raise ValueError(f"bad spec {spec!r}; expected 'provider:token' e.g. 'greenhouse:stripe'")
    provider, token = spec.split(":", 1)
    provider = provider.strip().lower()
    token = token.strip()
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; valid: {', '.join(VALID_PROVIDERS)}")
    if not token:
        raise ValueError(f"empty token in spec {spec!r}")
    return provider, token


def fetch_one(spec: str) -> list[Job]:
    """Fetch and normalize one board. On failure, warn and return []."""
    try:
        provider, token = parse_spec(spec)
    except ValueError as e:
        print(f"[skip] {e}", file=sys.stderr)
        return []
    build_url, parser = _REGISTRY[provider]
    try:
        data = _get_json(build_url(token))
        return parser(data, token)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {spec}: {type(e).__name__}: {e}", file=sys.stderr)
        return []


def fetch_all(specs: list[str], max_workers: int = 8, dedupe: bool = True) -> list[Job]:
    """Fetch many boards concurrently and merge into one list."""
    results: list[Job] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one, s): s for s in specs}
        for fut in as_completed(futures):
            results.extend(fut.result())

    if not dedupe:
        return results
    seen: set[str] = set()
    deduped: list[Job] = []
    for job in results:
        if job.dedupe_key in seen:
            continue
        seen.add(job.dedupe_key)
        deduped.append(job)
    return deduped
