"""Local persistence for scraped jobs.

Stores normalized Job rows in a SQLite DB so subsequent runs can dedupe on
Job.dedupe_key, track first_seen / last_seen, and mark postings closed when
they disappear from fresh fetches of a board we did scan.

Deliberately stdlib-only (sqlite3). The schema uses TEXT for ISO 8601
timestamps and no SQLite dialect-specific features, so swapping in Postgres
later is a client change, not a schema rewrite.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from .models import Job

DEFAULT_DB_PATH = ".cache/jobs.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  dedupe_key       TEXT PRIMARY KEY,
  source           TEXT NOT NULL,
  company          TEXT NOT NULL,
  title            TEXT NOT NULL,
  url              TEXT NOT NULL,
  location         TEXT NOT NULL DEFAULT '',
  department       TEXT NOT NULL DEFAULT '',
  team             TEXT NOT NULL DEFAULT '',
  employment_type  TEXT NOT NULL DEFAULT '',
  remote           INTEGER,
  description      TEXT NOT NULL DEFAULT '',
  posted_at        TEXT,
  compensation     TEXT NOT NULL DEFAULT '',
  external_id      TEXT NOT NULL DEFAULT '',
  first_seen       TEXT NOT NULL,
  last_seen        TEXT NOT NULL,
  is_open          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_jobs_open ON jobs(is_open);
CREATE INDEX IF NOT EXISTS idx_jobs_posted ON jobs(posted_at);
CREATE INDEX IF NOT EXISTS idx_jobs_source_company ON jobs(source, company);
"""


@dataclass
class UpsertResult:
    inserted: int
    updated: int


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        source=row["source"],
        company=row["company"],
        title=row["title"],
        url=row["url"],
        location=row["location"],
        department=row["department"],
        team=row["team"],
        employment_type=row["employment_type"],
        remote=None if row["remote"] is None else bool(row["remote"]),
        description=row["description"],
        posted_at=_parse_iso(row["posted_at"]),
        compensation=row["compensation"],
        external_id=row["external_id"],
    )


class JobStore:
    """SQLite-backed store of Job rows keyed by Job.dedupe_key."""

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_DB_PATH) -> None:
        self.path = str(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> JobStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- writes --

    def upsert_jobs(self, jobs: Iterable[Job], now: datetime | None = None) -> UpsertResult:
        """Insert new jobs; for existing ones (by dedupe_key), refresh mutable
        fields and set last_seen. Sets is_open=1 on any row touched -- a job
        that reappears after being marked closed is considered re-opened."""
        now = now or datetime.now(UTC)
        now_iso = _iso(now)
        inserted = updated = 0

        with self._conn:
            for job in jobs:
                key = job.dedupe_key
                exists = (
                    self._conn.execute("SELECT 1 FROM jobs WHERE dedupe_key = ?", (key,)).fetchone()
                    is not None
                )
                remote_int = None if job.remote is None else int(job.remote)
                if exists:
                    self._conn.execute(
                        """UPDATE jobs SET
                             source=?, company=?, title=?, url=?, location=?, department=?,
                             team=?, employment_type=?, remote=?, description=?, posted_at=?,
                             compensation=?, external_id=?, last_seen=?, is_open=1
                           WHERE dedupe_key=?""",
                        (
                            job.source,
                            job.company,
                            job.title,
                            job.url,
                            job.location,
                            job.department,
                            job.team,
                            job.employment_type,
                            remote_int,
                            job.description,
                            _iso(job.posted_at),
                            job.compensation,
                            job.external_id,
                            now_iso,
                            key,
                        ),
                    )
                    updated += 1
                else:
                    self._conn.execute(
                        """INSERT INTO jobs (
                             dedupe_key, source, company, title, url, location, department,
                             team, employment_type, remote, description, posted_at,
                             compensation, external_id, first_seen, last_seen, is_open
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                        (
                            key,
                            job.source,
                            job.company,
                            job.title,
                            job.url,
                            job.location,
                            job.department,
                            job.team,
                            job.employment_type,
                            remote_int,
                            job.description,
                            _iso(job.posted_at),
                            job.compensation,
                            job.external_id,
                            now_iso,
                            now_iso,
                        ),
                    )
                    inserted += 1

        return UpsertResult(inserted=inserted, updated=updated)

    def mark_closed_for_boards(
        self,
        scanned_boards: Iterable[tuple[str, str]],
        now: datetime,
    ) -> int:
        """Mark jobs closed if their (source, company) is in `scanned_boards`
        but their last_seen is strictly older than `now`.

        The caller must pass every (source, company) pair the current fetch
        actually covered; boards NOT in that list are left alone so a partial
        refresh can't false-close jobs from boards we didn't visit this run.
        Returns the number of rows just marked closed."""
        pairs = list(scanned_boards)
        if not pairs:
            return 0

        now_iso = _iso(now)
        placeholders = ",".join(["(?, ?)"] * len(pairs))
        flat: list[str] = []
        for src, comp in pairs:
            flat.extend([src, comp])

        with self._conn:
            cur = self._conn.execute(
                f"""UPDATE jobs SET is_open = 0
                    WHERE is_open = 1
                      AND last_seen < ?
                      AND (source, company) IN ({placeholders})""",
                [now_iso, *flat],
            )
            return cur.rowcount

    # -- reads --

    def open_jobs(self, *, include_closed: bool = False) -> list[Job]:
        """Return stored jobs. Ordered by posted_at desc (NULLs last)."""
        query = "SELECT * FROM jobs"
        if not include_closed:
            query += " WHERE is_open = 1"
        # SQLite doesn't support NULLS LAST in all versions; emulate it.
        query += " ORDER BY posted_at IS NULL, posted_at DESC"
        return [_row_to_job(r) for r in self._conn.execute(query).fetchall()]

    def count(self, *, include_closed: bool = False) -> int:
        query = "SELECT COUNT(*) FROM jobs"
        if not include_closed:
            query += " WHERE is_open = 1"
        return int(self._conn.execute(query).fetchone()[0])
