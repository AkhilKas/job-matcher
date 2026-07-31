"""Offline tests for the SQLite-backed JobStore.

Run: python tests/test_storage.py
"""

import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ats_matcher.models import Job  # noqa: E402
from ats_matcher.storage import JobStore  # noqa: E402


def _job(
    company: str,
    title: str,
    *,
    source: str = "greenhouse",
    location: str = "Boston, MA",
    description: str = "desc",
    remote: bool | None = False,
    posted_at: datetime | None = None,
) -> Job:
    return Job(
        source=source,
        company=company,
        title=title,
        url=f"https://boards.greenhouse.io/{company}/{title.lower().replace(' ', '-')}",
        location=location,
        remote=remote,
        description=description,
        posted_at=posted_at,
    )


def _fresh_store() -> tuple[JobStore, str]:
    tmpdir = tempfile.mkdtemp(prefix="jobstore_test_")
    return JobStore(os.path.join(tmpdir, "jobs.db")), tmpdir


def test_empty_store():
    store, _ = _fresh_store()
    try:
        assert store.count() == 0
        assert store.open_jobs() == []
    finally:
        store.close()
    print("empty store: OK")


def test_insert_then_reupsert():
    store, _ = _fresh_store()
    try:
        jobs = [_job("acme", "ML Engineer"), _job("northwind", "MLOps")]
        r1 = store.upsert_jobs(jobs, now=datetime(2026, 7, 30, tzinfo=UTC))
        assert r1.inserted == 2 and r1.updated == 0
        assert store.count() == 2

        # Same dedupe_keys -> updates, not new rows
        r2 = store.upsert_jobs(jobs, now=datetime(2026, 7, 31, tzinfo=UTC))
        assert r2.inserted == 0 and r2.updated == 2
        assert store.count() == 2  # no duplicates
    finally:
        store.close()
    print("insert + re-upsert dedupes: OK")


def test_mutable_field_update_preserves_first_seen():
    store, _ = _fresh_store()
    try:
        original = _job("acme", "ML Engineer", description="v1")
        t0 = datetime(2026, 7, 30, tzinfo=UTC)
        store.upsert_jobs([original], now=t0)

        # Re-upsert with a changed description (same dedupe_key)
        updated = _job("acme", "ML Engineer", description="v2 with more detail")
        t1 = datetime(2026, 7, 31, tzinfo=UTC)
        store.upsert_jobs([updated], now=t1)

        rows = store.open_jobs()
        assert len(rows) == 1
        assert rows[0].description == "v2 with more detail"

        # Check first_seen unchanged, last_seen bumped via a raw SELECT
        row = store._conn.execute("SELECT first_seen, last_seen FROM jobs").fetchone()
        assert row["first_seen"].startswith("2026-07-30")
        assert row["last_seen"].startswith("2026-07-31")
    finally:
        store.close()
    print("mutable update keeps first_seen, refreshes last_seen: OK")


def test_mark_closed_only_touches_scanned_boards():
    store, _ = _fresh_store()
    try:
        t0 = datetime(2026, 7, 30, tzinfo=UTC)
        # Seed three jobs across two boards
        acme_a = _job("acme", "ML Engineer")
        acme_b = _job("acme", "MLOps Engineer")
        nw_a = _job("northwind", "Data Engineer", source="lever")
        store.upsert_jobs([acme_a, acme_b, nw_a], now=t0)

        # Fresh scan of acme_greenhouse only -- acme_b disappears from feed,
        # northwind was NOT scanned so its job must stay open.
        t1 = t0 + timedelta(days=1)
        store.upsert_jobs([acme_a], now=t1)
        closed = store.mark_closed_for_boards([("greenhouse", "acme")], now=t1)
        assert closed == 1, f"expected exactly acme_b to close, got {closed}"

        keys_open = {j.dedupe_key for j in store.open_jobs()}
        assert acme_a.dedupe_key in keys_open
        assert acme_b.dedupe_key not in keys_open  # closed
        assert nw_a.dedupe_key in keys_open  # untouched board -> not closed
    finally:
        store.close()
    print("mark_closed respects scanned board scope: OK")


def test_reopened_job_flips_is_open_back():
    store, _ = _fresh_store()
    try:
        t0 = datetime(2026, 7, 30, tzinfo=UTC)
        j = _job("acme", "ML Engineer")
        store.upsert_jobs([j], now=t0)

        # Simulate closure
        t1 = t0 + timedelta(days=1)
        store.upsert_jobs([], now=t1)  # no jobs in this scan
        store.mark_closed_for_boards([("greenhouse", "acme")], now=t1)
        assert store.count() == 0  # excludes closed by default
        assert store.count(include_closed=True) == 1

        # It comes back on a later scan -> is_open flips to 1
        t2 = t1 + timedelta(days=1)
        store.upsert_jobs([j], now=t2)
        assert store.count() == 1
    finally:
        store.close()
    print("reopened job returns to is_open=1: OK")


def test_job_roundtrip_preserves_nullable_fields():
    store, _ = _fresh_store()
    try:
        # remote=None and posted_at=None are the interesting cases -- must
        # survive the SQL layer unchanged.
        j = _job("acme", "Recruiter", remote=None, posted_at=None)
        store.upsert_jobs([j], now=datetime(2026, 7, 30, tzinfo=UTC))
        [back] = store.open_jobs()
        assert back.remote is None
        assert back.posted_at is None
        assert back.title == "Recruiter"
    finally:
        store.close()
    print("roundtrip preserves None fields: OK")


def test_open_jobs_order_posted_desc_nulls_last():
    store, _ = _fresh_store()
    try:
        t0 = datetime(2026, 7, 30, tzinfo=UTC)
        older = _job("acme", "Older", posted_at=datetime(2026, 6, 1, tzinfo=UTC))
        newer = _job("acme", "Newer", posted_at=datetime(2026, 7, 20, tzinfo=UTC))
        unknown = _job("acme", "Undated", posted_at=None)
        store.upsert_jobs([older, unknown, newer], now=t0)

        titles = [j.title for j in store.open_jobs()]
        assert titles.index("Newer") < titles.index("Older") < titles.index("Undated")
    finally:
        store.close()
    print("ordering posted-desc with nulls last: OK")


if __name__ == "__main__":
    test_empty_store()
    test_insert_then_reupsert()
    test_mutable_field_update_preserves_first_seen()
    test_mark_closed_only_touches_scanned_boards()
    test_reopened_job_flips_is_open_back()
    test_job_roundtrip_preserves_nullable_fields()
    test_open_jobs_order_posted_desc_nulls_last()
    print("\nAll storage tests passed.")
