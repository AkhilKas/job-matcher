"""Command-line entrypoint.

Examples
--------
# Live: rank a resume against a list of ATS boards
python -m ats_matcher.run --companies companies.example.txt --resume resume.txt --top 15

# Fetch + persist to the local SQLite cache and close stale postings
python -m ats_matcher.run --companies companies.example.txt --resume resume.txt --persist

# Rank against the cached DB without hitting the network
python -m ats_matcher.run --from-db --resume resume.txt

# Only fresh, Boston-or-remote roles that mention pytorch
python -m ats_matcher.run --companies companies.example.txt --resume resume.txt \
    --posted-within-hours 24 --location Boston --must-have pytorch

# Offline demo (bundled sample jobs + sample resume, no network needed)
python -m ats_matcher.run --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime

from .env import load_dotenv
from .matching import CrossEncoderReranker, JobFilters, get_default_backend, rank
from .models import Job
from .providers import fetch_all, parse_spec
from .resume import build_profile, load_resume_text
from .storage import DEFAULT_DB_PATH, JobStore

load_dotenv()

_HERE = os.path.dirname(__file__)
_SAMPLE_JOBS = os.path.join(_HERE, "data", "sample_jobs.json")
_SAMPLE_RESUME = os.path.join(_HERE, "data", "sample_resume.txt")


def _read_specs(path: str) -> list[str]:
    specs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            specs.append(line)
    return specs


def _load_sample_jobs() -> list[Job]:
    with open(_SAMPLE_JOBS, encoding="utf-8") as f:
        rows = json.load(f)
    jobs: list[Job] = []
    for r in rows:
        posted = r.get("posted_at")
        dt = datetime.fromisoformat(posted) if posted else None
        jobs.append(
            Job(
                source=r.get("source", "demo"),
                company=r.get("company", ""),
                title=r.get("title", ""),
                url=r.get("url", ""),
                location=r.get("location", ""),
                department=r.get("department", ""),
                team=r.get("team", ""),
                employment_type=r.get("employment_type", ""),
                remote=r.get("remote"),
                description=r.get("description", ""),
                posted_at=dt,
                compensation=r.get("compensation", ""),
            )
        )
    return jobs


def _age_str(dt) -> str:
    if not dt:
        return "date n/a"
    days = (datetime.now(UTC) - dt).total_seconds() / 86400.0
    if days < 1:
        return "today"
    if days < 2:
        return "1 day ago"
    return f"{int(days)} days ago"


def _print_results(results, semantic: bool) -> None:
    if not results:
        print("\nNo matching jobs after filtering. Loosen the filters or add more companies.")
        return
    note = (
        ""
        if semantic
        else "  (lexical fallback -- install sentence-transformers for semantic scores)"
    )
    print(f"\nTop {len(results)} matches{note}\n" + "=" * 72)
    for i, r in enumerate(results, 1):
        j = r.job
        remote = "remote" if j.remote is True else ("on-site" if j.remote is False else "loc n/a")
        line1 = f"{i:>2}. [{r.final_score:.3f}] {j.title}  —  {j.company} ({j.source})"
        meta = "  ".join(
            x for x in [j.location or "location n/a", remote, _age_str(j.posted_at)] if x
        )
        print(line1)
        print(f"    {meta}")
        if r.keyword_hits:
            print(f"    matched: {', '.join(r.keyword_hits[:12])}")
        if j.url:
            print(f"    {j.url}")
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ats_matcher", description="Rank ATS jobs against a resume.")
    src = p.add_argument_group("job source")
    src.add_argument("--companies", help="file with one 'provider:token' per line")
    src.add_argument(
        "--company", action="append", default=[], help="a single 'provider:token' (repeatable)"
    )
    src.add_argument("--demo", action="store_true", help="use bundled offline sample data")
    src.add_argument(
        "--from-db",
        action="store_true",
        help="rank against the persisted SQLite cache instead of fetching live",
    )

    persist = p.add_argument_group("persistence")
    persist.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"SQLite path for --persist / --from-db (default: {DEFAULT_DB_PATH})",
    )
    persist.add_argument(
        "--persist",
        action="store_true",
        help="upsert fresh fetches into the DB and mark stale postings closed",
    )
    persist.add_argument(
        "--include-closed",
        action="store_true",
        help="with --from-db, also include postings marked closed",
    )

    p.add_argument("--resume", help="path to resume (.txt/.md/.pdf); defaults to sample in --demo")
    p.add_argument("--top", type=int, default=20)

    flt = p.add_argument_group("filters")
    flt.add_argument("--location", help="case-insensitive substring, e.g. Boston")
    flt.add_argument("--remote-only", action="store_true")
    flt.add_argument("--posted-within-hours", type=float, default=None)
    flt.add_argument(
        "--must-have", action="append", default=[], help="keyword that must appear (repeatable)"
    )
    flt.add_argument(
        "--exclude", action="append", default=[], help="keyword that must NOT appear (repeatable)"
    )

    mtl = p.add_argument_group("matching")
    mtl.add_argument(
        "--keyword-weight",
        type=float,
        default=0.0,
        help="0..1 blend of skill overlap into the score",
    )
    mtl.add_argument("--model", default=None, help="sentence-transformers model name")
    mtl.add_argument(
        "--rerank",
        action="store_true",
        help="rescore the top N candidates with a cross-encoder (needs sentence-transformers)",
    )
    mtl.add_argument(
        "--rerank-top-n",
        type=int,
        default=50,
        help="how many first-stage candidates to send to the reranker (default 50)",
    )
    mtl.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="cross-encoder model name for --rerank",
    )

    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--json", action="store_true", help="emit results as JSON")

    args = p.parse_args(argv)

    # ---- mutex validation ----
    live_specs = bool(args.company or args.companies)
    modes = [args.demo, args.from_db, live_specs]
    if sum(bool(m) for m in modes) > 1:
        p.error("--demo, --from-db, and live --companies/--company are mutually exclusive")
    if args.persist and not live_specs:
        p.error("--persist requires --companies / --company (nothing to persist otherwise)")
    if args.include_closed and not args.from_db:
        p.error("--include-closed only applies with --from-db")

    # ---- resolve job source ----
    if args.demo:
        jobs = _load_sample_jobs()
    elif args.from_db:
        with JobStore(args.db) as store:
            jobs = store.open_jobs(include_closed=args.include_closed)
        print(f"Loaded {len(jobs)} jobs from {args.db}.", file=sys.stderr)
    else:
        specs = list(args.company)
        if args.companies:
            specs += _read_specs(args.companies)
        if not specs:
            p.error("provide --companies / --company, --from-db, or --demo")
        print(f"Fetching {len(specs)} board(s)...", file=sys.stderr)
        jobs = fetch_all(specs, max_workers=args.max_workers)

        if args.persist:
            scanned = [parse_spec(s) for s in specs]
            now = datetime.now(UTC)
            with JobStore(args.db) as store:
                result = store.upsert_jobs(jobs, now=now)
                closed = store.mark_closed_for_boards(scanned, now=now)
            print(
                f"Persisted to {args.db}: +{result.inserted} new, ~{result.updated} refreshed, "
                f"{closed} marked closed.",
                file=sys.stderr,
            )

    if not jobs:
        print("No jobs fetched.", file=sys.stderr)
        return 1
    print(f"{len(jobs)} unique postings collected.", file=sys.stderr)

    # ---- resume ----
    resume_path = args.resume or (_SAMPLE_RESUME if args.demo else None)
    if not resume_path:
        p.error("provide --resume")
    profile = build_profile(load_resume_text(resume_path))

    # ---- rank ----
    filters = JobFilters(
        location=args.location,
        remote_only=args.remote_only,
        posted_within_hours=args.posted_within_hours,
        must_have=args.must_have,
        exclude=args.exclude,
    )
    backend = get_default_backend(args.model)
    reranker = None
    if args.rerank:
        print(f"Loading cross-encoder ({args.rerank_model})...", file=sys.stderr)
        reranker = CrossEncoderReranker(args.rerank_model)
    results = rank(
        profile,
        jobs,
        backend,
        filters=filters,
        top_k=args.top,
        keyword_weight=args.keyword_weight,
        reranker=reranker,
        rerank_top_n=args.rerank_top_n,
    )

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        _print_results(results, semantic=getattr(backend, "semantic", False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
