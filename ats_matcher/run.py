"""Command-line entrypoint.

Examples
--------
# Live: rank a resume against a list of ATS boards
python -m ats_matcher.run --companies companies.example.txt --resume resume.txt --top 15

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
from datetime import datetime, timezone

from .matching import JobFilters, get_default_backend, rank
from .models import Job
from .providers import fetch_all
from .resume import build_profile, load_resume_text

_HERE = os.path.dirname(__file__)
_SAMPLE_JOBS = os.path.join(_HERE, "data", "sample_jobs.json")
_SAMPLE_RESUME = os.path.join(_HERE, "data", "sample_resume.txt")


def _read_specs(path: str) -> list[str]:
    specs: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            specs.append(line)
    return specs


def _load_sample_jobs() -> list[Job]:
    with open(_SAMPLE_JOBS, "r", encoding="utf-8") as f:
        rows = json.load(f)
    jobs: list[Job] = []
    for r in rows:
        posted = r.get("posted_at")
        dt = datetime.fromisoformat(posted) if posted else None
        jobs.append(Job(
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
        ))
    return jobs


def _age_str(dt) -> str:
    if not dt:
        return "date n/a"
    days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    if days < 1:
        return "today"
    if days < 2:
        return "1 day ago"
    return f"{int(days)} days ago"


def _print_results(results, semantic: bool) -> None:
    if not results:
        print("\nNo matching jobs after filtering. Loosen the filters or add more companies.")
        return
    note = "" if semantic else "  (lexical fallback -- install sentence-transformers for semantic scores)"
    print(f"\nTop {len(results)} matches{note}\n" + "=" * 72)
    for i, r in enumerate(results, 1):
        j = r.job
        remote = "remote" if j.remote is True else ("on-site" if j.remote is False else "loc n/a")
        line1 = f"{i:>2}. [{r.final_score:.3f}] {j.title}  —  {j.company} ({j.source})"
        meta = "  ".join(x for x in [j.location or "location n/a", remote, _age_str(j.posted_at)] if x)
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
    src.add_argument("--company", action="append", default=[], help="a single 'provider:token' (repeatable)")
    src.add_argument("--demo", action="store_true", help="use bundled offline sample data")

    p.add_argument("--resume", help="path to resume (.txt/.md/.pdf); defaults to sample in --demo")
    p.add_argument("--top", type=int, default=20)

    flt = p.add_argument_group("filters")
    flt.add_argument("--location", help="case-insensitive substring, e.g. Boston")
    flt.add_argument("--remote-only", action="store_true")
    flt.add_argument("--posted-within-hours", type=float, default=None)
    flt.add_argument("--must-have", action="append", default=[], help="keyword that must appear (repeatable)")
    flt.add_argument("--exclude", action="append", default=[], help="keyword that must NOT appear (repeatable)")

    mtl = p.add_argument_group("matching")
    mtl.add_argument("--keyword-weight", type=float, default=0.0, help="0..1 blend of skill overlap into the score")
    mtl.add_argument("--model", default=None, help="sentence-transformers model name")

    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--json", action="store_true", help="emit results as JSON")

    args = p.parse_args(argv)

    # ---- resolve job source ----
    if args.demo:
        jobs = _load_sample_jobs()
    else:
        specs = list(args.company)
        if args.companies:
            specs += _read_specs(args.companies)
        if not specs:
            p.error("provide --companies / --company, or use --demo")
        print(f"Fetching {len(specs)} board(s)...", file=sys.stderr)
        jobs = fetch_all(specs, max_workers=args.max_workers)
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
    results = rank(profile, jobs, backend, filters=filters, top_k=args.top, keyword_weight=args.keyword_weight)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        _print_results(results, semantic=getattr(backend, "semantic", False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
