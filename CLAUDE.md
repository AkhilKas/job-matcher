# CLAUDE.md

Context for Claude Code working in this repo. Read this before making changes.

## What this project is

A tool that pulls open roles directly from companies' public ATS boards
(Greenhouse, Lever, Ashby) and ranks them against a user's resume. The pipeline is
three stages: **fetch → normalize → match**. It runs today as a CLI. Wrapping it in a
FastAPI backend with a React/TypeScript frontend is a planned next step, not built yet.

The point of using ATS endpoints (instead of scraping job boards) is that these three
platforms expose public, unauthenticated JSON that companies want discovered. No API
keys, no anti-bot fight, stable field shapes.

## Current status

Working and tested:
- Greenhouse / Lever / Ashby fetch + parse into one normalized `Job` schema.
- Resume loading (`.txt`/`.md` direct, `.pdf` via pdfplumber) and a heuristic skill list.
- Hard filters: location, remote-only, freshness (posted-within-hours), must-have / exclude keywords.
- Semantic ranking via cosine similarity, with an optional keyword-overlap blend.
- Offline demo (`--demo`) using bundled fixtures.
- Parser + matching unit tests (`tests/test_parsers.py`), all passing.

Incomplete / provisional:
- The default embedding backend (`sentence-transformers`) may not be installed in every
  environment. There is a pure-Python lexical fallback so the pipeline never hard-fails,
  but it is bag-of-words, not semantic. Do not treat fallback scores as meaningful.
- The matcher is a baseline (see Known limitations).
- No persistence, no dedup across runs, no company discovery.

## Scope and explicit non-goals

In scope: ATS (Greenhouse/Lever/Ashby) fetching + resume-based matching + filtering.

Deliberately OUT of scope. These were considered and rejected on purpose. **Do not add
them back, even if they seem helpful:**
- **No LinkedIn / Indeed / Glassdoor scraping.** Against their ToS, fragile, needs proxies.
  The ATS route is the intended foundation.
- **No harvesting of individual recruiters' names, emails, or contact info, and no bulk
  automated outreach.** That crosses ToS + privacy lines (GDPR/CCPA) and performs worse
  than targeted, human-written outreach. The tool surfaces company + department/team as
  context only; deciding who to contact stays a human step.

If a change request seems to pull toward either of these, flag it rather than implementing.

## Architecture

```
ats_matcher/
  models.py      # Job, ResumeProfile, MatchResult, JobFilters (dataclasses)
  textutil.py    # html_to_text (stdlib, no bs4), tokenize, contains_term
  chunking.py    # chunk_jd: split description into weighted sections
  providers.py   # per-ATS fetch + PURE parse functions, registry, concurrent fetch_all
  resume.py      # load_resume_text, extract_skills, build_profile, optional Gemini hook
  matching.py    # EmbeddingBackend protocol, backends, cosine, apply_filters, rank (chunked)
  env.py         # tiny stdlib .env loader
  run.py         # argparse CLI, --demo mode, pretty + --json output
  data/          # sample_jobs.json, sample_resume.txt (demo fixtures only)
tests/test_parsers.py
tests/test_chunking.py
```

Flow: `run.py` → `providers.fetch_all(specs)` (or demo fixtures) → `resume.build_profile`
→ `matching.rank(profile, jobs, backend, filters)` → print/JSON.

The parse functions in `providers.py` (`parse_greenhouse`, `parse_lever`, `parse_ashby`)
are deliberately separate from the network fetch so they stay unit-testable offline. Keep
that separation when adding providers.

## ATS endpoints and field mappings

| Provider   | Endpoint |
|------------|----------|
| Greenhouse | `https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true` |
| Lever      | `https://api.lever.co/v0/postings/{token}?mode=json` |
| Ashby      | `https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true` |

`token` is the slug from the careers URL (e.g. `boards.greenhouse.io/stripe` → `stripe`).
Spec format used throughout is `provider:token`.

Provider quirks that the parsers already handle (preserve these):
- **Greenhouse**: description is in `content`, HTML-**entity-encoded**, so unescape then
  strip tags. No remote flag (inferred from location/title only, not the description body).
  Date is `updated_at`, not a created date.
- **Lever**: response is a top-level list, not wrapped in `{"jobs": [...]}`. Use
  `descriptionPlain` when present. `createdAt` is epoch **milliseconds**. `workplaceType`
  gives remote/on-site/hybrid.
- **Ashby**: `{"apiVersion", "jobs": [...]}`. Has both `descriptionPlain` and
  `descriptionHtml`. Skip postings where `isListed` is `false` (direct-link only).
  `publishedAt` is ISO 8601.

Field mappings were verified against live Greenhouse/Lever responses and the official
Ashby docs. **Never invent or guess field names.** If you need a field that isn't mapped,
confirm it against a real response (or the provider docs) first, and prefer `.get()` with
graceful fallback since employers leave fields blank.

## Commands

```bash
pip install -r requirements.txt          # requests is the only hard dep

python -m ats_matcher.run --demo          # offline, no network, verifies the pipeline
python -m ats_matcher.run --companies companies.txt --resume resume.pdf --top 15
python -m ats_matcher.run --companies companies.txt --resume resume.txt \
    --posted-within-hours 24 --location Boston --must-have pytorch

python tests/test_parsers.py              # offline parser + matching tests
python tests/test_chunking.py             # offline JD-chunking tests
```

## Conventions

- Python 3.12. Use type hints and `@dataclass`; `from __future__ import annotations`.
- **stdlib-first.** `textutil` uses only the stdlib (no BeautifulSoup). Cosine is
  pure-Python (no hard numpy dependency in the core path).
- **Optional deps are lazy-imported** inside the function/class that needs them
  (`sentence_transformers`, `pdfplumber`, `google-genai`) so importing the package never
  fails when they are absent. Keep it that way.
- **Graceful degradation over crashing:** a bad board warns and returns `[]`; a missing
  field becomes `""`/`None`; a missing embedding backend falls back. Don't introduce hard
  failures on partial data.
- The embedding backend is a pluggable `Protocol`. Add backends (e.g. Vertex/Gemini) as a
  new class implementing `embed(texts) -> list[list[float]]`; don't hardcode a provider.

## Known limitations (be honest about these)

- JDs are now chunked by section (requirements / responsibilities / nice_to_have / other)
  with weighted scoring per `ats_matcher/chunking.py`, so a strong match on requirements
  outranks the same match in a nice-to-have bullet. The resume side is still a single blob,
  though -- a strong signal in one resume section still gets diluted by weaker sections.
- The lexical fallback is bag-of-words only.
- Coverage is limited to companies hosted on Greenhouse/Lever/Ashby. The user supplies the
  company list; there is no discovery of which companies to include.
- `--posted-within-hours` drops postings that have no usable date.

## Roadmap (rough priority)

1. Swap in **Vertex/Gemini embeddings** as a backend to match the user's stack.
2. Improve match quality: **chunk the JD**, weight requirements/responsibilities sections,
   add a **cross-encoder rerank** over the top ~50.
3. **Persist + dedupe** across runs (SQLite/Postgres); track when a posting closes.
4. **FastAPI** wrapper + React/TypeScript frontend.
5. Optional **Gemini resume-structuring** (scaffolded in `resume.structure_resume_gemini`)
   to drive smarter seniority/domain filters.

## Working agreements for Claude Code

- After any change, run every test file in `tests/` and `python -m ats_matcher.run --demo`;
  all must stay green. Add tests for new parsers/behavior.
- Keep parser field mappings accurate and verifiable. If unsure about a field, say so
  rather than guessing.
- Ask before adding a new heavy dependency; prefer stdlib or an optional lazy import.
- Respect the scope boundaries above (no board scraping, no contact harvesting).
- Keep the offline demo and the fallback path working so the tool runs without network or
  optional models.
