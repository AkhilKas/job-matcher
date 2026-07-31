# ATS Resume Matcher

Pulls open roles straight from companies' public ATS boards (Greenhouse, Lever, Ashby)
and ranks them against your resume. No API keys, no scraping-proxy fight, no LinkedIn ToS
minefield. These three ATS platforms expose public JSON that companies *want* discovered.

This is the core engine: **fetch → normalize → match**. Wrapping it in FastAPI + a React
frontend later is straightforward, but this runs today from the command line.

## Why ATS endpoints

A huge share of tech companies host their careers page on one of these three, and each
returns the full board in one unauthenticated request:

| Provider   | Endpoint                                                                   |
|------------|----------------------------------------------------------------------------|
| Greenhouse | `https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true`      |
| Lever      | `https://api.lever.co/v0/postings/{token}?mode=json`                       |
| Ashby      | `https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true` |

The `token` is the slug in the careers URL: `boards.greenhouse.io/stripe` → `stripe`.

Field mappings were checked against the live Greenhouse/Lever responses and the
[official Ashby docs](https://developers.ashbyhq.com/docs/public-job-posting-api). Still
worth eyeballing one live response per provider before you trust it in production, since a
company can leave any field blank.

## Install

```bash
pip install -r requirements.txt
```

`requests` is the only hard requirement. `sentence-transformers` is strongly recommended
(semantic matching); without it the tool falls back to a lexical scorer that runs but
matches worse. `pdfplumber` is only needed for PDF resumes.

## Quick start

```bash
# Offline demo — bundled sample jobs + sample resume, no network
python -m ats_matcher.run --demo

# Live: rank your resume against a list of boards
python -m ats_matcher.run --companies companies.example.txt --resume path/to/resume.pdf --top 15

# Narrow it: fresh, Boston-or-remote, must mention pytorch
python -m ats_matcher.run \
  --companies companies.example.txt --resume resume.txt \
  --posted-within-hours 24 --location Boston --must-have pytorch
```

Build `companies.txt` by dropping in `provider:token` lines (see `companies.example.txt`).

### Useful flags

- `--location Boston` — case-insensitive substring on the job's location
- `--remote-only` — keep only roles flagged remote
- `--posted-within-hours 24` — freshness filter (apply while you're an early applicant)
- `--must-have pytorch --exclude senior` — required / forbidden keywords (repeatable)
- `--keyword-weight 0.3` — blend skill-overlap into the score (0 = pure semantic)
- `--json` — machine-readable output
- `--model <name>` — swap the sentence-transformers model

## How matching works

1. **Resume → profile.** Text is extracted (`.txt`/`.md` directly, `.pdf` via pdfplumber)
   and a small skills list is pulled for the "why it matched" hints.
2. **Hard filters** prune the pool (location, remote, freshness, keywords).
3. **Semantic rank.** The resume and each job description are embedded and scored by cosine
   similarity. Skill overlap is shown as reasons and can optionally be blended into the score.

The embedding backend is pluggable (`ats_matcher/matching.py`): `SentenceTransformerBackend`
by default, a pure-Python `HashingBackend` as an offline fallback.

## Honest limitations

- **The MVP matcher is coarse.** Embedding a whole resume against a whole job description
  dilutes the signal — a strong requirement match and a throwaway line count the same. It's
  a solid baseline, not a great ranker yet. Biggest wins from here: chunk the JD and score
  section-by-section, weight the requirements/responsibilities sections, and add a
  cross-encoder reranker over the top ~50.
- **The lexical fallback is bag-of-words.** Fine for a smoke test; install
  sentence-transformers before drawing conclusions from scores.
- **ATS coverage only.** Companies not on Greenhouse/Lever/Ashby won't appear. You supply
  the company list; there's no discovery of *which* companies to include.
- **Freshness needs a date.** Greenhouse exposes `updated_at` (not created), Lever
  `createdAt`, Ashby `publishedAt`. `--posted-within-hours` drops postings with no date.

## Suggested next steps

- Swap in **Vertex/Gemini embeddings** to match your stack (backend interface is one class).
- Add the **Gemini resume-structuring** hook (`resume.structure_resume_gemini`, already
  scaffolded) to drive smarter seniority/domain filters.
- Add a **cross-encoder rerank** stage for precision on the shortlist.
- **Persist + dedupe across runs** (SQLite/Postgres) and track when a posting closes.
- Wrap in **FastAPI** and point your React frontend at it.

## Layout

```
ats_matcher/
  models.py       # Job, ResumeProfile, MatchResult, JobFilters
  textutil.py     # HTML→text, tokenization (stdlib only)
  providers.py    # Greenhouse/Lever/Ashby fetch + parse + concurrent fetch_all
  resume.py       # resume loading, skill extraction, optional Gemini hook
  matching.py     # embedding backends, filters, cosine ranking
  run.py          # CLI
  data/           # offline demo fixtures
tests/test_parsers.py   # offline parser + matching tests (python tests/test_parsers.py)
```
