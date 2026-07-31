"""FastAPI backend for the job matcher.

The API is a thin HTTP wrapper around the same pipeline the CLI uses. It
serves ranked-jobs endpoints backed by the SQLite JobStore that
`python -m ats_matcher.run --persist` populates. The API deliberately does
NOT live-fetch ATS boards on request -- cold-fetching 50+ boards per
`/rank` call would push latency into the tens of seconds and blow through
external rate limits. Populate the DB out of band, serve from it here.

Entrypoint for uvicorn:
    uvicorn ats_matcher.api.main:app --reload
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ..env import load_dotenv
from ..matching import (
    CrossEncoderReranker,
    EmbeddingBackend,
    JobFilters,
    Reranker,
    get_default_backend,
    rank,
)
from ..models import Job
from ..resume import build_profile, load_resume_text
from ..storage import DEFAULT_DB_PATH, JobStore
from .schemas import (
    FiltersIn,
    HealthResponse,
    JobOut,
    RankedJob,
    RankRequest,
    RankResponse,
    StatsResponse,
)

load_dotenv()


class AppState:
    """Process-wide holders for expensive-to-init objects (embedding backend,
    optional cross-encoder). Injected once at startup, reused across requests."""

    backend: EmbeddingBackend | None = None
    reranker: Reranker | None = None
    db_path: str = os.environ.get("JOB_MATCHER_DB", DEFAULT_DB_PATH)


state = AppState()


def _to_filters(f: FiltersIn) -> JobFilters:
    return JobFilters(
        location=f.location,
        remote_only=f.remote_only,
        posted_within_hours=f.posted_within_hours,
        must_have=list(f.must_have),
        exclude=list(f.exclude),
    )


def _to_job_out(j: Job) -> JobOut:
    return JobOut(
        source=j.source,
        company=j.company,
        title=j.title,
        url=j.url,
        location=j.location,
        department=j.department,
        team=j.team,
        employment_type=j.employment_type,
        remote=j.remote,
        description=j.description,
        posted_at=j.posted_at.isoformat() if j.posted_at else None,
        compensation=j.compensation,
    )


def _rank_from_db(
    resume_text: str,
    filters_in: FiltersIn,
    top_k: int,
    keyword_weight: float,
    rerank_flag: bool,
    rerank_top_n: int,
    include_closed: bool,
) -> RankResponse:
    if state.backend is None:
        raise HTTPException(status_code=503, detail="embedding backend not initialized")

    with JobStore(state.db_path) as store:
        jobs = store.open_jobs(include_closed=include_closed)
    if not jobs:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No jobs in DB ({state.db_path}). Populate it with: "
                "`python -m ats_matcher.run --companies companies.txt --persist`"
            ),
        )

    profile = build_profile(resume_text)
    reranker: Reranker | None = None
    if rerank_flag:
        if state.reranker is None:
            state.reranker = CrossEncoderReranker()
        reranker = state.reranker

    results = rank(
        profile,
        jobs,
        state.backend,
        filters=_to_filters(filters_in),
        top_k=top_k,
        keyword_weight=keyword_weight,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
    )
    return RankResponse(
        results=[
            RankedJob(
                job=_to_job_out(r.job),
                semantic_score=r.semantic_score,
                final_score=r.final_score,
                keyword_hits=r.keyword_hits,
                rerank_score=r.rerank_score,
            )
            for r in results
        ],
        total_candidates=len(jobs),
        reranked=rerank_flag,
    )


def create_app(
    *,
    backend: EmbeddingBackend | None = None,
    db_path: str | None = None,
    cors_origins: Iterable[str] = ("*",),
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        backend: Inject an EmbeddingBackend (e.g. HashingBackend in tests).
                 If None, the lifespan calls get_default_backend() at startup.
        db_path: Override the JobStore path. If None, uses JOB_MATCHER_DB env
                 var or DEFAULT_DB_PATH.
        cors_origins: CORS allowed origins. Tighten before public deploy.
    """
    if db_path is not None:
        state.db_path = db_path
    if backend is not None:
        # Explicit injection takes effect immediately so callers that don't
        # use `with TestClient(app):` still see a live backend.
        state.backend = backend

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state.backend is None:
            state.backend = get_default_backend()
        yield

    app = FastAPI(title="job-matcher API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        with JobStore(state.db_path) as store:
            return HealthResponse(status="ok", db_path=state.db_path, open_jobs=store.count())

    @app.get("/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        with JobStore(state.db_path) as store:
            total = store.count(include_closed=True)
            open_ = store.count()
        return StatsResponse(
            db_path=state.db_path,
            total_jobs=total,
            open_jobs=open_,
            closed_jobs=total - open_,
        )

    @app.post("/rank", response_model=RankResponse)
    def rank_endpoint(req: RankRequest) -> RankResponse:
        return _rank_from_db(
            req.resume_text,
            req.filters,
            req.top_k,
            req.keyword_weight,
            req.rerank,
            req.rerank_top_n,
            req.include_closed,
        )

    @app.post("/rank/upload", response_model=RankResponse)
    def rank_upload(
        file: UploadFile = File(...),
        top_k: int = Form(20),
        keyword_weight: float = Form(0.0),
        rerank: bool = Form(False),
        rerank_top_n: int = Form(50),
        include_closed: bool = Form(False),
        location: str | None = Form(None),
        remote_only: bool = Form(False),
        posted_within_hours: float | None = Form(None),
        must_have: list[str] = Form(default_factory=list),
        exclude: list[str] = Form(default_factory=list),
    ) -> RankResponse:
        # Persist upload to a tempfile so load_resume_text (which handles
        # .txt/.md/.pdf via pdfplumber) can parse it uniformly.
        suffix = os.path.splitext(file.filename or "")[1].lower() or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        try:
            text = load_resume_text(tmp_path)
        finally:
            os.unlink(tmp_path)
        filters = FiltersIn(
            location=location,
            remote_only=remote_only,
            posted_within_hours=posted_within_hours,
            must_have=must_have,
            exclude=exclude,
        )
        return _rank_from_db(
            text,
            filters,
            top_k,
            keyword_weight,
            rerank,
            rerank_top_n,
            include_closed,
        )

    return app


# Module-level app for `uvicorn ats_matcher.api.main:app` to import.
app = create_app()
