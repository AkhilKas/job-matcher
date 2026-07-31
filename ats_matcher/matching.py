"""Rank jobs against a resume.

Design: the embedding backend is pluggable. Default is sentence-transformers
(local, free, no API key). If it isn't installed we fall back to a pure-Python
lexical hashing vectorizer so the pipeline still runs anywhere -- but that
fallback is bag-of-words, NOT semantic, so install sentence-transformers (or
wire in Vertex/Gemini embeddings) for real matching quality.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import UTC, datetime
from typing import Protocol

from .models import Job, JobFilters, MatchResult, ResumeProfile
from .textutil import contains_term, tokenize

# --------------------------------------------------------------------------- #
# embedding backends
# --------------------------------------------------------------------------- #


class EmbeddingBackend(Protocol):
    name: str
    semantic: bool

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class GeminiEmbeddingBackend:
    """Gemini embeddings via the google-genai SDK. Reads the API key from the
    JOB_MATCHER env var (deliberately not GEMINI_API_KEY so the SDK doesn't
    silently auto-pick it up elsewhere)."""

    semantic = True

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        try:
            from google import genai  # lazy: optional dep
        except ImportError as e:
            raise ImportError(
                "gemini backend needs the google-genai SDK -> pip install google-genai"
            ) from e
        self.name = f"gemini:{model}"
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.models.embed_content(model=self._model, contents=texts)
        return [list(e.values) for e in resp.embeddings]


class SentenceTransformerBackend:
    """Local semantic embeddings. Fallback when Gemini isn't configured."""

    semantic = True

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.name = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]


class HashingBackend:
    """Pure-Python fallback: L2-normalized hashed bag-of-words. Lexical, not semantic."""

    semantic = False
    name = "hashing-fallback"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(t):
                vec[hash(tok) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


def get_default_backend(model_name: str | None = None) -> EmbeddingBackend:
    """Priority: Gemini (if JOB_MATCHER key + SDK are present) ->
    sentence-transformers -> pure-Python hashing fallback."""
    api_key = os.environ.get("JOB_MATCHER")
    if api_key:
        try:
            return GeminiEmbeddingBackend(api_key, model=model_name or "gemini-embedding-001")
        except Exception as e:  # noqa: BLE001
            print(
                f"[note] gemini backend unavailable ({type(e).__name__}: {e}); "
                f"trying sentence-transformers next.",
                file=sys.stderr,
            )
    try:
        return SentenceTransformerBackend(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:  # noqa: BLE001
        print(
            f"[note] sentence-transformers unavailable ({type(e).__name__}); "
            f"falling back to lexical hashing. Install it for semantic matching.",
            file=sys.stderr,
        )
        return HashingBackend()


# --------------------------------------------------------------------------- #
# similarity + filters
# --------------------------------------------------------------------------- #


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def apply_filters(jobs: list[Job], f: JobFilters) -> list[Job]:
    now = datetime.now(UTC)
    out: list[Job] = []
    for job in jobs:
        if f.location and f.location.lower() not in job.location.lower():
            continue
        if f.remote_only and job.remote is not True:
            continue
        if f.posted_within_hours is not None:
            if job.posted_at is None:
                continue
            age_h = (now - job.posted_at).total_seconds() / 3600.0
            if age_h > f.posted_within_hours:
                continue
        blob = f"{job.title}\n{job.description}"
        if f.must_have and not all(contains_term(blob, kw) for kw in f.must_have):
            continue
        if f.exclude and any(contains_term(blob, kw) for kw in f.exclude):
            continue
        out.append(job)
    return out


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #


def rank(
    profile: ResumeProfile,
    jobs: list[Job],
    backend: EmbeddingBackend,
    filters: JobFilters | None = None,
    top_k: int = 20,
    keyword_weight: float = 0.0,
) -> list[MatchResult]:
    """Filter, embed, score, and return the top_k matches.

    final = (1 - keyword_weight) * semantic + keyword_weight * skill_overlap
    keyword_weight=0 -> pure semantic ranking (skills still shown as reasons).
    """
    candidates = apply_filters(jobs, filters) if filters else list(jobs)
    if not candidates:
        return []

    job_texts = [j.search_text() for j in candidates]
    embeddings = backend.embed([profile.query_text] + job_texts)
    resume_vec, job_vecs = embeddings[0], embeddings[1:]

    n_skills = max(len(profile.skills), 1)
    results: list[MatchResult] = []
    for job, jv in zip(candidates, job_vecs, strict=True):
        sem = cosine(resume_vec, jv)
        blob = f"{job.title}\n{job.description}"
        hits = [s for s in profile.skills if contains_term(blob, s)]
        overlap = len(hits) / n_skills
        final = (1 - keyword_weight) * sem + keyword_weight * overlap
        results.append(
            MatchResult(job=job, semantic_score=sem, keyword_hits=hits, final_score=final)
        )

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:top_k]
