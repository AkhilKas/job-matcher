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

from .chunking import SECTION_WEIGHTS, chunk_jd
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


# --------------------------------------------------------------------------- #
# reranker (optional second stage)
# --------------------------------------------------------------------------- #


class Reranker(Protocol):
    """A second-stage rescorer that scores (query, candidate) text pairs
    directly, rather than embedding each side separately. Slower per pair
    but typically much more accurate -- use over the top N from the
    embedding stage."""

    name: str

    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class CrossEncoderReranker:
    """Local cross-encoder rerank via sentence-transformers. Higher score = more relevant."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder  # lazy: optional dep
        except ImportError as e:
            raise ImportError(
                "cross-encoder rerank needs sentence-transformers -> pip install sentence-transformers"
            ) from e
        self.name = f"cross-encoder:{model_name}"
        self._model = CrossEncoder(model_name)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


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
    reranker: Reranker | None = None,
    rerank_top_n: int = 50,
) -> list[MatchResult]:
    """Filter, embed, score, and return the top_k matches.

    Each job is embedded as: a title-prefix chunk (title + department + team)
    plus one or more description chunks produced by `chunk_jd` (requirements /
    responsibilities / nice_to_have / other). All chunks and the resume go
    through backend.embed in a single batched call. The semantic score for a
    job is a weighted average of chunk-vs-resume cosine similarities using
    SECTION_WEIGHTS -- so a strong match on 'requirements' outranks the same
    match on a 'nice_to_have' bullet.

    final = (1 - keyword_weight) * semantic + keyword_weight * skill_overlap
    """
    candidates = apply_filters(jobs, filters) if filters else list(jobs)
    if not candidates:
        return []

    # Per-job: list of (chunk_text, weight).
    per_job_parts: list[list[tuple[str, float]]] = []
    for job in candidates:
        parts: list[tuple[str, float]] = []
        title_prefix = "\n".join(p for p in [job.title, job.department, job.team] if p)
        if title_prefix:
            parts.append((title_prefix, SECTION_WEIGHTS["title"]))
        for label, text in chunk_jd(job.description):
            parts.append((text, SECTION_WEIGHTS.get(label, SECTION_WEIGHTS["other"])))
        # Guarantee at least one entry so the batching below stays uniform.
        if not parts:
            parts.append(("", SECTION_WEIGHTS["other"]))
        per_job_parts.append(parts)

    # Flatten into one batched embed call: [resume, *all chunks].
    flat_texts: list[str] = [profile.query_text]
    boundaries: list[int] = []  # cumulative end index in flat_texts per job
    for parts in per_job_parts:
        flat_texts.extend(text for text, _ in parts)
        boundaries.append(len(flat_texts))

    all_vecs = backend.embed(flat_texts)
    resume_vec = all_vecs[0]

    results: list[MatchResult] = []
    start = 1
    n_skills = max(len(profile.skills), 1)
    for job, parts, end in zip(candidates, per_job_parts, boundaries, strict=True):
        chunk_vecs = all_vecs[start:end]
        weights = [w for _, w in parts]
        sims = [cosine(resume_vec, v) for v in chunk_vecs]
        total_w = sum(weights) or 1.0
        sem = sum(s * w for s, w in zip(sims, weights, strict=True)) / total_w
        start = end

        blob = f"{job.title}\n{job.description}"
        hits = [s for s in profile.skills if contains_term(blob, s)]
        overlap = len(hits) / n_skills
        final = (1 - keyword_weight) * sem + keyword_weight * overlap
        results.append(
            MatchResult(job=job, semantic_score=sem, keyword_hits=hits, final_score=final)
        )

    results.sort(key=lambda r: r.final_score, reverse=True)

    if reranker is not None and results:
        # Rescore the top N with the cross-encoder, then take top_k from that.
        # Anything beyond rerank_top_n is dropped -- pumping the tail through
        # the rerank isn't worth the cost, and keeping unranked items mixed in
        # would create discontinuities in the final ordering.
        top_n = results[:rerank_top_n]
        pairs = [(profile.query_text, f"{r.job.title}\n{r.job.description}".strip()) for r in top_n]
        rerank_scores = reranker.score(pairs)
        for r, s in zip(top_n, rerank_scores, strict=True):
            r.rerank_score = s
            r.final_score = s
        top_n.sort(key=lambda r: r.final_score, reverse=True)
        return top_n[:top_k]

    return results[:top_k]
