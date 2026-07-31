"""Pydantic v2 request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FiltersIn(BaseModel):
    location: str | None = None
    remote_only: bool = False
    posted_within_hours: float | None = None
    must_have: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class RankRequest(BaseModel):
    resume_text: str = Field(..., min_length=10, description="raw resume text")
    filters: FiltersIn = Field(default_factory=FiltersIn)
    top_k: int = Field(20, ge=1, le=200)
    keyword_weight: float = Field(0.0, ge=0.0, le=1.0)
    rerank: bool = False
    rerank_top_n: int = Field(50, ge=1, le=500)
    include_closed: bool = False


class JobOut(BaseModel):
    source: str
    company: str
    title: str
    url: str
    location: str = ""
    department: str = ""
    team: str = ""
    employment_type: str = ""
    remote: bool | None = None
    description: str = ""
    posted_at: str | None = None
    compensation: str = ""


class RankedJob(BaseModel):
    job: JobOut
    semantic_score: float
    final_score: float
    keyword_hits: list[str]
    rerank_score: float | None = None


class RankResponse(BaseModel):
    results: list[RankedJob]
    total_candidates: int
    reranked: bool


class HealthResponse(BaseModel):
    status: str
    db_path: str
    open_jobs: int


class StatsResponse(BaseModel):
    db_path: str
    total_jobs: int
    open_jobs: int
    closed_jobs: int
