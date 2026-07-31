"""Core data models shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Job:
    """A single job posting, normalized across ATS providers."""

    source: str  # 'greenhouse' | 'lever' | 'ashby'
    company: str  # the board token / slug it came from
    title: str
    url: str  # apply / hosted URL
    location: str = ""
    department: str = ""
    team: str = ""
    employment_type: str = ""
    remote: bool | None = None  # None = unknown
    description: str = ""  # plain text
    posted_at: datetime | None = None
    compensation: str = ""
    external_id: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def dedupe_key(self) -> str:
        """Companies repost roles; collapse on company+title+location."""
        return f"{self.company}|{self.title.strip().lower()}|{self.location.strip().lower()}"

    def search_text(self) -> str:
        """Text handed to the embedding model. Title is weighted by repetition."""
        parts = [self.title, self.title, self.department, self.team, self.description]
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict:
        d = {
            "source": self.source,
            "company": self.company,
            "title": self.title,
            "url": self.url,
            "location": self.location,
            "department": self.department,
            "team": self.team,
            "employment_type": self.employment_type,
            "remote": self.remote,
            "compensation": self.compensation,
            "external_id": self.external_id,
            "posted_at": self.posted_at.isoformat() if self.posted_at else None,
            "description": self.description,
        }
        return d


@dataclass
class ResumeProfile:
    """A parsed resume. `query_text` is what gets embedded."""

    text: str
    skills: list[str] = field(default_factory=list)
    query_text: str = ""

    def __post_init__(self):
        if not self.query_text:
            self.query_text = self.text


@dataclass
class MatchResult:
    job: Job
    semantic_score: float  # cosine similarity, 0..1-ish
    keyword_hits: list[str] = field(default_factory=list)
    final_score: float = 0.0
    rerank_score: float | None = None  # cross-encoder score, populated only when reranked

    def to_dict(self) -> dict:
        d = self.job.to_dict()
        d["semantic_score"] = round(self.semantic_score, 4)
        d["final_score"] = round(self.final_score, 4)
        d["keyword_hits"] = self.keyword_hits
        if self.rerank_score is not None:
            d["rerank_score"] = round(self.rerank_score, 4)
        return d


@dataclass
class JobFilters:
    """Hard filters applied before ranking."""

    location: str | None = None  # case-insensitive substring
    remote_only: bool = False
    posted_within_hours: float | None = None
    must_have: list[str] = field(default_factory=list)  # all must appear
    exclude: list[str] = field(default_factory=list)  # none may appear
