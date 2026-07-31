"""Offline tests for the cross-encoder rerank integration.

Uses a FakeReranker instead of downloading the real cross-encoder model, so
these run in the same fast CI env as the other tests. The class under test
(CrossEncoderReranker) has a lazy import guard that is exercised implicitly
when sentence-transformers is absent -- not covered here.

Run: python tests/test_reranker.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ats_matcher.matching import HashingBackend, rank  # noqa: E402
from ats_matcher.models import Job, ResumeProfile  # noqa: E402


class FakeReranker:
    """Test double that returns a known score per pair. Score = length of the
    job text (deterministic and depends only on the pair, so we can predict
    ordering without touching real ML models)."""

    name = "fake-reranker"

    def __init__(self):
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [float(len(job_text)) for _, job_text in pairs]


def _jobs(n: int) -> list[Job]:
    # Each job has a distinct description length so scoring is deterministic.
    return [
        Job(
            source="greenhouse",
            company="acme",
            title=f"role {i}",
            url=f"https://example.com/{i}",
            description="x" * (10 + i),  # ascending length -> ascending rerank score
        )
        for i in range(n)
    ]


def _profile() -> ResumeProfile:
    return ResumeProfile(text="python pytorch ML engineer", query_text="python pytorch ML engineer")


def test_no_reranker_leaves_behavior_unchanged():
    jobs = _jobs(5)
    profile = _profile()
    backend = HashingBackend(dim=128)
    results = rank(profile, jobs, backend, top_k=3)
    assert len(results) == 3
    assert all(r.rerank_score is None for r in results)
    print("no reranker: rerank_score stays None, top_k respected: OK")


def test_reranker_flips_ranking_by_its_own_score():
    jobs = _jobs(5)
    profile = _profile()
    backend = HashingBackend(dim=128)
    reranker = FakeReranker()
    results = rank(profile, jobs, backend, top_k=3, reranker=reranker)

    # FakeReranker scores by len(job_text), so the longest description wins.
    # Longest = "x" * 14 = the job with title "role 4".
    assert results[0].job.title == "role 4"
    assert results[1].job.title == "role 3"
    assert results[2].job.title == "role 2"
    # Rerank scores propagated onto each returned result.
    assert all(r.rerank_score is not None for r in results)
    # final_score was replaced with the rerank score. The reranker sees
    # "title\ndescription", so score = length of that combined text.
    expected = float(len(f"{jobs[4].title}\n{jobs[4].description}"))
    assert results[0].final_score == results[0].rerank_score == expected
    print("reranker rewrites ordering + populates rerank_score: OK")


def test_rerank_top_n_bounds_the_rerank_window():
    jobs = _jobs(10)
    profile = _profile()
    backend = HashingBackend(dim=128)
    reranker = FakeReranker()
    rank(profile, jobs, backend, top_k=3, reranker=reranker, rerank_top_n=4)
    # Exactly one batch of 4 pairs sent to the reranker (top_n=4 out of 10).
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0]) == 4
    print("rerank_top_n bounds the batch: OK")


def test_reranker_never_called_when_no_candidates_survive_filters():
    profile = _profile()
    backend = HashingBackend(dim=128)
    reranker = FakeReranker()
    results = rank(profile, [], backend, top_k=3, reranker=reranker)
    assert results == []
    assert reranker.calls == []
    print("empty candidates: reranker not called: OK")


def test_pairs_carry_resume_query_text_and_job_content():
    jobs = _jobs(1)
    profile = _profile()
    backend = HashingBackend(dim=128)
    reranker = FakeReranker()
    rank(profile, jobs, backend, top_k=1, reranker=reranker)
    (pair,) = reranker.calls[0]
    resume_txt, job_txt = pair
    assert resume_txt == profile.query_text
    assert jobs[0].title in job_txt
    assert jobs[0].description in job_txt
    print("pair shape: (resume_query_text, title+description): OK")


if __name__ == "__main__":
    test_no_reranker_leaves_behavior_unchanged()
    test_reranker_flips_ranking_by_its_own_score()
    test_rerank_top_n_bounds_the_rerank_window()
    test_reranker_never_called_when_no_candidates_survive_filters()
    test_pairs_carry_resume_query_text_and_job_content()
    print("\nAll reranker tests passed.")
