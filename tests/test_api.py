"""Offline tests for the FastAPI backend.

Uses starlette's TestClient against an app built with `create_app` and
explicitly injected HashingBackend + temp SQLite DB. No network, no real
embedding-model download, no live ATS calls -- these tests are CI-safe.

Run: python tests/test_api.py
"""

import io
import os
import sys
import tempfile
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from ats_matcher.api.main import create_app  # noqa: E402
from ats_matcher.matching import HashingBackend  # noqa: E402
from ats_matcher.models import Job  # noqa: E402
from ats_matcher.storage import JobStore  # noqa: E402


def _seed(db_path: str) -> None:
    with JobStore(db_path) as store:
        store.upsert_jobs(
            [
                Job(
                    source="greenhouse",
                    company="acme",
                    title="ML Engineer",
                    url="https://acme.example.com/1",
                    location="Boston, MA",
                    description="Requirements:\n- python\n- pytorch\n- mlflow\n- airflow",
                    remote=False,
                    posted_at=datetime(2026, 7, 28, tzinfo=UTC),
                ),
                Job(
                    source="lever",
                    company="northwind",
                    title="MLOps Engineer",
                    url="https://nw.example.com/2",
                    location="Remote",
                    description="Requirements:\n- airflow\n- mlflow\n- kubernetes\n- docker",
                    remote=True,
                    posted_at=datetime(2026, 7, 29, tzinfo=UTC),
                ),
                Job(
                    source="ashby",
                    company="cobalt",
                    title="Sales Executive",
                    url="https://cobalt.example.com/3",
                    location="New York, NY",
                    description="Requirements:\n- quota\n- crm\n- salesforce\n- pipeline",
                    remote=False,
                    posted_at=datetime(2026, 7, 27, tzinfo=UTC),
                ),
            ]
        )


def _new_client() -> tuple[TestClient, str]:
    tmp = tempfile.mkdtemp(prefix="apitest_")
    db_path = os.path.join(tmp, "jobs.db")
    _seed(db_path)
    app = create_app(backend=HashingBackend(dim=256), db_path=db_path)
    return TestClient(app), db_path


def test_health():
    client, _ = _new_client()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["open_jobs"] == 3
    print("health: OK")


def test_stats():
    client, _ = _new_client()
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_jobs"] == 3
    assert body["open_jobs"] == 3
    assert body["closed_jobs"] == 0
    print("stats: OK")


def test_rank_by_resume_text():
    client, _ = _new_client()
    resume = "Senior ML Engineer with python pytorch mlflow airflow experience."
    r = client.post("/rank", json={"resume_text": resume, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["total_candidates"] == 3
    assert body["reranked"] is False
    assert len(body["results"]) == 3
    top = body["results"][0]["job"]
    # Sales is unambiguously wrong -- ML/MLOps roles should beat it.
    assert top["title"] in ("ML Engineer", "MLOps Engineer")
    # Score fields are populated.
    assert body["results"][0]["semantic_score"] > 0
    assert body["results"][0]["final_score"] > 0
    print("POST /rank basic: OK")


def test_rank_with_remote_only_filter():
    client, _ = _new_client()
    r = client.post(
        "/rank",
        json={
            "resume_text": "python developer with cloud experience",
            "filters": {"remote_only": True},
            "top_k": 10,
        },
    )
    assert r.status_code == 200
    body = r.json()
    for result in body["results"]:
        assert result["job"]["remote"] is True
    print("POST /rank with remote_only filter: OK")


def test_rank_top_k_bounded():
    client, _ = _new_client()
    r = client.post("/rank", json={"resume_text": "python developer", "top_k": 1})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1
    print("top_k bound: OK")


def test_rank_returns_503_when_db_empty():
    tmp = tempfile.mkdtemp(prefix="apitest_empty_")
    db_path = os.path.join(tmp, "empty.db")
    app = create_app(backend=HashingBackend(dim=256), db_path=db_path)
    client = TestClient(app)
    r = client.post("/rank", json={"resume_text": "python developer"})
    assert r.status_code == 503
    assert "No jobs in DB" in r.json()["detail"]
    print("empty DB -> 503: OK")


def test_rank_include_closed():
    client, db_path = _new_client()
    # Manually mark one job closed
    with JobStore(db_path) as store:
        store._conn.execute("UPDATE jobs SET is_open = 0 WHERE title = 'Sales Executive'")
        store._conn.commit()

    default = client.post("/rank", json={"resume_text": "sales quota crm salesforce"})
    assert default.status_code == 200
    default_titles = {res["job"]["title"] for res in default.json()["results"]}
    assert "Sales Executive" not in default_titles  # closed by default

    incl = client.post(
        "/rank", json={"resume_text": "sales quota crm salesforce", "include_closed": True}
    )
    assert incl.status_code == 200
    incl_titles = {res["job"]["title"] for res in incl.json()["results"]}
    assert "Sales Executive" in incl_titles
    print("include_closed toggle: OK")


def test_rank_upload_txt():
    client, _ = _new_client()
    resume_bytes = b"Senior ML Engineer with python and pytorch expertise."
    r = client.post(
        "/rank/upload",
        files={"file": ("resume.txt", io.BytesIO(resume_bytes), "text/plain")},
        data={"top_k": "2"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    print("POST /rank/upload (.txt): OK")


def test_rank_validates_short_resume():
    client, _ = _new_client()
    r = client.post("/rank", json={"resume_text": "hi"})
    assert r.status_code == 422  # pydantic min_length=10
    print("short resume -> 422 validation: OK")


if __name__ == "__main__":
    test_health()
    test_stats()
    test_rank_by_resume_text()
    test_rank_with_remote_only_filter()
    test_rank_top_k_bounded()
    test_rank_returns_503_when_db_empty()
    test_rank_include_closed()
    test_rank_upload_txt()
    test_rank_validates_short_resume()
    print("\nAll API tests passed.")
