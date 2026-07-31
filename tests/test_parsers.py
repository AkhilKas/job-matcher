"""Offline tests for the ATS parsers, using mock payloads shaped like the real
responses (Greenhouse/Lever public APIs and the official Ashby docs example).

Run: python tests/test_parsers.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ats_matcher.matching import HashingBackend, apply_filters, cosine  # noqa: E402
from ats_matcher.models import Job, JobFilters  # noqa: E402
from ats_matcher.providers import (
    parse_ashby,
    parse_greenhouse,
    parse_lever,
    parse_spec,
)  # noqa: E402
from ats_matcher.textutil import html_to_text  # noqa: E402


def test_greenhouse():
    data = {
        "jobs": [
            {
                "id": 12345,
                "title": "Senior ML Engineer",
                "updated_at": "2026-07-29T18:30:00-04:00",
                "location": {"name": "Remote - US"},
                "absolute_url": "https://boards.greenhouse.io/demo/jobs/12345",
                "content": "&lt;p&gt;Build &lt;strong&gt;ML&lt;/strong&gt; systems.&lt;/p&gt;&lt;p&gt;Great team.&lt;/p&gt;",
                "departments": [{"id": 1, "name": "Engineering"}],
            }
        ]
    }
    jobs = parse_greenhouse(data, "demo")
    assert len(jobs) == 1
    j = jobs[0]
    assert j.source == "greenhouse"
    assert j.title == "Senior ML Engineer"
    assert j.location == "Remote - US"
    assert j.department == "Engineering"
    assert j.url.endswith("/12345")
    assert "Build ML systems." in j.description  # entities decoded, tags stripped
    assert "<" not in j.description
    assert j.remote is True  # inferred from the location string ("Remote - US")
    assert j.posted_at is not None and j.posted_at.year == 2026
    print("greenhouse: OK")


def test_lever():
    data = [
        {
            "id": "abc-123",
            "text": "MLOps Engineer",
            "categories": {
                "commitment": "Full-time",
                "department": "Infra",
                "team": "MLOps",
                "location": "Remote",
            },
            "descriptionPlain": "Own the ML infrastructure with Airflow and MLflow.",
            "description": "<p>ignored when plain present</p>",
            "createdAt": 1753800000000,
            "hostedUrl": "https://jobs.lever.co/demo/abc-123",
            "applyUrl": "https://jobs.lever.co/demo/abc-123/apply",
            "workplaceType": "remote",
        }
    ]
    jobs = parse_lever(data, "demo")
    assert len(jobs) == 1
    j = jobs[0]
    assert j.source == "lever"
    assert j.title == "MLOps Engineer"
    assert j.employment_type == "Full-time"
    assert j.team == "MLOps"
    assert j.remote is True
    assert j.description.startswith("Own the ML infrastructure")
    assert j.url.endswith("abc-123")  # hostedUrl preferred
    assert j.posted_at is not None and j.posted_at.tzinfo is not None
    print("lever: OK")


def test_ashby():
    # Shape taken from developers.ashbyhq.com/docs/public-job-posting-api
    data = {
        "apiVersion": "1",
        "jobs": [
            {
                "title": "Product Manager",
                "location": "Houston, TX",
                "department": "Product",
                "team": "Growth",
                "isListed": True,
                "isRemote": True,
                "workplaceType": "Remote",
                "descriptionHtml": "<p>Join our team</p>",
                "descriptionPlain": "Join our team",
                "publishedAt": "2026-07-28T16:21:55.393+00:00",
                "employmentType": "FullTime",
                "jobUrl": "https://jobs.ashbyhq.com/demo/pm",
                "applyUrl": "https://jobs.ashbyhq.com/demo/pm/apply",
                "compensation": {"compensationTierSummary": "$120K - $150K"},
            },
            {
                "title": "Hidden Role",
                "location": "NYC",
                "isListed": False,  # should be skipped
                "descriptionPlain": "secret",
            },
        ],
    }
    jobs = parse_ashby(data, "demo")
    assert len(jobs) == 1, "isListed=False should be filtered out"
    j = jobs[0]
    assert j.source == "ashby"
    assert j.title == "Product Manager"
    assert j.department == "Product"
    assert j.team == "Growth"
    assert j.employment_type == "FullTime"
    assert j.remote is True
    assert j.description == "Join our team"
    assert j.url.endswith("/apply")  # applyUrl preferred
    assert j.compensation == "$120K - $150K"
    assert j.posted_at is not None and j.posted_at.year == 2026
    print("ashby: OK")


def test_spec_parsing():
    assert parse_spec("greenhouse:stripe") == ("greenhouse", "stripe")
    assert parse_spec("Lever: spotify ") == ("lever", "spotify")
    for bad in ("stripe", "unknown:x", "greenhouse:"):
        try:
            parse_spec(bad)
            raise AssertionError(f"expected failure for {bad!r}")
        except ValueError:
            pass
    print("spec parsing: OK")


def test_html_to_text():
    assert html_to_text("&lt;b&gt;hi&lt;/b&gt;") == "hi"
    assert "<" not in html_to_text("<div>a</div><div>b</div>")
    print("html_to_text: OK")


def test_cosine_and_hashing():
    b = HashingBackend(dim=256)
    v = b.embed(["python pytorch mlflow", "python pytorch mlflow", "sales quota crm pipeline"])
    same = cosine(v[0], v[1])
    diff = cosine(v[0], v[2])
    assert abs(same - 1.0) < 1e-9, f"identical text should be ~1.0, got {same}"
    assert diff < same, "unrelated text should score lower"
    print(f"cosine/hashing: OK (same={same:.3f}, diff={diff:.3f})")


def test_filters():
    jobs = [
        Job(
            source="x",
            company="c",
            title="ML Engineer",
            url="",
            location="Boston, MA",
            remote=False,
            description="pytorch mlops",
        ),
        Job(
            source="x",
            company="c",
            title="Remote ML",
            url="",
            location="Remote",
            remote=True,
            description="pytorch",
        ),
        Job(
            source="x",
            company="c",
            title="Sales",
            url="",
            location="Boston, MA",
            remote=False,
            description="quota crm",
        ),
    ]
    assert len(apply_filters(jobs, JobFilters(location="boston"))) == 2
    assert len(apply_filters(jobs, JobFilters(remote_only=True))) == 1
    assert len(apply_filters(jobs, JobFilters(must_have=["pytorch"]))) == 2
    assert len(apply_filters(jobs, JobFilters(exclude=["quota"]))) == 2
    print("filters: OK")


if __name__ == "__main__":
    test_greenhouse()
    test_lever()
    test_ashby()
    test_spec_parsing()
    test_html_to_text()
    test_cosine_and_hashing()
    test_filters()
    print("\nAll parser tests passed.")
