"""Offline tests for job-description chunking.

Run: python tests/test_chunking.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ats_matcher.chunking import SECTION_WEIGHTS, chunk_jd  # noqa: E402


def test_no_headers_falls_through_as_other():
    text = "We are a small team building ML systems. Come join us."
    assert chunk_jd(text) == [("other", text)]


def test_recognizes_requirements_and_responsibilities():
    text = """We're hiring an ML engineer.

Requirements:
- 5 years Python
- Deep learning experience

Responsibilities:
- Build ML models
- Ship them to production

Nice to have:
- LLM experience
"""
    result = chunk_jd(text)
    labels = [label for label, _ in result]
    assert labels == ["other", "requirements", "responsibilities", "nice_to_have"]
    # Requirements chunk actually contains the bullets, not the header.
    reqs = next(t for label, t in result if label == "requirements")
    assert "5 years Python" in reqs
    assert "Requirements" not in reqs


def test_header_case_and_punctuation_insensitive():
    text = "REQUIREMENTS\n- Python\n\nBASIC QUALIFICATIONS:\n- Docker"
    result = chunk_jd(text)
    # Second header ("BASIC QUALIFICATIONS") also maps to requirements; two
    # consecutive requirements chunks are fine.
    assert [label for label, _ in result] == ["requirements", "requirements"]


def test_synonyms_map_correctly():
    text = "What you'll do:\n- Ship code\n\nWho you are:\n- 5 years experience"
    result = chunk_jd(text)
    assert [label for label, _ in result] == ["responsibilities", "requirements"]


def test_empty_input():
    assert chunk_jd("") == []
    assert chunk_jd("   \n\n  ") == []


def test_sentence_looking_line_is_not_a_header():
    # A sentence that starts with a keyword shouldn't count as a header.
    text = "Requirements are simple, but you must have grit. Now let's begin."
    result = chunk_jd(text)
    assert result == [("other", text)]


def test_section_weights_shape():
    # Contract for callers: every label chunk_jd emits has a weight.
    emitted_labels = {"requirements", "responsibilities", "nice_to_have", "other"}
    assert emitted_labels.issubset(SECTION_WEIGHTS.keys())
    # 'title' is defined here (used by matching.py) even though chunk_jd never emits it.
    assert "title" in SECTION_WEIGHTS
    # Ordering matters for a sanity check on the weights themselves.
    assert SECTION_WEIGHTS["requirements"] > SECTION_WEIGHTS["responsibilities"]
    assert SECTION_WEIGHTS["responsibilities"] > SECTION_WEIGHTS["nice_to_have"]


if __name__ == "__main__":
    test_no_headers_falls_through_as_other()
    print("no headers: OK")
    test_recognizes_requirements_and_responsibilities()
    print("requirements + responsibilities: OK")
    test_header_case_and_punctuation_insensitive()
    print("case/punctuation insensitive: OK")
    test_synonyms_map_correctly()
    print("synonyms: OK")
    test_empty_input()
    print("empty input: OK")
    test_sentence_looking_line_is_not_a_header()
    print("sentence-like line ignored: OK")
    test_section_weights_shape()
    print("section weights shape: OK")
    print("\nAll chunking tests passed.")
