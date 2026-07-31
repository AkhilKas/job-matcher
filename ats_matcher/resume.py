"""Load a resume and turn it into a ResumeProfile the matcher can use."""

from __future__ import annotations

import os

from .models import ResumeProfile
from .textutil import contains_term, normalize_ws

# A starter skills vocabulary used only to surface "why this matched" hints and to
# power the optional keyword-blend. It is deliberately editable -- extend it for
# your field. It does NOT drive the semantic ranking, so gaps here are low-stakes.
DEFAULT_SKILLS = [
    # languages
    "python", "typescript", "javascript", "java", "c++", "go", "rust", "scala", "sql",
    # ml / ds
    "pytorch", "tensorflow", "scikit-learn", "keras", "jax", "numpy", "pandas",
    "machine learning", "deep learning", "nlp", "computer vision", "llm", "transformers",
    "lstm", "reinforcement learning", "recommendation", "embeddings", "rag",
    # mlops / infra
    "mlflow", "airflow", "kubeflow", "dvc", "docker", "kubernetes", "terraform",
    "vertex ai", "sagemaker", "gcp", "aws", "azure", "spark", "kafka",
    "ci/cd", "github actions", "mlops", "evidently", "great expectations",
    # backend / web
    "fastapi", "flask", "django", "react", "node", "graphql", "rest", "grpc",
    "postgres", "redis", "mongodb", "microservices",
]


def load_resume_text(path: str) -> str:
    """Read resume text from .txt/.md or .pdf. Raises with a clear hint on failure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md", ".text", ""):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return normalize_ws(f.read())
    if ext == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"unsupported resume type {ext!r}; use .txt, .md, or .pdf")


def _read_pdf(path: str) -> str:
    try:
        import pdfplumber  # lazy: only needed for PDF resumes
    except ImportError as e:
        raise ImportError(
            "reading a PDF resume needs pdfplumber -> pip install pdfplumber "
            "(or convert your resume to .txt)"
        ) from e
    chunks: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            chunks.append(page.extract_text() or "")
    text = "\n".join(chunks)
    if not text.strip():
        raise ValueError(
            "no text extracted from the PDF (it may be a scanned image; "
            "OCR it or paste the text into a .txt file)"
        )
    return normalize_ws(text)


def extract_skills(text: str, vocab: list[str] | None = None) -> list[str]:
    """Heuristic: which known skills appear in the resume, order preserved."""
    vocab = vocab or DEFAULT_SKILLS
    return [s for s in vocab if contains_term(text, s)]


def build_profile(text: str, vocab: list[str] | None = None) -> ResumeProfile:
    text = normalize_ws(text)
    return ResumeProfile(text=text, skills=extract_skills(text, vocab), query_text=text)


# --------------------------------------------------------------------------- #
# OPTIONAL: structure the resume with Gemini for better filtering/matching.
# Off the default path. Requires your own credentials + the google-genai SDK.
# Verify the model name and SDK call against your Vertex/AI Studio setup before
# relying on it -- this is a scaffold, not tested here.
# --------------------------------------------------------------------------- #

_STRUCTURE_PROMPT = """You are parsing a resume into JSON. Return ONLY valid JSON,
no markdown fences, with these keys:
  "titles": string[]          (roles the person has held or targets)
  "seniority": string         (one of: intern, junior, mid, senior, staff, lead)
  "years_experience": number
  "skills": string[]          (concrete tools/technologies)
  "domains": string[]         (e.g. "computer vision", "mlops", "fintech")
Resume:
---
{resume}
---"""


def structure_resume_gemini(text: str, model: str = "gemini-2.5-flash") -> dict:
    """Optional LLM structuring. Returns a dict; falls back to {} on any error."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise ImportError("pip install google-genai to use structure_resume_gemini") from e
    import json

    client = genai.Client()  # reads GEMINI_API_KEY / Vertex config from env
    resp = client.models.generate_content(
        model=model,
        contents=_STRUCTURE_PROMPT.format(resume=text[:12000]),
    )
    raw = (resp.text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
