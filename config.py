"""
Shared configuration for experimental (agent) and control (legacy) pipelines.
Load once via get_config(); validates required environment on access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root (this file's directory)
ROOT_DIR = Path(__file__).resolve().parent

# Load .env from project root if present
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration shared by all entry scripts."""

    # --- Gemini (same model for both pipelines) ---
    gemini_api_key: str
    gemini_model: str
    gemini_fallback_model: str | None
    gemini_embedding_model: str
    gemini_max_retries: int
    gemini_retry_base_sec: float

    # --- Paths ---
    root_dir: Path
    data_dir: Path
    chroma_dir: Path
    documents_dir: Path
    exports_dir: Path

    # --- Chroma collections ---
    collection_proposed: str
    collection_legacy: str

    # --- Crawling ---
    crawl_mode: str  # "http" (no Playwright) or "browser"
    crawl_timeout_sec: int
    max_collect_retries: int
    max_urls_per_run: int

    # --- Chunking ---
    legacy_chunk_tokens: int
    legacy_chars_per_token: int
    adaptive_max_chunk_chars: int
    adaptive_min_chunk_chars: int
    proposed_max_chunks_per_doc: int

    # --- RAG retrieval ---
    rag_top_k: int

    # --- Verifier / Approver thresholds (0.0–1.0) ---
    verifier_min_semantic_similarity: float
    verifier_min_llm_score: float
    approver_min_overall: float

    @property
    def legacy_chunk_chars(self) -> int:
        return self.legacy_chunk_tokens * self.legacy_chars_per_token


def _require(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or not str(value).strip():
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and set your Gemini API key."
        )
    return str(value).strip()


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def load_config() -> Config:
    """Build and validate configuration from environment."""
    root = ROOT_DIR
    data = root / "data"
    chroma = data / "chroma"
    documents = root / "Documents"
    exports = data / "exports"

    for path in (data, chroma, documents, exports):
        path.mkdir(parents=True, exist_ok=True)

    api_key = _require("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
    fallback_raw = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash").strip()
    gemini_fallback = fallback_raw if fallback_raw else None
    embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()

    if not model:
        raise ValueError("GEMINI_MODEL must not be empty.")
    if not embed_model:
        raise ValueError("GEMINI_EMBEDDING_MODEL must not be empty.")

    verifier_sem = _float_env("VERIFIER_MIN_SEMANTIC_SIMILARITY", 0.45)
    verifier_llm = _float_env("VERIFIER_MIN_LLM_SCORE", 0.55)
    approver_min = _float_env("APPROVER_MIN_OVERALL", 0.60)

    for label, val in (
        ("VERIFIER_MIN_SEMANTIC_SIMILARITY", verifier_sem),
        ("VERIFIER_MIN_LLM_SCORE", verifier_llm),
        ("APPROVER_MIN_OVERALL", approver_min),
    ):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{label} must be between 0.0 and 1.0, got {val}")

    crawl_mode = os.getenv("CRAWL_MODE", "http").strip().lower()
    if crawl_mode not in ("http", "browser"):
        raise ValueError('CRAWL_MODE must be "http" or "browser".')

    return Config(
        gemini_api_key=api_key,
        gemini_model=model,
        gemini_fallback_model=gemini_fallback,
        gemini_embedding_model=embed_model,
        gemini_max_retries=_int_env("GEMINI_MAX_RETRIES", 5),
        gemini_retry_base_sec=_float_env("GEMINI_RETRY_BASE_SEC", 2.0),
        root_dir=root,
        data_dir=data,
        chroma_dir=chroma,
        documents_dir=documents,
        exports_dir=exports,
        collection_proposed=os.getenv("CHROMA_COLLECTION_PROPOSED", "rag_proposed").strip(),
        collection_legacy=os.getenv("CHROMA_COLLECTION_LEGACY", "rag_legacy").strip(),
        crawl_mode=crawl_mode,
        crawl_timeout_sec=_int_env("CRAWL_TIMEOUT_SEC", 60),
        max_collect_retries=_int_env("MAX_COLLECT_RETRIES", 2),
        max_urls_per_run=_int_env("MAX_URLS_PER_RUN", 5),
        legacy_chunk_tokens=_int_env("LEGACY_CHUNK_TOKENS", 500),
        legacy_chars_per_token=_int_env("LEGACY_CHARS_PER_TOKEN", 4),
        adaptive_max_chunk_chars=_int_env("ADAPTIVE_MAX_CHUNK_CHARS", 3500),
        adaptive_min_chunk_chars=_int_env("ADAPTIVE_MIN_CHUNK_CHARS", 200),
        proposed_max_chunks_per_doc=_int_env("PROPOSED_MAX_CHUNKS_PER_DOC", 3),
        rag_top_k=_int_env("RAG_TOP_K", 5),
        verifier_min_semantic_similarity=verifier_sem,
        verifier_min_llm_score=verifier_llm,
        approver_min_overall=approver_min,
    )


_cached: Config | None = None


def get_config() -> Config:
    """Singleton-style access; validates on first call."""
    global _cached
    if _cached is None:
        _cached = load_config()
    return _cached


def reset_config() -> None:
    """Clear cache (useful in tests)."""
    global _cached
    _cached = None
