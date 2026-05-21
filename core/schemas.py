"""Data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SourceType = Literal["web", "upload"]


@dataclass
class RawDocument:
    """Normalized document before verification."""

    doc_id: str
    source_type: SourceType
    title: str
    text: str
    query: str
    source_ref: str  # URL or file path
    metadata: dict[str, Any] = field(default_factory=dict)


ExtractMode = Literal["section_only", "full_page", "follow_url", "index_line"]


@dataclass
class VerifierReport:
    keyword_match: float
    source_traceability: float
    content_integrity: float
    semantic_similarity: float
    overall_score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    chunking_hint: str = "section"  # section | qa | table_heavy
    relevance_scope: str = "section"
    extract_mode: ExtractMode = "section_only"
    target_fragment: str = ""
    target_heading: str = ""
    follow_url: str = ""
    needs_follow_crawl: bool = False
    too_broad: bool = False
    collector_feedback: str = ""
    ready_for_indexing: bool = False


@dataclass
class ApproverDecision:
    approved: bool
    overall_score: float
    feedback: str = ""
    reasons: list[str] = field(default_factory=list)
    extract_mode: ExtractMode = "section_only"
    follow_url: str = ""
    target_fragment: str = ""
    target_heading: str = ""


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
