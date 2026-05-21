"""Verifier agent: scores + too_broad rejection with collector feedback (LLM only)."""

from __future__ import annotations

from urllib.parse import urlparse

from config import Config, get_config
from core.gemini import GeminiService
from core.schemas import ExtractMode, RawDocument, VerifierReport
from core.section_slice import list_markdown_links
from core.url_focus import extract_focused_text


class VerifierAgent:
    def __init__(
        self,
        gemini: GeminiService | None = None,
        cfg: Config | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.gemini = gemini or GeminiService(self.cfg)

    def verify(self, doc: RawDocument) -> VerifierReport:
        excerpt, focused = self._excerpt(doc)
        if focused:
            frag = urlparse(doc.source_ref).fragment
            print(f"[verifier] Excerpt aligned to URL #{frag}")
        semantic = self._semantic_similarity(doc, excerpt)
        data = self._llm_review(doc, excerpt, focused)

        keyword_match = float(data.get("keyword_match", 0))
        source_traceability = float(data.get("source_traceability", 0))
        content_integrity = float(data.get("content_integrity", 0))
        semantic_llm = float(data.get("semantic_similarity", semantic))

        overall = (
            keyword_match * 0.25
            + source_traceability * 0.2
            + content_integrity * 0.25
            + max(semantic, semantic_llm) * 0.3
        )

        reasons: list[str] = [str(r) for r in (data.get("reasons") or [])]
        too_broad = bool(data.get("too_broad", False))
        ready = bool(data.get("ready_for_indexing", False))
        collector_feedback = str(data.get("collector_feedback") or "").strip()

        scores_ok = (
            keyword_match >= self.cfg.verifier_min_llm_score * 0.85
            and source_traceability >= 0.4
            and content_integrity >= 0.5
            and max(semantic, semantic_llm) >= self.cfg.verifier_min_semantic_similarity
            and overall >= self.cfg.verifier_min_llm_score
        )

        passed = scores_ok and ready and not too_broad

        if too_broad and not collector_feedback:
            collector_feedback = (
                "Page is too broad for the query. Crawl only the single subsection "
                "that answers the query, not the full chapter or site index."
            )

        if not passed:
            if too_broad:
                reasons.append("too_broad: unnecessary content for this query")
            if not ready:
                reasons.append("not ready for indexing at current scope")
            if not scores_ok:
                if max(semantic, semantic_llm) < self.cfg.verifier_min_semantic_similarity:
                    reasons.append("semantic similarity below threshold")
                if keyword_match < self.cfg.verifier_min_llm_score * 0.85:
                    reasons.append("keyword match too low")

        extract_mode = str(data.get("extract_mode") or "section_only")
        if extract_mode not in ("section_only", "full_page", "follow_url", "index_line"):
            extract_mode = "section_only"

        if too_broad:
            print(f"[verifier] REJECT (too broad) → feedback for Collector")
            if collector_feedback:
                print(f"[verifier] Collector feedback: {collector_feedback[:200]}...")

        return VerifierReport(
            keyword_match=keyword_match,
            source_traceability=source_traceability,
            content_integrity=content_integrity,
            semantic_similarity=max(semantic, semantic_llm),
            overall_score=overall,
            passed=passed,
            reasons=reasons,
            chunking_hint=str(data.get("chunking_hint") or "section"),
            relevance_scope=str(data.get("relevance_scope") or "section"),
            extract_mode=extract_mode,  # type: ignore[arg-type]
            target_fragment=str(data.get("target_fragment") or ""),
            target_heading=str(data.get("target_heading") or ""),
            follow_url=str(data.get("follow_url") or ""),
            needs_follow_crawl=bool(data.get("needs_follow_crawl", False)),
            too_broad=too_broad,
            collector_feedback=collector_feedback,
            ready_for_indexing=ready,
        )

    def _excerpt(self, doc: RawDocument) -> tuple[str, bool]:
        return extract_focused_text(doc.text, doc.source_ref, max_chars=8000)

    def _semantic_similarity(self, doc: RawDocument, excerpt: str) -> float:
        q_emb = self.gemini.embed(doc.query)
        d_emb = self.gemini.embed(f"{doc.title}\n{excerpt[:4000]}")
        return self.gemini.cosine_similarity(q_emb, d_emb)

    def _llm_review(self, doc: RawDocument, excerpt: str, focused: bool) -> dict:
        links = list_markdown_links(doc.text[:10000], doc.source_ref)
        link_preview = "\n".join(
            f"  - {x['label'][:60]} | {x['url']}" for x in links[:25]
        ) or "  (none)"

        focus_note = (
            "Excerpt starts at URL fragment."
            if focused
            else "Excerpt is from the beginning of the crawled page."
        )

        prompt = f"""
You are the Verifier agent in a multi-agent RAG pipeline.

User query: {doc.query}
Crawled URL: {doc.source_ref}
Title: {doc.title}
{focus_note}

Sample links on page (for narrowing):
{link_preview}

Document excerpt for review:
{excerpt[:5500]}

Tasks:
1) Score 0.0-1.0: keyword_match, source_traceability, content_integrity, semantic_similarity
2) Decide if this crawl is TOO BROAD for the query (e.g. full chapter when user only needs one subsection like "4.2 for 문" from a TOC page).
3) If too_broad=true, write collector_feedback telling the Collector agent how to narrow (which subsection URL to crawl, in natural language). This feedback will be sent to the Collector — be specific.
4) ready_for_indexing=true ONLY when the current excerpt already contains just what is needed to answer the query (minimal scope). If the page is an index/TOC listing many sections, set too_broad=true and ready_for_indexing=false.

Return JSON:
{{
  "keyword_match": 0.0,
  "source_traceability": 0.0,
  "content_integrity": 0.0,
  "semantic_similarity": 0.0,
  "too_broad": false,
  "ready_for_indexing": false,
  "collector_feedback": "",
  "follow_url": "",
  "target_fragment": "",
  "target_heading": "",
  "needs_follow_crawl": false,
  "extract_mode": "section_only",
  "relevance_scope": "section",
  "chunking_hint": "section",
  "reasons": []
}}
"""
        return self.gemini.generate_json(prompt)
