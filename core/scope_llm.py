"""LLM-driven scope narrowing and section extraction (no keyword hardcoding)."""

from __future__ import annotations

from urllib.parse import urlparse

from core.gemini import GeminiService
from core.schemas import ApproverDecision, RawDocument, VerifierReport
from core.section_slice import (
    list_markdown_links,
    slice_by_fragment_hint,
    slice_by_heading,
    slice_by_line_range,
)


def collector_narrow_from_feedback(
    gemini: GeminiService,
    query: str,
    doc: RawDocument,
    report: VerifierReport,
) -> dict:
    """
    Collector agent: Verifier said too broad → pick ONE narrower URL from page links.
    """
    links = list_markdown_links(doc.text[:12000], doc.source_ref)
    link_lines = "\n".join(
        f"- {lnk['label'][:80]} | {lnk['url']}" for lnk in links[:50]
    ) or "(no links parsed)"

    prompt = f"""
You are the Collector agent. The Verifier rejected the current page as TOO BROAD for the user query.
Your job: choose ONE narrower URL (subsection) to crawl next. Use the link list or add a fragment to the current page.

User query: {query}
Current URL: {doc.source_ref}
Page title: {doc.title}

Verifier feedback (follow this):
{report.collector_feedback}

Verifier reasons: {report.reasons}
Suggested follow_url from Verifier (if any): {report.follow_url or "none"}

Links found on this page:
{link_lines}

Return JSON:
{{
  "action": "crawl_url",
  "url": "https://...",
  "reason": "why this narrower scope answers the query"
}}
Rules:
- url must be https and more specific than crawling the whole index/chapter when possible
- prefer the single subsection that answers the query (e.g. one TOC entry), not the parent page
- if Verifier gave follow_url, you may use it if appropriate
"""
    return gemini.generate_json(prompt)


def processor_extract_section(
    gemini: GeminiService,
    doc: RawDocument,
    report: VerifierReport,
    decision: ApproverDecision,
) -> tuple[str, str, str]:
    """
    Processor: LLM picks line range or heading to index (minimal scope).
    Returns (text, source_ref, scope_note).
    """
    fragment = decision.target_fragment or report.target_fragment
    heading = decision.target_heading or report.target_heading
    follow = (decision.follow_url or report.follow_url or "").strip()
    source_ref = follow or doc.source_ref

    excerpt = doc.text[:14000]
    prompt = f"""
You are the Processor agent. Select the MINIMAL excerpt of this document to index for the user query.
Do NOT include unrelated sections (other chapters, full TOC, navigation boilerplate).

User query: {doc.query}
Source: {doc.source_ref}
Approved extract_mode: {decision.extract_mode}
Target heading (hint): {heading or "none"}
Target fragment (hint): {fragment or "none"}

Document (truncated):
{excerpt}

Return JSON:
{{
  "start_line": 1,
  "end_line": 120,
  "target_heading": "exact heading line text or empty",
  "target_fragment": "anchor slug without # or empty",
  "source_ref": "{source_ref}",
  "scope_note": "brief description of what was kept",
  "use_line_range": true
}}
use_line_range: true if start_line/end_line bound the answer; false if use target_heading only.
Lines are 1-based inclusive against the document above.
"""
    data = gemini.generate_json(prompt)

    use_lines = bool(data.get("use_line_range", True))
    start_line = int(data.get("start_line", 1))
    end_line = int(data.get("end_line", min(200, len(doc.text.splitlines()))))
    th = str(data.get("target_heading") or heading).strip()
    frag = str(data.get("target_fragment") or fragment).strip()
    ref = str(data.get("source_ref") or source_ref).strip()
    note = str(data.get("scope_note") or "llm_section")

    if use_lines and end_line > start_line:
        text = slice_by_line_range(doc.text, start_line, end_line)
        if len(text) >= 150:
            if frag and "#" not in ref:
                ref = f"{ref}#{frag}"
            return text, ref, note

    if th:
        text = slice_by_heading(doc.text, th)
        if len(text) >= 150:
            return text, ref, note

    if frag:
        text, ok = slice_by_fragment_hint(doc.text, frag)
        if ok:
            if "#" not in ref:
                ref = f"{ref}#{frag}"
            return text, ref, "fragment_hint"

    return doc.text[:8000], ref, "truncated_fallback"
