"""Approver agent: final approve/reject + confirmed indexing scope."""

from __future__ import annotations

from config import Config, get_config
from core.gemini import GeminiService
from core.schemas import ApproverDecision, ExtractMode, RawDocument, VerifierReport


class ApproverAgent:
    def __init__(
        self,
        gemini: GeminiService | None = None,
        cfg: Config | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.gemini = gemini or GeminiService(self.cfg)

    def decide(self, doc: RawDocument, report: VerifierReport) -> ApproverDecision:
        if not report.passed or not report.ready_for_indexing:
            return ApproverDecision(
                approved=False,
                overall_score=report.overall_score,
                feedback="Verifier rejected or scope not ready. " + "; ".join(report.reasons),
                reasons=["verifier_failed", *report.reasons],
            )

        # Proposed pipeline: never approve full-page indexing for web docs when scope is known
        extract_mode: ExtractMode = report.extract_mode
        if doc.source_type == "web" and extract_mode == "full_page":
            if report.follow_url or report.target_fragment:
                extract_mode = "follow_url" if report.follow_url else "section_only"

        prompt = f"""
You are the Approver agent. Final gate before vector DB indexing.

Query: {doc.query}
Source: {doc.source_ref} ({doc.source_type})
Title: {doc.title}

Verifier scores:
- keyword_match: {report.keyword_match:.2f}
- source_traceability: {report.source_traceability:.2f}
- content_integrity: {report.content_integrity:.2f}
- semantic_similarity: {report.semantic_similarity:.2f}
- overall: {report.overall_score:.2f}

Verifier scope:
- ready_for_indexing: {report.ready_for_indexing}
- too_broad: {report.too_broad}
- extract_mode: {report.extract_mode}
- target_fragment: {report.target_fragment}
- target_heading: {report.target_heading}
- follow_url: {report.follow_url}

Rules:
- Approve only when Verifier marked ready_for_indexing (minimal scope).
- Prefer section_only over full_page.

Minimum overall for approval: {self.cfg.approver_min_overall}

Return JSON:
{{
  "approved": true,
  "overall_score": 0.0,
  "extract_mode": "{extract_mode}",
  "follow_url": "{report.follow_url}",
  "target_fragment": "{report.target_fragment}",
  "target_heading": "{report.target_heading}",
  "feedback": "",
  "reasons": []
}}
"""
        data = self.gemini.generate_json(prompt)
        approved = bool(data.get("approved", True))
        overall = float(data.get("overall_score", report.overall_score))
        feedback = str(data.get("feedback", ""))
        reasons = [str(r) for r in (data.get("reasons") or [])]

        mode = str(data.get("extract_mode") or extract_mode).strip()
        if mode not in ("section_only", "full_page", "follow_url", "index_line"):
            mode = extract_mode

        follow = str(data.get("follow_url") or report.follow_url).strip()
        fragment = str(data.get("target_fragment") or report.target_fragment).strip()
        heading = str(data.get("target_heading") or report.target_heading).strip()

        if overall < self.cfg.approver_min_overall:
            approved = False
            reasons.append(f"overall_score {overall:.2f} below {self.cfg.approver_min_overall}")

        if doc.source_type == "web" and mode == "full_page" and (follow or fragment):
            mode = "section_only"
            reasons.append("downgraded full_page to section_only for web source")

        if not approved and not feedback:
            feedback = "Document not approved for indexing. " + "; ".join(reasons)

        return ApproverDecision(
            approved=approved,
            overall_score=overall,
            feedback=feedback,
            reasons=reasons,
            extract_mode=mode,  # type: ignore[arg-type]
            follow_url=follow,
            target_fragment=fragment,
            target_heading=heading,
        )
