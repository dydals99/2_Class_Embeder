"""Collector agent: crawl + narrow scope from Verifier feedback (LLM)."""

from __future__ import annotations

from config import Config, get_config
from core.crawler import crawl_urls, make_doc_id
from core.gemini import GeminiService
from core.schemas import RawDocument, VerifierReport
from core.scope_llm import collector_narrow_from_feedback


class CollectorAgent:
    def __init__(
        self,
        gemini: GeminiService | None = None,
        cfg: Config | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.gemini = gemini or GeminiService(self.cfg)

    def resolve_urls(
        self,
        query: str,
        seed_urls: list[str],
        feedback: str = "",
    ) -> list[str]:
        """
        Initial URLs: user seeds, or LLM suggestion when empty.
        feedback: legacy string or Verifier collector_feedback for re-collect.
        """
        seeds = [u.strip() for u in seed_urls if str(u).strip().startswith("http")]
        if seeds and not feedback:
            print(f"[collector] Using {len(seeds)} user seed URL(s).")
            return seeds[: self.cfg.max_urls_per_run]

        if feedback:
            return self._urls_from_feedback_text(query, seeds, feedback)

        return self.suggest_urls(query, [], "")

    def apply_verifier_feedback(
        self,
        query: str,
        doc: RawDocument,
        report: VerifierReport,
    ) -> RawDocument | None:
        """
        Verifier rejected (too broad) → LLM picks narrower URL → re-crawl.
        """
        if not report.collector_feedback and not report.too_broad:
            return None

        print("[collector] Received Verifier feedback — narrowing crawl scope...")
        print(f"[collector] Feedback: {report.collector_feedback[:300]}")

        if report.follow_url.startswith("http"):
            data = {"action": "crawl_url", "url": report.follow_url, "reason": "verifier follow_url"}
        else:
            data = collector_narrow_from_feedback(self.gemini, query, doc, report)

        url = str(data.get("url") or "").strip()
        if not url.startswith("http"):
            print("[collector] Could not resolve narrower URL from feedback.")
            return None

        print(f"[collector] Narrower URL: {url}")
        docs = self.collect_from_urls(query, [url])
        return docs[0] if docs else None

    def suggest_urls(
        self,
        query: str,
        seed_urls: list[str],
        feedback: str = "",
    ) -> list[str]:
        prompt = f"""
You are the Collector agent. Suggest URLs to crawl for the user query.

Query: {query}
Seed URLs: {seed_urls}
Prior feedback: {feedback or "none"}

Return JSON:
{{"urls": ["https://...", ...]}}
Up to {self.cfg.max_urls_per_run} real https URLs. No placeholders.
"""
        data = self.gemini.generate_json(prompt)
        suggested = [
            str(u).strip() for u in (data.get("urls") or []) if str(u).startswith("http")
        ]
        merged: list[str] = []
        seen: set[str] = set()
        for u in list(seed_urls) + suggested:
            if u.startswith("http") and u not in seen:
                seen.add(u)
                merged.append(u)
        return merged[: self.cfg.max_urls_per_run]

    def _urls_from_feedback_text(
        self,
        query: str,
        seeds: list[str],
        feedback: str,
    ) -> list[str]:
        prompt = f"""
Collector: choose URL(s) to crawl after Verifier feedback.

Query: {query}
Seed URLs: {seeds}
Verifier/Collector feedback:
{feedback}

Return JSON: {{"urls": ["https://..."]}}
Pick the most specific URL(s) implied by the feedback (subsection, not whole site).
"""
        data = self.gemini.generate_json(prompt)
        urls = [str(u).strip() for u in (data.get("urls") or []) if str(u).startswith("http")]
        return urls[: self.cfg.max_urls_per_run] or seeds[:1]

    def collect_from_urls(
        self,
        query: str,
        urls: list[str],
    ) -> list[RawDocument]:
        results = crawl_urls(urls, self.cfg)
        docs: list[RawDocument] = []
        for item in results:
            if not item.get("success"):
                print(f"[collector] Crawl failed or empty: {item.get('url')}")
                continue
            url = item["url"]
            docs.append(
                RawDocument(
                    doc_id=make_doc_id(url),
                    source_type="web",
                    title=item.get("title") or url,
                    text=item.get("markdown") or "",
                    query=query,
                    source_ref=url,
                    metadata={"crawl_metadata": item.get("metadata", {})},
                )
            )
        return docs
