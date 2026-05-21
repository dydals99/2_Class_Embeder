"""Agentic pipeline with Verifier ↔ Collector feedback loop (LLM-driven scope)."""

from __future__ import annotations

from pathlib import Path

from config import Config, get_config
from core.agents.approver import ApproverAgent
from core.agents.collector import CollectorAgent
from core.agents.processor import ProcessorAgent
from core.agents.verifier import VerifierAgent
from core.chunking import export_chunks_csv
from core.documents import load_documents_dir
from core.gemini import GeminiService
from core.schemas import ChunkRecord, RawDocument, VerifierReport
from core.vectorstore import VectorStore


class AgentPipeline:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or get_config()
        self.gemini = GeminiService(self.cfg)
        self.collector = CollectorAgent(self.gemini, self.cfg)
        self.verifier = VerifierAgent(self.gemini, self.cfg)
        self.approver = ApproverAgent(self.gemini, self.cfg)
        self.processor = ProcessorAgent(self.cfg, self.gemini)
        self.store = VectorStore(
            self.cfg.collection_proposed,
            self.cfg,
            self.gemini,
        )

    def run(
        self,
        query: str,
        *,
        urls: list[str] | None = None,
        documents_dir: Path | None = None,
        use_uploads_only: bool = False,
        reset_index: bool = False,
    ) -> dict:
        if reset_index:
            self.store.reset_collection()

        all_chunks: list[ChunkRecord] = []
        stats = {
            "ingested": 0,
            "verified_pass": 0,
            "approved": 0,
            "rejected": 0,
            "collector_retries": 0,
            "chunks_indexed": 0,
            "rejections": [],
        }

        raw_docs: list[RawDocument] = []

        if not use_uploads_only:
            target_urls = self.collector.resolve_urls(query, urls or [], "")
            print(f"[pipeline] Initial collect: {len(target_urls)} URL(s)")
            raw_docs = self.collector.collect_from_urls(query, target_urls)

        upload_docs = load_documents_dir(query, documents_dir, self.cfg)
        raw_docs.extend(upload_docs)

        if not raw_docs:
            print("[pipeline] No documents to process.")
            return stats

        stats["ingested"] = len(raw_docs)

        for initial in list(raw_docs):
            doc, report = self._verify_with_collector_loop(initial, stats)
            if doc is None or report is None or not report.passed:
                continue

            stats["verified_pass"] += 1
            decision = self.approver.decide(doc, report)
            print(
                f"[approver] {doc.source_ref[:55]}... "
                f"approved={decision.approved}"
            )

            if not decision.approved:
                stats["rejected"] += 1
                stats["rejections"].append(
                    {
                        "source": doc.source_ref,
                        "stage": "approver",
                        "feedback": decision.feedback,
                        "reasons": decision.reasons,
                    }
                )
                continue

            stats["approved"] += 1
            all_chunks.extend(self.processor.process(doc, report, decision))

        if all_chunks:
            count = self.store.add_chunks(all_chunks)
            stats["chunks_indexed"] = count
            export_path = self.cfg.exports_dir / "proposed_chunks.csv"
            export_chunks_csv(all_chunks, str(export_path))
            print(f"[pipeline] Exported CSV: {export_path}")

        return stats

    def _verify_with_collector_loop(
        self,
        doc: RawDocument,
        stats: dict,
    ) -> tuple[RawDocument | None, VerifierReport | None]:
        """
        Verify → if too broad, feedback to Collector → re-crawl → verify again.
        """
        report: VerifierReport | None = None

        for attempt in range(self.cfg.max_collect_retries + 1):
            report = self.verifier.verify(doc)
            print(
                f"[verifier] attempt {attempt + 1} | {doc.source_ref[:50]}... "
                f"passed={report.passed} too_broad={report.too_broad}"
            )

            if report.passed:
                return doc, report

            if report.too_broad and report.collector_feedback:
                if attempt >= self.cfg.max_collect_retries:
                    break
                stats["collector_retries"] = stats.get("collector_retries", 0) + 1
                narrowed = self.collector.apply_verifier_feedback(
                    doc.query, doc, report
                )
                if narrowed is None:
                    print("[pipeline] Collector could not narrow scope.")
                    break
                if narrowed.source_ref == doc.source_ref and narrowed.text == doc.text:
                    print("[pipeline] No narrower document obtained; stop retry.")
                    break
                doc = narrowed
                print(f"[pipeline] Re-crawled narrower doc: {doc.source_ref[:70]}")
                continue

            stats["rejected"] += 1
            stats["rejections"].append(
                {
                    "source": doc.source_ref,
                    "stage": "verifier",
                    "reasons": report.reasons,
                    "collector_feedback": report.collector_feedback,
                }
            )
            if doc.source_type == "web":
                print("[pipeline] Verifier rejected — no further collector retry.")
            return None, None

        if report and not report.passed:
            stats["rejected"] += 1
            stats["rejections"].append(
                {
                    "source": doc.source_ref,
                    "stage": "verifier",
                    "reasons": report.reasons,
                    "collector_feedback": report.collector_feedback,
                }
            )
        return None, None
