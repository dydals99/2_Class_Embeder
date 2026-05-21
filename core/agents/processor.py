"""Processor: LLM-minimal section extract → adaptive chunk."""

from __future__ import annotations

from config import Config, get_config
from core.chunking import adaptive_chunks
from core.crawler import make_doc_id
from core.gemini import GeminiService
from core.schemas import ApproverDecision, ChunkRecord, RawDocument, VerifierReport
from core.scope_llm import processor_extract_section


class ProcessorAgent:
    def __init__(
        self,
        cfg: Config | None = None,
        gemini: GeminiService | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.gemini = gemini or GeminiService(self.cfg)

    def process(
        self,
        doc: RawDocument,
        report: VerifierReport,
        decision: ApproverDecision,
    ) -> list[ChunkRecord]:
        text, source_ref, scope_note = processor_extract_section(
            self.gemini, doc, report, decision
        )
        sliced_doc = RawDocument(
            doc_id=make_doc_id(source_ref),
            source_type=doc.source_type,
            title=doc.title,
            text=text,
            query=doc.query,
            source_ref=source_ref,
            metadata={
                **doc.metadata,
                "scope_note": scope_note,
                "extract_mode": decision.extract_mode,
                "original_source": doc.source_ref,
            },
        )

        chunks = adaptive_chunks(
            sliced_doc,
            hint=report.chunking_hint,
            cfg=self.cfg,
        )
        max_n = self.cfg.proposed_max_chunks_per_doc
        if len(chunks) > max_n:
            chunks = self._merge_to_limit(chunks, max_n, sliced_doc)

        for ch in chunks:
            ch.metadata["chunking"] = "adaptive_scoped"
            ch.metadata["scope_note"] = scope_note

        print(
            f"[processor] Indexed {len(chunks)} chunk(s), {len(text)} chars — {scope_note}"
        )
        return chunks

    def _merge_to_limit(
        self,
        chunks: list[ChunkRecord],
        max_n: int,
        doc: RawDocument,
    ) -> list[ChunkRecord]:
        if max_n <= 1:
            combined = "\n\n".join(c.text for c in chunks)
            return [
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}_0",
                    doc_id=doc.doc_id,
                    text=combined,
                    metadata={**chunks[0].metadata, "chunk_index": 0, "merged": True},
                )
            ]
        group_size = (len(chunks) + max_n - 1) // max_n
        merged: list[ChunkRecord] = []
        for i in range(0, len(chunks), group_size):
            group = chunks[i : i + group_size]
            combined = "\n\n".join(c.text for c in group)
            meta = {**group[0].metadata, "chunk_index": len(merged), "merged": True}
            merged.append(
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}_{len(merged)}",
                    doc_id=doc.doc_id,
                    text=combined,
                    metadata=meta,
                )
            )
        return merged[:max_n]
