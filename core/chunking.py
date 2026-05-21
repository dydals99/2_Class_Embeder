"""Fixed-size (legacy) and adaptive (proposed) chunking."""

from __future__ import annotations

import re

from config import Config, get_config
from core.schemas import ChunkRecord, RawDocument


def fixed_size_chunks(doc: RawDocument, cfg: Config | None = None) -> list[ChunkRecord]:
    """Legacy: approximate fixed token windows as character blocks."""
    config = cfg or get_config()
    size = config.legacy_chunk_chars
    text = doc.text
    chunks: list[ChunkRecord] = []
    if not text:
        return chunks

    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            break_at = text.rfind("\n", start, end)
            if break_at > start + size // 2:
                end = break_at + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}_{idx}",
                    doc_id=doc.doc_id,
                    text=piece,
                    metadata={
                        "source_type": doc.source_type,
                        "source_ref": doc.source_ref,
                        "title": doc.title,
                        "chunk_index": idx,
                        "chunking": "fixed",
                    },
                )
            )
            idx += 1
        start = end if end > start else start + size
    return chunks


def adaptive_chunks(
    doc: RawDocument,
    *,
    hint: str = "section",
    cfg: Config | None = None,
) -> list[ChunkRecord]:
    """Proposed: split by markdown structure and semantic boundaries."""
    config = cfg or get_config()
    text = doc.text
    if not text:
        return []

    if hint == "table_heavy" and "|" in text:
        sections = _split_tables(text)
    elif hint == "qa":
        sections = _split_qa(text)
    else:
        sections = _split_markdown_sections(text)

    merged = _merge_small_sections(
        sections,
        min_chars=config.adaptive_min_chunk_chars,
        max_chars=config.adaptive_max_chunk_chars,
    )

    chunks: list[ChunkRecord] = []
    for idx, piece in enumerate(merged):
        piece = piece.strip()
        if not piece:
            continue
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc.doc_id}_{idx}",
                doc_id=doc.doc_id,
                text=piece,
                metadata={
                    "source_type": doc.source_type,
                    "source_ref": doc.source_ref,
                    "title": doc.title,
                    "chunk_index": idx,
                    "chunking": "adaptive",
                    "chunking_hint": hint,
                },
            )
        )
    return chunks


def _split_markdown_sections(text: str) -> list[str]:
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []
    heading_re = re.compile(r"^#{1,6}\s+")

    for line in lines:
        if heading_re.match(line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    if len(sections) <= 1 and len(text) > 0:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p for p in paragraphs if p.strip()]
    return sections


def _split_tables(text: str) -> list[str]:
    blocks = re.split(r"\n(?=\|)", text)
    return [b for b in blocks if b.strip()] or [text]


def _split_qa(text: str) -> list[str]:
    pattern = re.compile(
        r"(?=(?:^|\n)(?:Q[:：]|Question[:：]|질문[:：]|A[:：]|Answer[:：]|답변[:：]))",
        re.IGNORECASE | re.MULTILINE,
    )
    parts = pattern.split(text)
    parts = [p for p in parts if p.strip()]
    return parts if len(parts) > 1 else _split_markdown_sections(text)


def _merge_small_sections(
    sections: list[str],
    *,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) > max_chars:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.extend(_hard_split(sec, max_chars))
            continue
        candidate = f"{buffer}\n\n{sec}".strip() if buffer else sec
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = sec
    if buffer:
        merged.append(buffer)

    result: list[str] = []
    for block in merged:
        if len(block) < min_chars and result:
            result[-1] = f"{result[-1]}\n\n{block}"
        else:
            result.append(block)
    return result


def _hard_split(text: str, max_chars: int) -> list[str]:
    out = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        out.append(text[start:end])
        start = end
    return out


def export_chunks_csv(chunks: list[ChunkRecord], path: str) -> None:
    import csv
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["chunk_id", "doc_id", "title", "source_ref", "chunking", "text"]
        )
        for c in chunks:
            writer.writerow(
                [
                    c.chunk_id,
                    c.doc_id,
                    c.metadata.get("title", ""),
                    c.metadata.get("source_ref", ""),
                    c.metadata.get("chunking", ""),
                    c.text[:5000],
                ]
            )
