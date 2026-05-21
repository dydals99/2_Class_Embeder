"""RAG answer generation (shared retrieval + Gemini)."""

from __future__ import annotations

from config import Config, get_config
from core.gemini import GeminiService
from core.vectorstore import VectorStore


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata") or {}
        source = meta.get("source_ref", "unknown")
        title = meta.get("title", "")
        blocks.append(f"[{i}] ({title}) {source}\n{hit.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(
    query: str,
    collection_name: str,
    *,
    cfg: Config | None = None,
    top_k: int | None = None,
) -> dict:
    config = cfg or get_config()
    gemini = GeminiService(config)
    store = VectorStore(collection_name, config, gemini)
    hits = store.query(query, top_k=top_k)
    context = build_context(hits)

    if not context.strip():
        return {
            "answer": "검색된 문맥이 없습니다. 먼저 agent_crawl.py 또는 legacy_crawl.py로 인덱스를 구축하세요.",
            "hits": [],
            "collection": collection_name,
        }

    prompt = f"""
You are a RAG assistant. Answer ONLY from the provided context.
If the context is insufficient, say so clearly in Korean.

User question: {query}

Context:
{context}

Write a concise, accurate answer in Korean. Cite sources by number [1], [2], etc.
"""
    answer = gemini.generate(prompt)
    return {
        "answer": answer,
        "hits": hits,
        "collection": collection_name,
    }
