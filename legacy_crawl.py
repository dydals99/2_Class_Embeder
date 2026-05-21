#!/usr/bin/env python3
"""
Control pipeline (legacy / baseline):
  Crawl4AI collect → NO verification → fixed-size chunking → Chroma

Usage:
  python legacy_crawl.py --query "RAG 청킹 전략"
  python legacy_crawl.py --query "..." --urls https://example.com/page
  python legacy_crawl.py --query "..." --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_config
from core.agents.collector import CollectorAgent
from core.chunking import export_chunks_csv, fixed_size_chunks
from core.gemini import GeminiService
from core.schemas import RawDocument
from core.vectorstore import VectorStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Legacy pipeline: crawl → fixed chunk → index (no verifier/approver)"
    )
    p.add_argument("--query", "-q", required=True, help="User research query")
    p.add_argument("--urls", nargs="*", default=[], help="Seed URLs to crawl")
    p.add_argument(
        "--reset",
        action="store_true",
        help="Clear legacy Chroma collection before indexing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    print(f"[legacy_crawl] Model: {cfg.gemini_model}")
    print(f"[legacy_crawl] Collection: {cfg.collection_legacy}")
    print(f"[legacy_crawl] Crawl mode: {cfg.crawl_mode}")

    gemini = GeminiService(cfg)
    collector = CollectorAgent(gemini, cfg)
    store = VectorStore(cfg.collection_legacy, cfg, gemini)

    if args.reset:
        store.reset_collection()

    urls = collector.resolve_urls(args.query, args.urls)
    if not urls:
        print(
            "[legacy_crawl] No URLs to crawl. Pass --urls https://... "
            "or retry later when Gemini is available for URL suggestion."
        )
        return
    print(f"[legacy_crawl] Crawling {len(urls)} URL(s)...")
    docs = collector.collect_from_urls(args.query, urls)

    if not docs:
        print("[legacy_crawl] No documents crawled. Exiting.")
        return

    all_chunks = []
    for doc in docs:
        chunks = fixed_size_chunks(doc, cfg)
        all_chunks.extend(chunks)
        print(f"[legacy_crawl] Fixed chunks: {len(chunks)} — {doc.title}")

    if not all_chunks:
        print("[legacy_crawl] No chunks produced.")
        return

    n = store.add_chunks(all_chunks)
    export_path = cfg.exports_dir / "legacy_chunks.csv"
    export_chunks_csv(all_chunks, str(export_path))

    print("\n=== Legacy pipeline summary ===")
    print(f"  documents: {len(docs)}")
    print(f"  chunks_indexed: {n}")
    print(f"  csv_export: {export_path}")


if __name__ == "__main__":
    main()
