#!/usr/bin/env python3
"""
Experimental pipeline (proposed):
  Route A — Collector (broad crawl) → Verifier → [feedback loop] Collector (narrow) → Approver → Processor → Chroma
  Verifier may reject as too_broad; Collector re-crawls narrower URL per LLM feedback (no keyword rules).
  Route B — User Documents/ → same verify/approve/process chain

Usage:
  python agent_crawl.py --query "RAG 청킹 전략"
  python agent_crawl.py --query "..." --urls https://example.com/a https://example.com/b
  python agent_crawl.py --query "..." --uploads-only
  python agent_crawl.py --query "..." --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_config
from core.agents.orchestrator import AgentPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experimental agent pipeline: verify → approve → adaptive chunk → index"
    )
    p.add_argument("--query", "-q", required=True, help="User research query")
    p.add_argument(
        "--urls",
        nargs="*",
        default=[],
        help="Seed URLs for collector (optional)",
    )
    p.add_argument(
        "--documents-dir",
        type=Path,
        default=None,
        help="Folder with PDF/DOCX/TXT (default: ./Documents)",
    )
    p.add_argument(
        "--uploads-only",
        action="store_true",
        help="Skip collector; only process files in Documents/",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Clear proposed Chroma collection before indexing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    print(f"[agent_crawl] Model: {cfg.gemini_model}")
    print(f"[agent_crawl] Collection: {cfg.collection_proposed}")
    print(f"[agent_crawl] Crawl mode: {cfg.crawl_mode}")

    pipeline = AgentPipeline(cfg)
    stats = pipeline.run(
        args.query,
        urls=args.urls,
        documents_dir=args.documents_dir,
        use_uploads_only=args.uploads_only,
        reset_index=args.reset,
    )

    print("\n=== Agent pipeline summary ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
