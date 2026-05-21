#!/usr/bin/env python3
"""
RAG query runner — same Gemini model, different Chroma collections.

Usage:
  python rag.py --query "RAG에서 청킹이 왜 중요한가?"
  python rag.py --query "..." --collection proposed
  python rag.py --query "..." --collection legacy
  python rag.py --query "..." --collection both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_config
from core.rag_engine import generate_answer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query RAG index (proposed / legacy / both)")
    p.add_argument("--query", "-q", required=True, help="Question for RAG")
    p.add_argument(
        "--collection",
        "-c",
        choices=("proposed", "legacy", "both"),
        default="both",
        help="Which index to query",
    )
    p.add_argument("--top-k", type=int, default=None, help="Override RAG_TOP_K")
    return p.parse_args()


def _print_result(label: str, result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}  [{result.get('collection')}]")
    print("=" * 60)
    print(result.get("answer", ""))
    hits = result.get("hits") or []
    if hits:
        print("\n--- Retrieved chunks ---")
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata") or {}
            print(f"  [{i}] dist={h.get('distance', 0):.4f} | {meta.get('title', '')} | {meta.get('source_ref', '')}")


def main() -> None:
    args = parse_args()
    cfg = get_config()
    print(f"[rag] Model: {cfg.gemini_model}")

    targets: list[tuple[str, str]] = []
    if args.collection in ("proposed", "both"):
        targets.append(("실험군 (Proposed)", cfg.collection_proposed))
    if args.collection in ("legacy", "both"):
        targets.append(("대조군 (Legacy)", cfg.collection_legacy))

    for label, name in targets:
        try:
            result = generate_answer(
                args.query,
                name,
                cfg=cfg,
                top_k=args.top_k,
            )
            _print_result(label, result)
        except Exception as exc:
            print(f"\n[{label}] Error: {exc}")


if __name__ == "__main__":
    main()
