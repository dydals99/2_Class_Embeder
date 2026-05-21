"""Focus document text on URL hash fragment (slug-derived only)."""

from __future__ import annotations

from core.section_slice import slice_by_fragment_hint


def extract_focused_text(
    text: str,
    url: str,
    *,
    max_chars: int = 8000,
) -> tuple[str, bool]:
    """Use #fragment from URL; no query-specific keyword lists."""
    from urllib.parse import urlparse

    if not text.strip():
        return "", False

    frag = urlparse(url).fragment.strip()
    if not frag:
        return text[:max_chars], False

    focused, ok = slice_by_fragment_hint(text, frag)
    if ok:
        return focused[:max_chars], True
    return text[:max_chars], False
