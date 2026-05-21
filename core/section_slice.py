"""Structural text utilities only (no query-keyword hardcoding)."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse


def list_markdown_links(text: str, base_url: str, *, limit: int = 60) -> list[dict[str, str]]:
    """Extract [label](url) pairs for LLM to choose from."""
    pattern = re.compile(r"\[([^\]]*)\]\((https?://[^)\s#]+(?:#[^)\s]+)?)\)")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, href in pattern.findall(text):
        href = href.strip()
        if not href.startswith("http"):
            href = urljoin(base_url, href)
        if href in seen:
            continue
        seen.add(href)
        out.append({"label": label.strip(), "url": href})
        if len(out) >= limit:
            break
    return out


def slice_by_line_range(text: str, start_line: int, end_line: int) -> str:
    """1-based inclusive line range from LLM."""
    lines = text.splitlines()
    if not lines:
        return ""
    s = max(1, start_line) - 1
    e = min(len(lines), max(s + 1, end_line))
    return "\n".join(lines[s:e]).strip()


def slice_by_heading(text: str, target_heading: str) -> str:
    """Slice from first line containing heading text until next peer markdown heading."""
    target = target_heading.strip().lower()
    if not target:
        return ""
    lines = text.splitlines()
    start = -1
    for i, line in enumerate(lines):
        if target in line.lower():
            start = max(0, i)
            break
    if start < 0:
        return ""

    heading_re = re.compile(r"^(#{1,6})\s+")
    start_level = None
    m0 = heading_re.match(lines[start])
    if m0:
        start_level = len(m0.group(1))
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = heading_re.match(lines[j])
        if m and start_level is not None and len(m.group(1)) <= start_level:
            end = j
            break
    return "\n".join(lines[start:end]).strip()


def slice_by_fragment_hint(text: str, fragment: str) -> tuple[str, bool]:
    """
    Match URL fragment as slug variants only (derived from fragment string, not query).
    """
    frag = fragment.strip().lower()
    if not frag:
        return "", False

    needles = {
        frag,
        frag.replace("-", " "),
        frag.replace("-", "_"),
    }
    parts = re.split(r"[-_]+", frag)
    needles.update(p for p in parts if len(p) > 2)

    lines = text.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        low = line.lower()
        if not any(n in low for n in needles):
            continue
        stripped = line.strip()
        if len(stripped) < 100 and ("](" in stripped or stripped.startswith("- [")):
            continue
        start_idx = max(0, i - 1)
        break

    if start_idx < 0:
        return "", False

    section_lines = lines[start_idx:]
    heading_re = re.compile(r"^(#{1,6})\s+")
    start_level = None
    end_idx = len(section_lines)
    for j, line in enumerate(section_lines):
        m = heading_re.match(line)
        if j == 0 and m:
            start_level = len(m.group(1))
        elif j > 0 and m and start_level is not None and len(m.group(1)) <= start_level:
            end_idx = j
            break

    sliced = "\n".join(section_lines[:end_idx]).strip()
    return (sliced, len(sliced) >= 200)
