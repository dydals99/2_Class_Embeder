"""Crawl4AI wrapper: HTTP mode (default, no Playwright) or browser mode."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from config import Config, get_config


def _configure_console_utf8() -> None:
    """Avoid cp949 Unicode errors when Crawl4AI prints status on Windows."""
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONUTF8", "1")
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8")
                except Exception:
                    pass


def _normalize_urls(urls: list[str], max_urls: int) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for u in urls:
        u = u.strip()
        if u.startswith("http") and u not in seen:
            seen.add(u)
            unique.append(u)
    return unique[:max_urls]


def _result_from_crawl(url: str, result: Any) -> dict[str, Any]:
    success = getattr(result, "success", True)
    markdown = getattr(result, "markdown", None) or ""
    title = ""
    metadata = getattr(result, "metadata", None) or {}
    if isinstance(metadata, dict):
        title = str(metadata.get("title", "") or "")
    if not title:
        title = url
    text = markdown.strip()
    return {
        "success": bool(success) and len(text) > 50,
        "url": url,
        "title": title,
        "markdown": text,
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


class _SimpleHTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts))


async def _fetch_http_fallback(url: str, timeout_sec: int) -> dict[str, Any]:
    """Plain HTTP fetch when Crawl4AI is unavailable or returns empty body."""
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout_sec,
            headers=headers,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        print(f"[crawler] HTTP fallback failed for {url}: {exc}")
        return {
            "success": False,
            "url": url,
            "title": url,
            "markdown": "",
            "metadata": {"error": str(exc), "mode": "http_fallback"},
        }

    parser = _SimpleHTMLText()
    parser.feed(html)
    text = parser.get_text().strip()
    title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
    title = title_match.group(1).strip() if title_match else urlparse(url).netloc

    return {
        "success": len(text) > 50,
        "url": url,
        "title": title,
        "markdown": text,
        "metadata": {"mode": "http_fallback"},
    }


def _build_crawler(use_browser: bool):
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig, HTTPCrawlerConfig
    from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

    if use_browser:
        print("[crawler] Mode: browser (Playwright Chromium)")
        return AsyncWebCrawler(
            verbose=False,
            config=BrowserConfig(headless=True, verbose=False),
        )

    print("[crawler] Mode: http (no Playwright required)")
    strategy = AsyncHTTPCrawlerStrategy(browser_config=HTTPCrawlerConfig())
    return AsyncWebCrawler(crawler_strategy=strategy, verbose=False)


async def _crawl_all(urls: list[str], config: Config) -> list[dict[str, Any]]:
    _configure_console_utf8()
    use_browser = config.crawl_mode == "browser"
    results: list[dict[str, Any]] = []

    try:
        crawler = _build_crawler(use_browser)
        async with crawler:
            for url in urls:
                try:
                    crawl_result = await crawler.arun(url=url)
                    item = _result_from_crawl(url, crawl_result)
                except Exception as exc:
                    print(f"[crawler] Crawl4AI error for {url}: {exc}")
                    if use_browser and "Executable doesn't exist" in str(exc):
                        print(
                            "[crawler] Playwright browsers missing. Run in venv:\n"
                            "  python -m playwright install chromium\n"
                            "Or set CRAWL_MODE=http in .env"
                        )
                    item = {
                        "success": False,
                        "url": url,
                        "title": url,
                        "markdown": "",
                        "metadata": {"error": str(exc)},
                    }

                if not item["success"]:
                    fallback = await _fetch_http_fallback(url, config.crawl_timeout_sec)
                    if fallback["success"]:
                        print(f"[crawler] Used HTTP fallback for {url}")
                        item = fallback

                results.append(item)
    except Exception as exc:
        print(f"[crawler] Crawler startup failed ({exc}); using HTTP fallback for all URLs.")
        for url in urls:
            results.append(await _fetch_http_fallback(url, config.crawl_timeout_sec))

    return results


def crawl_urls(urls: list[str], cfg: Config | None = None) -> list[dict[str, Any]]:
    """Sync entry: crawl URLs with Crawl4AI (HTTP or browser mode)."""
    config = cfg or get_config()
    unique = _normalize_urls(urls, config.max_urls_per_run)
    if not unique:
        return []
    return asyncio.run(_crawl_all(unique, config))


def make_doc_id(source_ref: str) -> str:
    return hashlib.sha256(source_ref.encode("utf-8")).hexdigest()[:16]
