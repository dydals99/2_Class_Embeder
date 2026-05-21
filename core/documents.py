"""Load user Documents (PDF, Word, plain text)."""

from __future__ import annotations

from pathlib import Path

from config import Config, get_config
from core.crawler import make_doc_id
from core.schemas import RawDocument

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md"}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_file(path: Path, query: str) -> RawDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix in (".docx", ".doc"):
        text = _read_docx(path)
    else:
        text = _read_text(path)

    title = path.stem
    ref = str(path.resolve())
    return RawDocument(
        doc_id=make_doc_id(ref),
        source_type="upload",
        title=title,
        text=text.strip(),
        query=query,
        source_ref=ref,
        metadata={"filename": path.name, "suffix": suffix},
    )


def load_documents_dir(
    query: str,
    directory: Path | None = None,
    cfg: Config | None = None,
) -> list[RawDocument]:
    """Load all supported files from Documents folder."""
    config = cfg or get_config()
    base = directory or config.documents_dir
    if not base.exists():
        return []

    docs: list[RawDocument] = []
    for path in sorted(base.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                docs.append(load_file(path, query))
            except Exception as exc:  # noqa: BLE001 — collect per-file errors
                print(f"[documents] Skip {path.name}: {exc}")
    return docs
