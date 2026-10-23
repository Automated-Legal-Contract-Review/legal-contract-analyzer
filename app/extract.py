"""Text extraction for PDF, DOCX and TXT contracts."""
from __future__ import annotations

import codecs
import os
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}
VIRTUAL_PAGE_CHARS = 3000
MIN_PAGE_TEXT = 25  # characters of real text for a PDF page to count as "has text"


class ExtractionError(ValueError):
    """Raised with a message that is safe and helpful to show to the user."""


@dataclass
class Extracted:
    pages: list[str]
    page_kind: str  # "exact" = real page numbers, "approx" = estimated pages (DOCX/TXT)
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- helpers
def check_file_signature(path: Path, suffix: str) -> None:
    """Make sure the bytes match the extension so a renamed file fails with a clear message."""
    with path.open("rb") as handle:
        head = handle.read(2048)
    if not head:
        raise ExtractionError("The uploaded file is empty.")
    if suffix == ".pdf" and b"%PDF" not in head:
        raise ExtractionError("This file has a .pdf extension but is not a valid PDF.")
    if suffix == ".docx":
        if not zipfile.is_zipfile(path):
            raise ExtractionError(
                "This file has a .docx extension but is not a valid Word document. "
                "Old .doc files must be re-saved as .docx first."
            )
        with zipfile.ZipFile(path) as archive:
            if "word/document.xml" not in archive.namelist():
                raise ExtractionError("This .docx file does not contain a Word document body.")
    if suffix == ".txt" and head.count(b"\x00") > 20 and not head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        raise ExtractionError("This file does not look like plain text.")


def virtual_pages(text: str, size: int = VIRTUAL_PAGE_CHARS) -> list[str]:
    """Split text into roughly page-sized pieces at line boundaries (used when real pages are unknown)."""
    lines = text.split("\n")
    pages: list[str] = []
    current: list[str] = []
    length = 0
    for line in lines:
        if length + len(line) + 1 > size and current:
            pages.append("\n".join(current))
            current, length = [], 0
        # A single enormous line is hard-split so one page never explodes.
        while len(line) > size * 2:
            pages.append(line[:size])
            line = line[size:]
        current.append(line)
        length += len(line) + 1
    if current:
        pages.append("\n".join(current))
    return [p for p in pages if p.strip()] or [text]


def decode_text_bytes(raw: bytes) -> str:
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def clean_text(value: str) -> str:
    value = value.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t\u00a0]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


# ----------------------------------------------------------------------------- OCR (optional)
def _find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def ocr_available() -> bool:
    return _find_tesseract() is not None


def _prepare_tesseract_env() -> None:
    exe = _find_tesseract()
    if not exe:
        return
    folder = str(Path(exe).parent)
    if folder not in os.environ.get("PATH", ""):
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    if not os.environ.get("TESSDATA_PREFIX"):
        for candidate in (Path(folder) / "tessdata", Path("/usr/share/tesseract-ocr/5/tessdata"), Path("/usr/share/tesseract-ocr/4.00/tessdata"), Path("/usr/share/tessdata")):
            if candidate.exists():
                os.environ["TESSDATA_PREFIX"] = str(candidate)
                break


# ----------------------------------------------------------------------------- per-format readers
def _extract_pdf(path: Path) -> Extracted:
    import pymupdf  # imported lazily so tests of other code do not need it

    notes: list[str] = []
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:  # corrupt files raise several different exception types
        raise ExtractionError(f"The PDF could not be opened ({exc}).") from exc
    with pdf:
        if pdf.needs_pass:
            raise ExtractionError("This PDF is password-protected. Remove the password and upload it again.")
        if pdf.page_count == 0:
            raise ExtractionError("This PDF has no pages.")
        pages: list[str] = []
        failed_pages = 0
        for page in pdf:
            try:
                pages.append(page.get_text("text") or "")
            except Exception:
                pages.append("")
                failed_pages += 1

        empty = [i for i, text in enumerate(pages) if len(re.sub(r"\s+", "", text)) < MIN_PAGE_TEXT]
        if empty and ocr_available():
            _prepare_tesseract_env()
            recovered = 0
            for i in empty:
                try:
                    page = pdf[i]
                    textpage = page.get_textpage_ocr(language="eng", dpi=200, full=True)
                    text = page.get_text("text", textpage=textpage) or ""
                    if len(re.sub(r"\s+", "", text)) >= MIN_PAGE_TEXT:
                        pages[i] = text
                        recovered += 1
                except Exception:
                    # One page can contain an unsupported image or malformed OCR layer;
                    # keep trying later pages rather than silently abandoning the rest.
                    continue
            if recovered:
                notes.append(f"OCR was used to read {recovered} scanned page(s). OCR can contain recognition errors.")
            empty = [i for i, text in enumerate(pages) if len(re.sub(r"\s+", "", text)) < MIN_PAGE_TEXT]

        if failed_pages:
            notes.append(f"{failed_pages} page(s) could not be read.")
        if empty:
            if len(empty) == len(pages):
                raise ExtractionError(
                    "No readable text was found. This looks like a scanned (image-only) PDF. "
                    "Install Tesseract OCR (see README) or run OCR on the file first, then upload it again."
                )
            shown = ", ".join(str(i + 1) for i in empty[:15]) + ("..." if len(empty) > 15 else "")
            notes.append(f"{len(empty)} page(s) contain no extractable text (likely images or scans): pages {shown}.")
    return Extracted(pages=[clean_text(p) for p in pages], page_kind="exact", notes=notes)


def _iter_docx_blocks(doc):
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(child, doc)
        elif tag == "tbl":
            yield Table(child, doc)


def _extract_docx(path: Path) -> Extracted:
    from docx import Document
    from docx.table import Table

    try:
        doc = Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"The Word document could not be opened ({exc}).") from exc
    lines: list[str] = []
    for block in _iter_docx_blocks(doc):
        if isinstance(block, Table):
            for row in block.rows:
                cells: list[str] = []
                seen_cells: set[int] = set()
                for cell in row.cells:
                    if id(cell._tc) in seen_cells:  # merged cells repeat; keep one copy
                        continue
                    seen_cells.add(id(cell._tc))
                    text = " ".join(cell.text.split())
                    if text:
                        cells.append(text)
                if cells:
                    lines.append(" | ".join(cells))
        else:
            text = block.text.strip()
            if text:
                lines.append(text)
    text = clean_text("\n".join(lines))
    if not text:
        raise ExtractionError("No text was found in this Word document.")
    return Extracted(
        pages=virtual_pages(text),
        page_kind="approx",
        notes=["Word files have no fixed page numbers, so page references are estimates."],
    )


def _extract_txt(path: Path) -> Extracted:
    text = clean_text(decode_text_bytes(path.read_bytes()))
    if not text:
        raise ExtractionError("The text file is empty.")
    if "\f" in text:  # form feeds are real page breaks
        pages = [clean_text(p) for p in text.split("\f")]
        return Extracted(pages=[p for p in pages if p] or [text], page_kind="exact")
    return Extracted(
        pages=virtual_pages(text),
        page_kind="approx",
        notes=["Plain text files have no page numbers, so page references are estimates."],
    )


def extract(path: Path) -> Extracted:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ExtractionError("Unsupported file type. Upload a PDF, DOCX, or TXT file.")
    check_file_signature(path, suffix)
    if suffix == ".pdf":
        result = _extract_pdf(path)
    elif suffix == ".docx":
        result = _extract_docx(path)
    else:
        result = _extract_txt(path)
    total = sum(len(re.sub(r"\s+", "", p)) for p in result.pages)
    if total < 80:
        raise ExtractionError("Too little readable text was found in this file to review.")
    return result
