"""Document model, boundary-aware chunking and exact quote-to-page mapping."""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Any

PAGE_SEPARATOR = "\n\n"


def squash(value: str) -> str:
    """Letters and digits only, lower-cased. Makes quote matching immune to spacing, line breaks,
    hyphenation artefacts, curly quotes and dash variants."""
    return re.sub(r"[\W_]+", "", (value or "").lower())


@dataclass
class Document:
    full_text: str
    page_starts: list[int]  # offset in full_text where each page begins (index 0 = page 1)

    @property
    def page_count(self) -> int:
        return len(self.page_starts)

    def page_at(self, offset: int) -> int:
        """1-based page number containing a character offset."""
        offset = max(0, min(offset, max(0, len(self.full_text) - 1)))
        return max(1, bisect.bisect_right(self.page_starts, offset))


def build_document(pages: list[str]) -> Document:
    parts: list[str] = []
    starts: list[int] = []
    cursor = 0
    for index, page in enumerate(pages):
        text = (page or "").strip()
        starts.append(cursor)
        parts.append(text)
        cursor += len(text) + (len(PAGE_SEPARATOR) if index < len(pages) - 1 else 0)
    return Document(full_text=PAGE_SEPARATOR.join(parts), page_starts=starts)


def _best_break(text: str, low: int, high: int) -> int:
    """Pick a natural split point in text[low:high]: paragraph > line > sentence > space."""
    window = text[low:high]
    for token in ("\n\n", "\n", ". ", "; ", " "):
        position = window.rfind(token)
        if position != -1:
            return low + position + len(token)
    return high


def make_chunks(doc: Document, chunk_chars: int, overlap: int) -> list[dict[str, Any]]:
    text = doc.full_text
    total = len(text)
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < total:
        end = min(total, start + chunk_chars)
        if end < total:
            end = _best_break(text, start + int(chunk_chars * 0.6), end)
        end = max(end, start + 1)
        chunks.append({
            "index": len(chunks) + 1,
            "start": start,
            "end": end,
            "text": text[start:end],
            "page_start": doc.page_at(start),
            "page_end": doc.page_at(end - 1),
        })
        if end >= total:
            break
        next_start = max(end - overlap, start + 1)
        # Start the next chunk on a word boundary instead of in the middle of a word.
        boundary = text.find(" ", next_start, end)
        if boundary != -1 and boundary - next_start < 200:
            next_start = boundary + 1
        start = next_start
    return chunks


class Matcher:
    """Finds a quoted passage inside a text and returns where it starts in the ORIGINAL text."""

    def __init__(self, text: str):
        self.text = text
        chars: list[str] = []
        index_map: list[int] = []
        for position, char in enumerate(text):
            lowered = char.lower()
            if re.match(r"[\w]", lowered) and lowered != "_":
                chars.append(lowered)
                index_map.append(position)
        self.squashed = "".join(chars)
        self.index_map = index_map

    def find(self, quote: str) -> int | None:
        needle = squash(quote)
        if len(needle) < 15:
            return None
        found = self.squashed.find(needle)
        if found == -1:
            return None
        return self.index_map[found]

    def find_loose(self, quote: str) -> tuple[int, int] | None:
        """Like find, but also accepts quotes shortened with an ellipsis ('start ... end').
        Returns (start_offset, end_offset) in the original text."""
        direct = self.find(quote)
        if direct is not None:
            needle_len = len(squash(quote))
            end_squashed = self.squashed.find(squash(quote)) + needle_len - 1
            return direct, self.index_map[end_squashed]
        parts = [p for p in re.split(r"\.{3,}|\u2026", quote) if len(squash(p)) >= 15]
        if not parts:
            return None
        first = self.find(parts[0])
        if first is None:
            return None
        end_offset = first
        for part in parts[1:]:
            position = self.find(part)
            if position is None:
                return None
            end_offset = max(end_offset, position)
        return first, end_offset


def split_passages(doc: Document, size: int, overlap: int) -> list[dict[str, Any]]:
    """Smaller pieces used for retrieval in the 'Ask the contract' feature."""
    return [
        {
            "i": chunk["index"],
            "text": chunk["text"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
        }
        for chunk in make_chunks(doc, size, overlap)
    ]
