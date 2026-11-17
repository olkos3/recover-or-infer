"""Pages, tokens and the joined document text that all offsets refer to."""
import re
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.get_encoding("cl100k_base")


def tokens(text):
    return _encoding().encode(text, disallowed_special=())


def n_tokens(text):
    return len(tokens(text))


def paginate(text, page_chars=3000):
    """Cut plain text into pseudo-pages of about 3,000 characters at line breaks (as done for LegalBench-RAG).
    Form feeds (\\f) are honoured as hard page breaks and kept at the end of their page, so ''.join(pages) == text."""
    pages = []
    for block in re.split(r"(?<=\f)", text):
        start = 0
        while start < len(block):
            end = min(len(block), start + page_chars)
            if end < len(block):
                nl = block.rfind("\n", start + page_chars // 2, end + page_chars // 2)
                end = nl + 1 if nl > 0 else end
            pages.append({"page": len(pages) + 1, "text": block[start:end]})
            start = end
    return pages


def load_pages(path):
    """A .txt file becomes pseudo-pages; a .pdf keeps its physical pages (text via pypdf)."""
    if str(path).lower().endswith(".pdf"):
        from pypdf import PdfReader
        return [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(PdfReader(path).pages)]
    with open(path, encoding="utf-8") as f:
        return paginate(f.read().replace("\r\n", "\n"))


def document_text(pages):
    """Join pages with a newline. Returns the text and each page's (start, end) offsets in it."""
    text, spans = "", []
    for p in pages:
        start = len(text)
        text += p["text"]
        spans.append({"page": p["page"], "start": start, "end": len(text)})
        text += "\n"
    return text, spans


def windows(text, limit):
    """Lossless, non-overlapping character spans of at most `limit` tokens, preferring to end at a space."""
    pos = 0
    while pos < len(text):
        lo, hi = pos + 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if n_tokens(text[pos:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        end = lo
        if end < len(text):
            space = text.rfind(" ", pos, end)
            if space > pos + (end - pos) // 2:
                end = space + 1
        yield pos, end
        pos = end


def prose(text, min_run=4, short_max_words=4):
    """Drop runs of table-like or very short lines (a conservative line heuristic, not a layout parser)."""
    lines = text.splitlines()
    flags = []
    for line in lines:
        digits = sum(c.isdigit() for c in line)
        cells = [c for c in re.split(r"\t|\s{2,}|\|", line.strip()) if c]
        table = (len(cells) >= 3 and digits > 0) or (digits >= 6 and digits / max(1, len(line.replace(" ", ""))) > .45)
        short = bool(line.strip()) and len(line.strip().split()) <= short_max_words and len(line.strip()) <= 64
        flags.append(table or short)
    remove, run = set(), []
    for i, flag in enumerate(flags + [False]):
        if flag:
            run.append(i)
        else:
            if len(run) >= min_run:
                remove.update(run)
            run = []
    return "\n".join(line for i, line in enumerate(lines) if i not in remove)
