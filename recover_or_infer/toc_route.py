"""ToC route: recover the author's table of contents and align its entries to the body.

1. One LLM call finds the contents pages among the first 20 pages.
2. One LLM call extracts the entries in order: title, nesting level, printed page label.
3. Each entry is placed on a physical body page. Deterministic evidence wins over the LLM:
   a unique whole-line heading in the body, then a unique printed page label (page footer);
   only entries still unresolved go to one LLM alignment call.
4. Within its page, an entry starts at its heading if the title text is found there.
5. A grouping entry (one with sub-entries) starts at its first descendant unless its own heading was verified.

Simplified relative to the paper's implementation: no scan for local contents lists later in the document,
no LLM titles for front-matter runs (text before the first entry is labelled "Front matter").
"""
import re

from .llm import obj

INT = {"type": "integer"}
STR = {"type": "string"}

DETECT = ("Identify physical pages containing the TABLE OF CONTENTS itself: a substantive list of section headings with page references. "
          "A repeated Table of Contents navigation/header on an ordinary body page is NOT a contents page. "
          "Do not classify ordinary tables or financial statements as contents pages. "
          "Return empty if no explicit ToC. Physical page numbers are 1-based.")
EXTRACT = ("Extract document section headings from the Table of Contents in order, including nested levels. "
           "Exclude lists of tables, lists of figures, individual tables/figures, and their captions. "
           "Preserve grouping: headings such as Part I and Part II are parent nodes, and numbered items beneath a part belong one level deeper. "
           "Do not add body headings that are absent from the ToC. level=1 is top-level. Preserve printed page labels as strings.")
ALIGN = ("Map each provided entry index to the physical page where its BODY heading starts. "
         "Match heading text to page excerpts; printed page labels may differ from physical pages. "
         "Return a matches object keyed by every entry index as a string.")


def printed_label(text):
    """The page label printed as the last line of a page ('12', 'iv', 'Page 12'), if any."""
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    m = re.fullmatch(r"(?:Page\s+)?([0-9]+|[ivxlcdm]+)", lines[-1], re.IGNORECASE) if lines else None
    return m.group(1).lower() if m else None


def heading_offset(text, title, minimum=0):
    """Character offset of the title in the page text (case-insensitive, whitespace-tolerant), else the page start."""
    pattern = r"\s+".join(re.escape(w) for w in title.split())
    m = re.search(pattern, text[minimum:], re.IGNORECASE)
    return (minimum + m.start(), True) if m else (minimum, False)


def recover_toc(pages, llm):
    """Returns (contents_pages, entries); entries carry title, level, page and offset. Empty entries: no usable ToC."""
    found = llm.structured(DETECT, [{"page": p["page"], "text": p["text"][:12000]} for p in pages[:20]],
                           obj({"toc_pages": {"type": "array", "items": INT}}))
    toc_pages = [n for n in found["toc_pages"] if any(p["page"] == n for p in pages)]
    if not toc_pages:
        return [], []
    extracted = llm.structured(EXTRACT, [p for p in pages if p["page"] in toc_pages],
                               obj({"entries": {"type": "array", "items": obj({"title": STR, "level": INT, "printed_page": STR})}}))
    entries = [e for e in extracted["entries"] if e["title"].strip() and e["level"] >= 1]
    if not entries:
        return toc_pages, []
    body = [p for p in pages if p["page"] not in toc_pages]

    placed, unresolved = {}, []
    for i, e in enumerate(entries):
        line = r"(?:^|\n)\s*" + r"\s+".join(re.escape(w) for w in e["title"].split()) + r"[.:]?\s*(?=\n|$)"
        heading_pages = [p["page"] for p in body if re.search(line, p["text"], re.IGNORECASE)]
        label_pages = [p["page"] for p in body if printed_label(p["text"]) == e["printed_page"].strip().lower()]
        if len(heading_pages) == 1:
            placed[i] = (heading_pages[0], True)
        elif len(label_pages) == 1:
            placed[i] = (label_pages[0], False)
        else:
            unresolved.append(i)
    if unresolved:
        page_schema = {"type": "integer", "enum": [p["page"] for p in body]}
        aligned = llm.structured(ALIGN, {
            "entries": {str(i): entries[i] for i in unresolved},
            "body_page_excerpts": [{"page": p["page"], "printed_label": printed_label(p["text"]), "text": p["text"][:600]} for p in body]},
            obj({"matches": obj({str(i): obj({"physical_page": page_schema}) for i in unresolved})}))
        for i in unresolved:
            placed[i] = (aligned["matches"][str(i)]["physical_page"], False)

    for i, e in enumerate(entries):                      # grouping entries start with their first descendant
        children = []
        for j in range(i + 1, len(entries)):
            if entries[j]["level"] <= e["level"]:
                break
            children.append(placed[j][0])
        if children and not (placed[i][1] and placed[i][0] <= min(children)):
            placed[i] = (min(children), placed[i][1])

    text_of = {p["page"]: p["text"] for p in pages}
    out, previous = [], (0, 0)
    for i, e in enumerate(entries):
        page = placed[i][0]
        minimum = previous[1] if previous[0] == page else 0
        offset, _ = heading_offset(text_of[page], e["title"], minimum)
        previous = (page, offset)
        out.append({"title": e["title"].strip(), "level": e["level"], "page": page, "offset": offset})
    return toc_pages, out


def toc_sections(pages, spans, toc_pages, entries):
    """Section starts (offsets in the joined text) and breadcrumb paths from aligned entries."""
    base = {s["page"]: s["start"] for s in spans}
    mapping, stack = {0: ["Front matter"]}, []
    for p in toc_pages:
        mapping[base[p]] = ["Table of Contents"]
    for e in entries:
        while stack and stack[-1]["level"] >= e["level"]:
            stack.pop()
        stack.append(e)
        mapping[base[e["page"]] + e["offset"]] = [x["title"] for x in stack]
    for p in toc_pages:                                   # the text after a contents page returns to the previous path
        end = base[p] + len(next(x["text"] for x in pages if x["page"] == p)) + 1
        if end not in mapping:
            before = [s for s in mapping if s < base[p] and mapping[s] != ["Table of Contents"]]
            mapping[end] = mapping[max(before)] if before else ["Front matter"]
    starts = sorted(mapping)
    return starts, [mapping[s] for s in starts]
