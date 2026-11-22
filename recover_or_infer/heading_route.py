"""Heading route: section structure read off the body text by a pattern, without any LLM call.

The middle step of the cascade (ToC, else headings, else inferred structure). Level-1 headings are ARTICLE lines and
numbered all-caps titles; level-2 headings are "Section n.n Title" and "n.n Title", either on their own line or at
the start of a line that runs on into the body ("Section 7.2 Termination. The Company may ..."). For run-on
headings the title must start with a capital letter and end at the first period within 120 characters, so
cross-references such as "Section 7.2 of the Agreement" do not match. The breadcrumb is article > section.
These are the patterns of the paper's inline-aware regex baseline.
"""
import re

H1 = [re.compile(r"^\s*(?:ARTICLE|Article)\s+[IVXLC\d]+\b[^\n]{0,100}$"),
      re.compile(r"^\s*\d{1,2}\.\s+[A-Z][A-Z0-9 ,;&/'’()-]{2,80}\.?\s*$")]
H2 = [re.compile(r"^\s*(?:SECTION|Section)\s+\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{0,200}$"),
      re.compile(r"^\s*\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\.?\s+[A-Z][^\n]{0,200}$")]
INLINE = [re.compile(r"^\s*(?:SECTION|Section)\s+\d+(?:\.\d+)+\.?\s+([A-Z][^.\n]{2,120})\."),
          re.compile(r"^\s*\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\.?\s+([A-Z][^.\n]{2,120})\.")]


def heading_title(line):
    """The heading text: the line up to the end of its title, at most 100 characters."""
    t = re.sub(r"\s+\d{1,3}$", "", " ".join(line.split()))       # drop a trailing page number (contents-page lines)
    m = re.match(r"^((?:ARTICLE|Article|SECTION|Section)?\s*[\dIVXLC.]+\.?\s+[^.]{1,100}?)\.(?:\s|$)", t)
    return (m.group(1) if m else t)[:100].strip()


def find_headings(text):
    """Section starts (character offsets) and breadcrumb paths. starts[0] is 0 ("Document start");
    the number of headings found is len(starts) - 1."""
    starts, paths, article, pos = [0], [["Document start"]], None, 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        level = 1 if any(p.match(line) for p in H1) else 2 if any(p.match(line) for p in H2) else 0
        title = None
        if level:
            title = heading_title(line)
            if level == 1:
                nxt = next((l.strip() for l in lines[i + 1:i + 3] if l.strip()), "")
                if re.fullmatch(r"(?:ARTICLE|Article)\s+[IVXLC\d]+", title) and nxt.isupper() and len(nxt) <= 80:
                    title = title + " " + nxt                          # "ARTICLE V" + title on the next line
        else:
            m = next((p.match(line) for p in INLINE if p.match(line)), None)
            if m:
                level, title = 2, " ".join(line[:m.end()].split()).rstrip(".")[:100]
        if level:
            if level == 1:
                article, path = title, [title]
            else:
                path = ([article] if article else []) + [title]
            if pos > starts[-1]:
                starts.append(pos)
                paths.append(path)
            else:
                paths[-1] = path
        pos += len(line) + 1
    return starts, paths
