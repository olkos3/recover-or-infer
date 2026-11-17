"""Section-aware chunking (v4): chunks never cross a leaf section and carry the leaf's breadcrumb.

- A leaf of T tokens is cut into ceil(T/350) pieces of about equal size (no short tail), cuts snapped to line breaks.
- Leaves under 150 tokens are packed with a neighbour; the merged breadcrumb joins their titles.
- A leaf over 2,000 tokens with at least 5 paragraph-initial definitions ('"Borrower" means ...') is split at each
  definition, and the defined term is appended to the breadcrumb.
"""
import math
import re

from .text import n_tokens

DEFINITION = re.compile(r'[“"]([A-Z][A-Za-z0-9&\'’\-]*(?:\s+[A-Za-z0-9&\'’\-]+){0,7})[”"]\s*(?:\([^)]{0,80}\)\s*)?'
                        r'(?:means|shall\s+mean|has\s+the\s+meaning|shall\s+have\s+the\s+meaning)\b')


def leaves_from_starts(text, starts, paths):
    """Consecutive regions [start_i, start_{i+1}) with their paths; regions with the same path are merged."""
    out = []
    for (a, path), b in zip(zip(starts, paths), starts[1:] + [len(text)]):
        if b <= a:
            continue
        if out and out[-1]["path"] == path and out[-1]["end"] == a:
            out[-1]["end"] = b
        else:
            out.append({"start": a, "end": b, "path": list(path)})
    return out


def definition_units(text, leaf, cfg):
    segment = text[leaf["start"]:leaf["end"]]
    starts = []
    for m in DEFINITION.finditer(segment):
        line_start = segment.rfind("\n", 0, m.start()) + 1
        if m.start() - line_start <= 12:
            starts.append((line_start, " ".join(m.group(1).split())))
    if len(starts) < cfg.definitions_min:
        return None
    out = []
    if starts[0][0] > 0:
        out.append({"start": leaf["start"], "end": leaf["start"] + starts[0][0], "path": leaf["path"]})
    for (a, term), (b, _) in zip(starts, starts[1:] + [(len(segment), None)]):
        if b > a:
            out.append({"start": leaf["start"] + a, "end": leaf["start"] + b, "path": leaf["path"] + [term]})
    return out


def merged_path(paths):
    if all(p == paths[0] for p in paths):
        return paths[0]
    k = 0
    while all(len(p) > k for p in paths) and len({tuple(p[:k + 1]) for p in paths}) == 1:
        k += 1
    labels = list(dict.fromkeys(" > ".join(p[k:]) for p in paths if p[k:]))
    return paths[0][:k] + (["; ".join(labels)] if labels else [])


def pack(units, text, cap, minimum):
    groups = []
    for u in units:
        t = n_tokens(text[u["start"]:u["end"]])
        if groups and groups[-1]["tokens"] + t <= cap and (groups[-1]["tokens"] < minimum or t < minimum):
            g = groups[-1]; g["end"] = u["end"]; g["paths"].append(u["path"]); g["tokens"] += t
        else:
            groups.append({"start": u["start"], "end": u["end"], "paths": [u["path"]], "tokens": t})
    return groups


def _cut_at_tokens(text, target):
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if n_tokens(text[:mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _snap(text, pos, lo):
    for sep in ("\n", " "):
        k = text.rfind(sep, max(lo + 1, pos - 300), pos)
        if k > lo:
            return k + 1
    return pos


def balanced(text, lo, hi, cap):
    total = n_tokens(text[lo:hi])
    n = max(1, math.ceil(total / cap))
    if n == 1:
        return [(lo, hi)]
    segment, cuts = text[lo:hi], [0]
    for k in range(1, n):
        c = _snap(segment, _cut_at_tokens(segment, round(k * total / n)), cuts[-1])
        if c > cuts[-1]:
            cuts.append(c)
    cuts.append(len(segment))
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if n_tokens(segment[a:b]) <= cap:
            out.append((lo + a, lo + b))
        else:                                            # rare: snapping overflowed the cap
            m = _snap(segment, _cut_at_tokens(segment[:b], n_tokens(segment[:a]) + cap), a)
            out += [(lo + a, lo + m), (lo + m, lo + b)]
    return out


def section_chunks(doc_id, text, starts, paths, cfg):
    units = []
    for leaf in leaves_from_starts(text, starts, paths):
        sub = None
        if n_tokens(text[leaf["start"]:leaf["end"]]) > cfg.definitions_split_tokens:
            sub = definition_units(text, leaf, cfg)
        units += sub or [leaf]
    chunks = []
    for g in pack(units, text, cfg.chunk_tokens, cfg.min_chunk_tokens):
        path = merged_path(g["paths"])
        for lo, hi in balanced(text, g["start"], g["end"], cfg.chunk_tokens):
            if text[lo:hi].strip():
                chunks.append({"id": f"{doc_id}:{lo}:{hi}", "start": lo, "end": hi, "text": text[lo:hi], "path": path})
    return chunks


def flat_chunks(doc_id, text, cfg):
    """The flat baseline: consecutive 350-token windows with no breadcrumb."""
    from .text import windows
    return [{"id": f"{doc_id}:{a}:{b}", "start": a, "end": b, "text": text[a:b], "path": []}
            for a, b in windows(text, cfg.chunk_tokens) if text[a:b].strip()]
