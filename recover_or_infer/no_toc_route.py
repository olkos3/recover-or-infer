"""No-ToC route: infer a two-level structure from embedding dissimilarity and label it with one LLM call.

1. The text is cut into units of consecutive lines holding at least 150 tokens; each unit is embedded.
2. The dissimilarity at the gap after unit i is the cosine distance between the mean embeddings of the
   2 units before and the 2 units after it.
3. Sections: gaps above the 80th percentile, kept only if both sides hold at least 3,000 tokens.
   Segments: inside each section, gaps above the 50th percentile of the whole document, at least 100 tokens.
4. One LLM call titles every section and segment from the first ~250 tokens of each segment's prose.
   A segment's breadcrumb is [section title, segment title]; a section with a single segment keeps one level.
These settings were chosen on development agreements by retrieval quality, not by agreement with the ToC.
"""
import numpy as np

from .llm import obj
from .text import n_tokens, prose, windows

LABEL = ("Use 3-8 words per title. Each leaf excerpt is its first approximately 250 tokens of table-filtered prose. "
         "Synthesize section titles from their leaves. Ignore repeated navigation headers. "
         "Give concise content-grounded titles to each section and leaf segment. Use only the supplied excerpts. "
         "Return sections and segments objects keyed by their original IDs as strings, with a title for every key. Never renumber IDs.")


def make_units(text, unit_tokens):
    """Start offsets, token sizes and texts of line blocks with at least `unit_tokens` tokens."""
    starts, sizes, texts, current, current_start, current_tokens, pos = [], [], [], [], 0, 0, 0
    for line in text.splitlines(keepends=True):
        if not current:
            current_start = pos
        current.append(line)
        current_tokens += n_tokens(line)
        pos += len(line)
        if current_tokens >= unit_tokens:
            starts.append(current_start); sizes.append(current_tokens); texts.append("".join(current))
            current, current_tokens = [], 0
    if current:
        if starts:
            texts[-1] += "".join(current); sizes[-1] += current_tokens
        else:
            starts.append(current_start); sizes.append(current_tokens); texts.append("".join(current))
    return starts, sizes, texts


def dissimilarity(vectors, window):
    out = []
    for i in range(len(vectors) - 1):
        a = vectors[max(0, i - window + 1):i + 1].mean(0)
        b = vectors[i + 1:i + 1 + window].mean(0)
        out.append(float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return out


def boundaries(dist, sizes, percentile, minimum, threshold=None):
    """Unit indices that start a new region: gaps above the threshold leaving >= `minimum` tokens on both sides."""
    if not dist:
        return [0]
    t = float(np.percentile(dist, percentile)) if threshold is None else float(threshold)
    cum = np.concatenate([[0], np.cumsum(sizes)])
    selected = [0]
    for i in [i + 1 for i, d in enumerate(dist) if d > t and d > 1e-8]:
        if cum[i] - cum[selected[-1]] >= minimum and cum[-1] - cum[i] >= minimum:
            selected.append(i)
    return selected


def infer_structure(text, llm, cfg):
    """Section starts (offsets in `text`) and breadcrumb paths, using one labelling call."""
    starts, sizes, texts = make_units(text, cfg.unit_tokens)
    dist = dissimilarity(llm.embed(texts), cfg.window)
    sections = boundaries(dist, sizes, cfg.section_percentile, cfg.section_min_tokens)
    seg_threshold = float(np.percentile(dist, cfg.segment_percentile)) if dist else None
    leaves, leaf_starts = [], []
    for a, b in zip(sections, sections[1:] + [len(sizes)]):
        splits = [a + i for i in boundaries(dist[a:b - 1], sizes[a:b], cfg.segment_percentile, cfg.segment_min_tokens, threshold=seg_threshold)]
        for x, y in zip(splits, splits[1:] + [b]):
            region = prose("\n".join(texts[x:y]), cfg.table_min_run_lines, cfg.table_short_line_max_words)
            first = next(iter(windows(region, cfg.label_excerpt_tokens)), None)
            leaves.append({"section_id": a, "excerpt": region[first[0]:first[1]] if first else ""})
            leaf_starts.append(starts[x])
    section_ids = sorted({leaf["section_id"] for leaf in leaves})
    title = {"type": "string", "maxLength": 100}
    r = llm.structured(LABEL, {"expected_section_ids": section_ids, "expected_segment_ids": list(range(len(leaves))),
                               "leaves": [dict(segment_id=i, **leaf) for i, leaf in enumerate(leaves)]},
                       obj({"sections": obj({str(i): title for i in section_ids}),
                            "segments": obj({str(i): title for i in range(len(leaves))})}))
    seg_titles = {int(k): v.strip() for k, v in r["segments"].items()}
    sec_titles = {int(k): v.strip() for k, v in r["sections"].items()}
    for i, leaf in enumerate(leaves):                   # empty titles fall back to the first words of the excerpt
        seg_titles[i] = seg_titles[i] or " ".join(leaf["excerpt"].split()[:8]) or "Untitled"
    count = {s: sum(leaf["section_id"] == s for leaf in leaves) for s in section_ids}
    paths = []
    for i, leaf in enumerate(leaves):
        section = sec_titles[leaf["section_id"]] or seg_titles[i]
        paths.append([section] if count[leaf["section_id"]] == 1 else [section, seg_titles[i]])
    return leaf_starts, paths
