"""Reproduce the paper's MAUD comparison: flat chunks vs ToC route vs no-ToC route on LegalBench-RAG merger agreements.

Setting, as in the paper: one known document per query; queries are document-scoped (the leading clause
"Consider the Merger Agreement between ...;" is removed); the context is filled to a budget of raw chunk-text
tokens (2,000 and 8,000); the metric is gold-character recall, the share of the lawyer-marked answer characters
that the context contains, averaged per document and then over documents. Differences are paired bootstrap
intervals over documents (10,000 resamples, seed 42).

    python reproduce/evaluate_maud.py --archive path/to/legalbenchrag.zip                # the paper's 30 agreements
    python reproduce/evaluate_maud.py --archive path/to/legalbenchrag.zip --limit 3      # a quick check
"""
import argparse
import hashlib
import json
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from recover_or_infer import OpenAIClient, StructureIndex, paginate

HERE = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d"
PAPER = {"flat": (48.4, 77.3), "toc": (61.4, 85.1), "no_toc": (59.2, 83.6)}   # Table 2 of the paper: R@2k, R@8k


def strip_prefix(query):
    return query.split(";", 1)[1].strip() if query.startswith("Consider") and ";" in query else query


def load_maud(archive, files):
    """Documents (normalised text) and their single-document queries with gold character spans."""
    z = zipfile.ZipFile(archive)
    names = {unicodedata.normalize("NFC", n): n for n in z.namelist()}
    tests = json.loads(z.read("benchmarks/maud.json"))["tests"]
    by_file = defaultdict(list)
    for t in tests:
        paths = {s["file_path"] for s in t["snippets"]}
        if len(paths) == 1:
            by_file[paths.pop()].append(t)
    docs = []
    for f in files:
        text = z.read(names[unicodedata.normalize("NFC", "corpus/" + f)]).decode("utf-8").replace("\r\n", "\n")
        for t in by_file[f]:                    # LegalBench-RAG spans index the text after CRLF -> LF normalisation
            for s in t["snippets"]:
                assert text[s["span"][0]:s["span"][1]] == s["answer"], f"span mismatch in {f}"
        docs.append({"file": f, "text": text, "queries": [{"text": strip_prefix(t["query"]), "gold": [s["span"] for s in t["snippets"]]} for t in by_file[f]]})
    return docs


def to_joined(pages):
    """Map an offset in the raw text to the joined document text (pages are joined with one newline each)."""
    bounds, start = [], 0
    for k, p in enumerate(pages):
        bounds.append((start, start + len(p["text"]), k))
        start += len(p["text"])
    def f(offset):
        for lo, hi, k in bounds:
            if offset < hi or k == len(bounds) - 1:
                return offset + k
    return f


def recall(chunks, gold, mapper):
    """Share of gold characters inside the selected chunks."""
    total = covered = 0
    for a, b in gold:
        a, b = mapper(a), mapper(b - 1) + 1
        total += b - a
        pieces = sorted((max(a, c["start"]), min(b, c["end"])) for c in chunks if min(b, c["end"]) > max(a, c["start"]))
        end = a
        for lo, hi in pieces:
            if hi > end:
                covered += hi - max(lo, end)
                end = hi
    return covered / total if total else 0.0


def bootstrap(diffs, n=10000, seed=42):
    d = np.asarray(diffs)
    means = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--archive", required=True, help="LegalBench-RAG data archive (zip)")
    a.add_argument("--files", default=str(HERE / "maud005_files.txt"))
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--methods", nargs="+", default=["flat", "toc", "no_toc"])
    a.add_argument("--budgets", nargs="+", type=int, default=[2000, 8000])
    a.add_argument("--reranker-dir", default="models/jina_v2_base", help="downloaded on first use")
    a.add_argument("--no-rerank", action="store_true")
    a.add_argument("--out", default="reproduce/maud_results.json")
    x = a.parse_args()

    if hashlib.sha256(Path(x.archive).read_bytes()).hexdigest() != ARCHIVE_SHA256:
        print("warning: this archive differs from the one used in the paper")
    files = [f for f in Path(x.files).read_text(encoding="utf-8").splitlines() if f.strip()]
    docs = load_maud(x.archive, files[:x.limit] if x.limit else files)
    llm = OpenAIClient(cache_dir=".cache/openai")
    reranker = None
    if not x.no_rerank:
        from recover_or_infer.reranker import JinaReranker, download
        reranker = JinaReranker(download(x.reranker_dir))

    per_doc = defaultdict(dict)                 # (method, budget) -> {file: mean recall}
    for i, d in enumerate(docs, 1):
        pages = paginate(d["text"])
        mapper = to_joined(pages)
        for m in x.methods:
            index = StructureIndex(pages, llm, route=m, doc_id=d["file"])
            for b in x.budgets:
                r = [recall(index.search(q["text"], budget=b, reranker=reranker), q["gold"], mapper) for q in d["queries"]]
                per_doc[(m, b)][d["file"]] = float(np.mean(r))
        print(f"[{i}/{len(docs)}] {d['file'][5:60]}  " + "  ".join(
            f"{m}@{b // 1000}k {100 * per_doc[(m, b)][d['file']]:.0f}" for m in x.methods for b in x.budgets), flush=True)

    print(f"\n{len(docs)} agreements, {sum(len(d['queries']) for d in docs)} queries, API calls {llm.calls}\n")
    print("Method      " + "".join(f"R@{b // 1000}k (paper)   " for b in x.budgets))
    for m in x.methods:
        cells = []
        for k, b in enumerate(x.budgets):
            paper = PAPER.get(m, (None, None))[k] if b in (2000, 8000) else None
            cells.append(f"{100 * np.mean(list(per_doc[(m, b)].values())):5.1f} ({paper if paper is not None else '-':>4})   ")
        print(f"{m:<11} " + "".join(cells))
    if "flat" in x.methods:
        print("\nDifference vs flat, paired bootstrap 95% interval over documents:")
        for m in [m for m in x.methods if m != "flat"]:
            for b in x.budgets:
                keys = sorted(per_doc[(m, b)])
                mean, lo, hi = bootstrap([per_doc[(m, b)][k] - per_doc[("flat", b)][k] for k in keys])
                print(f"  {m} - flat @ {b // 1000}k: {100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    Path(x.out).parent.mkdir(parents=True, exist_ok=True)
    Path(x.out).write_text(json.dumps({f"{m}@{b}": v for (m, b), v in per_doc.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
