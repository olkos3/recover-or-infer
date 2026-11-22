# Recover or Infer

Training-free document structure for retrieval over long documents. This is a reference implementation of the paper *Recover or Infer: Training-Free Document Structure for Retrieval over Long Documents* (Vadym Soroka, 2026).

> **About this code.** This is a compact re-implementation of the method, written for clarity and reuse. It follows the paper's algorithm and frozen parameters, but it is **not the research code that produced the paper's numbers**.
>
> In a check against the paper's frozen runs on the same agreements, the flat baseline reproduced the per-document results exactly. The ToC and no-ToC routes differed per document by a few points in both directions. Their structure comes from LLM calls, which are not deterministic, and the research code handles some edge cases this version omits ([details](reproduce/README.md#4-what-to-expect)).
>
> The heading route and the default dense ranking were checked against the research code with its cached embeddings: on the 36 agreements of the paper's fifth credit holdout, the headings and chunks are identical, and so are the top-50 candidate pools of the 54 query-and-route pairs checked.
>
> The frozen protocols and per-query results of the paper's experiments are available from the author on request.

Fixed-size chunking cuts a long document into passages that no longer say which section they belong to. Contextual retrieval fixes this with one LLM call per chunk. This method instead gives every chunk the document's own structure, with a few LLM calls per document:

The structure comes from a cascade:

1. **ToC route.** If the document has a table of contents, a small LLM finds it and extracts the entries. Each entry is then aligned to the page and offset where its body heading starts.
2. **Heading route.** Otherwise, if a simple pattern finds at least 10 headings in the body (ARTICLE lines, `Section 7.2 Title`, `7.2 Title`, also at the start of a run-on line), those headings give the sections. No LLM call.
3. **No-ToC route (inferred structure).** Otherwise the text is cut into 150-token blocks and embedded. Dissimilarity peaks between neighbouring blocks split it into sections and segments, and one LLM call titles them.

Every route produces a section tree, and all feed the same retriever:

- Chunks of at most 350 tokens respect section boundaries, after sections shorter than 150 tokens have been merged with a neighbour. Each chunk carries its **breadcrumb** (e.g. `ARTICLE VII EVENTS OF DEFAULT > Section 7.2 Grace Periods and Cure Rights`).
- The breadcrumb is embedded together with the chunk text, so a chunk that starts mid-clause still carries the topic of its section.
- Dense retrieval over *breadcrumb + text* takes the top 50 chunks, and a cross-encoder reranks them. This is the configuration the paper recommends (`Config.fusion = "dense"`, the default).
- The paper's earlier frozen comparisons (all before its fifth credit holdout) instead fused three ranked lists by reciprocal rank fusion: dense, BM25 over breadcrumbs and BM25 over text. It is available as `Config(fusion="three_list")` and is what reproduces the paper's earlier tables; the paper's post-hoc and fresh-holdout results show the BM25 lists can be dropped.

```
document ──► ToC found? ──yes──────────────────────► recover + align ToC ──┐
                 │                                                         │
                 └─no─► ≥ 10 headings found? ──yes──► headings from text ──┼─► section tree ─► chunks + breadcrumbs ─► dense top 50 ─► rerank ─► context
                               │                                           │
                               └─no──────────────────► infer + label ──────┘
```

## Install

```bash
git clone https://github.com/olkos3/recover-or-infer && cd recover-or-infer
pip install -e .
export OPENAI_API_KEY=...        # embeddings (text-embedding-3-small) and the structure LLM
```

## Quick start

```bash
python examples/quickstart.py                       # fictional sample agreement, ToC route
python examples/quickstart.py --route no_toc        # same document, structure inferred instead
python examples/quickstart.py --doc report.pdf --question "What are the termination rights?"
python examples/quickstart.py --rerank              # add the cross-encoder (downloads ~280 MB once)
```

In Python:

```python
from recover_or_infer import OpenAIClient, StructureIndex, load_pages

llm = OpenAIClient(cache_dir=".cache/openai")              # API responses cached on disk
index = StructureIndex(load_pages("agreement.pdf"), llm)   # route="auto": ToC, else headings, else inferred
chunks = index.search("Which events of default have grace periods?", budget=2000)
print(StructureIndex.format_context(chunks))               # numbered excerpts under their breadcrumbs
```

`route` can be `"auto"` (the cascade), `"toc"`, `"headings"`, `"no_toc"` or `"flat"` (the baseline). Retrieval is dense + reranker by default; pass `config=Config(fusion="three_list")` for the fusion of the paper's earlier frozen comparisons. Every parameter of the paper is fixed in [`recover_or_infer/config.py`](recover_or_infer/config.py).

## Code map

| File | What it does |
|---|---|
| [`toc_route.py`](recover_or_infer/toc_route.py) | Contents-page detection, entry extraction, alignment of entries to body headings |
| [`heading_route.py`](recover_or_infer/heading_route.py) | Headings read off the body text by a pattern (the paper's inline-aware regex), no LLM call |
| [`no_toc_route.py`](recover_or_infer/no_toc_route.py) | Dissimilarity segmentation with the calibrated setting, one labelling call |
| [`chunking.py`](recover_or_infer/chunking.py) | Section-bounded, equal-size chunks; packing of tiny sections; splitting long definition sections |
| [`retrieval.py`](recover_or_infer/retrieval.py) | Breadcrumb-aware embeddings, dense ranking (default) or three-list fusion, budget fill |
| [`reranker.py`](recover_or_infer/reranker.py) | Jina reranker v2 base (int8 ONNX, CPU), with a hash-checked download |
| [`pipeline.py`](recover_or_infer/pipeline.py) | `StructureIndex`: the cascade (route choice) and search |

## Reproducing the paper

[`reproduce/README.md`](reproduce/README.md) gives step-by-step instructions for re-running the paper's merger-agreement comparison (flat chunks, ToC route, no-ToC route) on the same 30 LegalBench-RAG agreements, with the paper's metric and statistics, and explains how this implementation relates to the code behind the paper's numbers.

Main results reported in the paper (evidence recall or checklist coverage, %; protocols frozen before evaluation). The first four rows use the frozen three-list fusion; the last row is the fresh credit holdout frozen for the recommended dense + reranker configuration (contextual retrieval there: one call per chunk; flat chunks: dense + reranker):

| Test set | Flat chunks | Contextual retrieval | ToC route | No-ToC route |
|---|---:|---:|---:|---:|
| Merger agreements, MAUD-005 (30 docs), recall @ 8k | 77.3 | 84.6 | 85.1 | 83.6 |
| Merger agreements, MAUD-005, recall @ 2k | 48.4 | 64.6 | 61.4 | 59.2 |
| Credit agreements, fresh holdout (24 docs), coverage @ 8k | 52.9 (61.8 dense-only) | 58.7 | 71.8 | 63.5 |
| Contracts without a ToC, CUAD (36 docs), recall @ 2k | 67.8 | 74.0 | — | 71.5 |
| Credit agreements, fresh holdout 5 (36 docs), dense + reranker, coverage @ 8k | 69.2 | 78.0 | 79.7 | 75.3 |

Indexing cost: 39× fewer LLM calls than contextual retrieval on the merger agreements. See the paper for intervals, confirmations, answer-accuracy results and limitations.

## Citation

```bibtex
@misc{soroka2026recover,
  title  = {Recover or Infer: Training-Free Document Structure for Retrieval over Long Documents},
  author = {Soroka, Vadym},
  year   = {2026},
  url    = {https://github.com/olkos3/recover-or-infer}
}
```

## License

MIT ([`LICENSE`](LICENSE)). The sample agreement in `examples/` is fictional. Benchmarks and documents you evaluate on keep their own licences.
