# Reproducing the merger-agreement results

These steps re-run the paper's comparison of flat chunks, the ToC route and the no-ToC route on LegalBench-RAG merger agreements (MAUD), using the same 30 agreements as the paper's test set (MAUD-005), the same queries, metric and statistics.

## 1. Install and set the API key

```bash
pip install -e .
export OPENAI_API_KEY=...
```

## 2. Get the data

Download the LegalBench-RAG data archive from the [LegalBench-RAG repository](https://github.com/zeroentropy-ai/legalbenchrag) (link in its README) and save the zip, for example as `data/legalbenchrag.zip`. The paper used the archive with SHA-256

```
27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d
```

The script warns if your copy differs. [`maud005_files.txt`](maud005_files.txt) lists the 30 agreements. They were drawn by a fixed hash order from the MAUD agreements not used in development.

## 3. Run

```bash
python reproduce/evaluate_maud.py --archive data/legalbenchrag.zip --limit 3   # quick check, a few minutes
python reproduce/evaluate_maud.py --archive data/legalbenchrag.zip             # all 30 agreements, 344 queries
```

The first run downloads the reranker (Jina reranker v2 base, int8 ONNX, about 280 MB). It is checked against the SHA-256 that Hugging Face publishes. API responses are cached in `.cache/openai`, so a re-run makes no new calls.

For every agreement and method (`flat`, `toc`, `no_toc`), the script:

1. Cuts the text into 3,000-character pseudo-pages, as in the paper.
2. Builds the index. The ToC route uses about 3–4 LLM calls per agreement and the no-ToC route one, plus embeddings.
3. Retrieves a context for each query at 2,000 and 8,000 tokens of raw chunk text.
4. Computes **gold-character recall**: the share of the lawyer-marked answer characters inside the context.

Queries are **document-scoped**: the leading clause *"Consider the Merger Agreement between X and Y;"* is removed. Within a known document that clause carries no information, but it distorts reranking (see the paper's section on the document-prefix pitfall).

The output gives the document-averaged recall of each method next to the paper's values, and the difference from flat chunking with a paired bootstrap 95% interval over documents (10,000 resamples, seed 42). Per-document results are written to `reproduce/maud_results.json`.

## 4. What to expect

The paper's values (Table 11, in App. B) were measured with three-list fusion, which is the script's default (`--fusion three_list`). With `--fusion dense` the script runs the recommended dense + reranker configuration instead; compare that run with the MAUD-005 columns of the paper's Table 3. The paper's Table 11 values are:

| Method | R@2k | R@8k |
|---|---:|---:|
| Flat chunks | 48.4 | 77.3 |
| ToC route | 61.4 | 85.1 |
| No-ToC route | 59.2 | 83.6 |

This repository is a compact re-implementation of the paper's method. The frozen experiments used the research code, which follows the same algorithm and parameters but differs in several ways:

- It also scans for local contents lists later in a document.
- It gives front-matter runs LLM-written titles.
- It has extra alignment rules for page labels of SEC filings.

Expect the same ordering and similar gaps, not identical numbers. Results also vary slightly between runs because LLM outputs are not deterministic.

A check on the first three agreements of the list (`--limit 3`, 36 queries, 46 API calls) compared this implementation with the paper's frozen per-document results for the same agreements:

| Method | This implementation, R@2k / R@8k | Paper's frozen runs, same 3 agreements |
|---|---:|---:|
| Flat chunks | 38.4 / 72.2 | 38.4 / 72.2 (identical per agreement) |
| ToC route | 59.7 / 81.6 | 56.1 / 83.4 |
| No-ToC route | 60.8 / 83.7 | 56.9 / 74.5 |

The flat baseline reproduces exactly, which confirms the chunking, embeddings, fusion, reranking and metric. The structure routes differ per agreement by a few points in both directions, because the structure comes from LLM calls and the simplifications above. Three agreements are far too few for conclusions; the 30-agreement run is the comparison to make.

## Not covered here

- **Credit agreements (EDGAR).** These need document selection from EDGAR, PDF rendering and machine-annotated evidence checklists. The procedure is described in the paper, and the frozen protocols, selection logs and per-query results are available from the author on request.
- **Contextual retrieval.** The paper's baseline asks the LLM for a short context for every chunk, given the document opening and the 2,000 tokens before the chunk. It costs one call per chunk and is not included here.
- **Answer-accuracy experiments.** These use the contexts from step 3 with an answering and a grading LLM call; see the paper for the prompts.
