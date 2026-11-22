"""Retrieval over one document: candidate ranking, cross-encoder reranking, budget fill.

Two candidate rankings (Config.fusion):
- "dense" (default, the configuration the paper recommends): dense cosine over embeddings of "breadcrumb\\nchunk text".
- "three_list" (the configuration of the paper's frozen runs): reciprocal rank fusion of that dense list, BM25 over
  breadcrumbs alone and BM25 over chunk text alone. Flat baseline: dense over chunk text and BM25 over chunk text.
The top 50 candidates are reranked by a cross-encoder that sees "breadcrumb\\nchunk text". The context is filled in
rank order up to a budget of raw chunk-text tokens; breadcrumbs are not counted, although an answerer is shown them.
"""
import math
import re
from collections import Counter

import numpy as np

from .text import n_tokens


def with_breadcrumb(chunk):
    """What the embedding model and the reranker see: the breadcrumb line, then the chunk."""
    return (" > ".join(chunk["path"]) + "\n" if chunk["path"] else "") + chunk["text"]


def _words(text):
    return re.findall(r"\w+", text.casefold())


class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        self.docs = [Counter(_words(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(1, len(self.docs))
        self.df = Counter(w for d in self.docs for w in d)
        self.k1, self.b = k1, b

    def scores(self, query):
        n, terms = len(self.docs), set(_words(query))
        return [sum(math.log(1 + (n - self.df[t] + .5) / (self.df[t] + .5)) * d[t] * (self.k1 + 1)
                    / (d[t] + self.k1 * (1 - self.b + self.b * length / max(1, self.avg))) for t in terms if d[t])
                for d, length in zip(self.docs, self.lengths)]


class Index:
    def __init__(self, chunks, llm, cfg, structured=True):
        self.chunks, self.llm, self.cfg, self.structured = chunks, llm, cfg, structured
        if cfg.fusion not in ("dense", "three_list"):
            raise ValueError(f"Unknown fusion: {cfg.fusion}")
        self.vectors = llm.embed([with_breadcrumb(c) for c in chunks])
        if cfg.fusion == "dense":
            self.lexical = []
        elif structured:
            self.lexical = [BM25([" > ".join(c["path"]) for c in chunks]), BM25([c["text"] for c in chunks])]
        else:
            self.lexical = [BM25([c["text"] for c in chunks])]

    def fused(self, query):
        """Chunk indices, best first: by dense cosine ("dense") or by reciprocal rank fusion of the dense and BM25 lists."""
        dense = self.vectors @ self.llm.embed([query])[0]
        if self.cfg.fusion == "dense":
            return sorted(range(len(self.chunks)), key=lambda i: (-float(dense[i]), self.chunks[i]["id"]))
        lists = [(dense, False)] + [(np.asarray(b.scores(query)), True) for b in self.lexical]
        fused = np.zeros(len(self.chunks))
        for scores, sparse in lists:
            order = sorted(range(len(self.chunks)), key=lambda i: (-float(scores[i]), self.chunks[i]["id"]))
            if sparse:
                order = [i for i in order if scores[i] > 0]   # zero-score BM25 matches carry no evidence
            for rank, i in enumerate(order[:self.cfg.candidate_depth], 1):
                fused[i] += 1 / (self.cfg.rrf_constant + rank)
        return sorted(range(len(self.chunks)), key=lambda i: (-fused[i], self.chunks[i]["id"]))

    def search(self, query, budget=2000, reranker=None):
        """Chunks for a context of at most `budget` raw-text tokens, best first."""
        ordered = [self.chunks[i] for i in self.fused(query)]
        if reranker is not None:
            ordered = reranker.rerank(query, ordered[:self.cfg.rerank_top])
        selected, used = [], 0
        for c in ordered:
            n = n_tokens(c["text"])
            if used + n > budget:
                break
            selected.append(c)
            used += n
        return selected
