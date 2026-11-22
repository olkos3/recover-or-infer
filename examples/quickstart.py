"""Index one document with recovered or inferred structure, and retrieve a context for a question.

    export OPENAI_API_KEY=...
    python examples/quickstart.py                                   # sample agreement, ToC route
    python examples/quickstart.py --route no_toc                    # same document, structure inferred
    python examples/quickstart.py --route headings                  # same document, headings read by a pattern
    python examples/quickstart.py --doc my.pdf --question "..."     # your own document
    python examples/quickstart.py --rerank                          # add the cross-encoder (downloads ~280 MB once)
"""
import argparse
from pathlib import Path

from recover_or_infer import OpenAIClient, StructureIndex, load_pages

HERE = Path(__file__).resolve().parent


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--doc", default=str(HERE / "sample_agreement.txt"))
    a.add_argument("--question", default="What cure periods or thresholds prevent a breach from becoming an Event of Default?")
    a.add_argument("--route", default="auto", choices=["auto", "toc", "headings", "no_toc", "flat"])
    a.add_argument("--budget", type=int, default=1000, help="context budget in raw chunk-text tokens")
    a.add_argument("--rerank", action="store_true")
    x = a.parse_args()

    llm = OpenAIClient(cache_dir=".cache/openai")      # responses are cached on disk; reruns cost nothing
    index = StructureIndex(load_pages(x.doc), llm, route=x.route)
    print(f"route: {index.route} | {len(index.chunks)} chunks | API calls (LLM + embeddings): {llm.calls}\n")
    print("Sections (breadcrumbs):")
    for path in dict.fromkeys(" > ".join(c["path"]) for c in index.chunks):
        print("  ", path or "(none)")

    reranker = None
    if x.rerank:
        from recover_or_infer.reranker import JinaReranker, download
        reranker = JinaReranker(download())
    context = index.search(x.question, budget=x.budget, reranker=reranker)
    print(f"\nQuestion: {x.question}\n\nContext ({x.budget}-token budget):\n")
    print(StructureIndex.format_context(context))


if __name__ == "__main__":
    main()
