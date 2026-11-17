"""Build a structure-aware index for one document and search it."""
from .chunking import flat_chunks, section_chunks
from .config import Config
from .no_toc_route import infer_structure
from .retrieval import Index, with_breadcrumb
from .text import document_text
from .toc_route import recover_toc, toc_sections


class StructureIndex:
    """route='auto' uses the ToC when one is recovered and infers structure otherwise; 'toc', 'no_toc' and 'flat' force a route."""

    def __init__(self, pages, llm, route="auto", doc_id="doc", config=Config()):
        self.pages, self.config = pages, config
        self.text, self.spans = document_text(pages)
        self.route = route
        if route == "flat":
            self.chunks = flat_chunks(doc_id, self.text, config)
        else:
            entries = []
            if route in ("auto", "toc"):
                toc_pages, entries = recover_toc(pages, llm)
                if entries:
                    starts, paths = toc_sections(pages, self.spans, toc_pages, entries)
                    self.route = "toc"
                elif route == "toc":
                    raise ValueError("No usable table of contents was found.")
            if not entries:
                starts, paths = infer_structure(self.text, llm, config)
                self.route = "no_toc"
            self.chunks = section_chunks(doc_id, self.text, starts, paths, config)
        self.index = Index(self.chunks, llm, config, structured=self.route != "flat")

    def search(self, query, budget=2000, reranker=None):
        return self.index.search(query, budget=budget, reranker=reranker)

    @staticmethod
    def format_context(chunks):
        """Numbered excerpts, each under its breadcrumb (the format the paper's answer experiments used)."""
        return "\n\n".join(f"[{i}] " + with_breadcrumb(c) for i, c in enumerate(chunks, 1))
