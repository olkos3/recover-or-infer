"""Build a structure-aware index for one document and search it."""
from .chunking import flat_chunks, section_chunks
from .config import Config
from .heading_route import find_headings
from .no_toc_route import infer_structure
from .retrieval import Index, with_breadcrumb
from .text import document_text
from .toc_route import recover_toc, toc_sections


class StructureIndex:
    """route='auto' is the paper's cascade: the recovered ToC if there is one, else headings read off the text by a
    pattern if it finds at least `config.min_headings`, else inferred structure. 'toc', 'headings', 'no_toc' and
    'flat' force one route ('no_toc' is the inferred structure; 'flat' the baseline)."""

    def __init__(self, pages, llm, route="auto", doc_id="doc", config=Config()):
        if route not in ("auto", "toc", "headings", "no_toc", "flat"):
            raise ValueError(f"Unknown route: {route}")
        self.pages, self.config = pages, config
        self.text, self.spans = document_text(pages)
        self.route = route
        if route == "flat":
            self.chunks = flat_chunks(doc_id, self.text, config)
        else:
            starts = None
            if route in ("auto", "toc"):
                toc_pages, entries = recover_toc(pages, llm)
                if entries:
                    starts, paths = toc_sections(pages, self.spans, toc_pages, entries)
                    self.route = "toc"
                elif route == "toc":
                    raise ValueError("No usable table of contents was found.")
            if starts is None and route in ("auto", "headings"):
                h_starts, h_paths = find_headings(self.text)
                if route == "headings" or len(h_starts) - 1 >= config.min_headings:
                    starts, paths = h_starts, h_paths
                    self.route = "headings"
            if starts is None:
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
