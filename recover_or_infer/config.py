"""Frozen parameters of the paper's system. Changing them gives a different system from the one evaluated."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Models
    llm_model: str = "gpt-6-luna"
    reasoning_effort: str = "low"
    embedding_model: str = "text-embedding-3-small"

    # Section-aware chunking (v4)
    chunk_tokens: int = 350            # no chunk is longer; a leaf of T tokens becomes ceil(T/350) equal pieces
    min_chunk_tokens: int = 150        # leaves shorter than this are packed with a neighbour
    definitions_split_tokens: int = 2000   # a leaf this long ...
    definitions_min: int = 5               # ... with this many paragraph-initial definitions is split per definition

    # No-ToC route: calibrated against recovered ToCs on 125 development agreements
    unit_tokens: int = 150             # text is cut into blocks of consecutive lines with >= 150 tokens
    window: int = 2                    # dissimilarity compares the mean embedding of 2 blocks on each side of a gap
    section_percentile: float = 80     # section boundaries: gaps above this percentile ...
    section_min_tokens: int = 3000     # ... leaving sections of at least this size
    segment_percentile: float = 50     # segment boundaries inside sections (threshold over the whole document) ...
    segment_min_tokens: int = 100      # ... leaving segments of at least this size
    label_excerpt_tokens: int = 250    # each segment is shown to the labelling call by its first ~250 tokens of prose
    table_min_run_lines: int = 4       # table-like line runs removed from label excerpts
    table_short_line_max_words: int = 4

    # Cascade: headings read off the text are used when the pattern finds at least this many
    min_headings: int = 10

    # Retrieval
    fusion: str = "dense"              # "dense": dense + reranker (recommended); "three_list": the frozen runs' fusion
    rrf_constant: int = 60             # three_list only
    candidate_depth: int = 100         # three_list only: each ranked list contributes its top 100 to the fusion
    rerank_top: int = 50               # the cross-encoder reorders the top 50 candidates
    reranker_max_tokens: int = 512
