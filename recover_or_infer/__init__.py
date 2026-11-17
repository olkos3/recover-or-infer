"""Recover or Infer: zero-shot document structure for retrieval over long documents (reference implementation)."""
from .config import Config
from .llm import OpenAIClient
from .pipeline import StructureIndex
from .text import load_pages, paginate

__all__ = ["Config", "OpenAIClient", "StructureIndex", "load_pages", "paginate"]
