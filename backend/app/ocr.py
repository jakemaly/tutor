"""Docling text extraction, one markdown string per page"""

import io
from functools import lru_cache

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    return DocumentConverter()

def docling_pages(data: bytes, name: str, page_count: int) -> tuple[str, ...]:
    """Always returns exactly page_count strings"""
    doc = _converter().convert(DocumentStream(name=name, stream=io.BytesIO(data))).document
    return tuple(doc.export_to_markdown(page_no=n).strip() for n in range(1, page_count + 1))

