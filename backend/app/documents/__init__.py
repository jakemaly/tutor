"""Bounded study-document generation: validated structure in, fixed-template file bytes out."""

from dataclasses import dataclass
from typing import Literal

from app.documents.render_docx import render_docx
from app.documents.render_pdf import render_pdf
from app.documents.schema import StudyGuide, filename_stem

DocumentFormat = Literal["pdf", "docx"]
MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
# A hard ceiling on output size, regardless of what the schema permits.
MAX_DOCUMENT_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class RenderedDocument:
    filename: str
    media_type: str
    data: bytes


def render_document(guide: StudyGuide, fmt: DocumentFormat) -> RenderedDocument:
    """Dispatch to a fixed renderer; the format is an allowlist, never a model-chosen path."""
    renderers = {"pdf": render_pdf, "docx": render_docx}
    if fmt not in renderers:
        raise ValueError(f"Unsupported format {fmt!r}. Use 'pdf' or 'docx'.")
    data = renderers[fmt](guide)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("The generated document is too large. Condense it.")
    return RenderedDocument(f"{filename_stem(guide)}.{fmt}", MEDIA_TYPES[fmt], data)
