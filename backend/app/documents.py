"""Validated in-memory document sources for chat attachments."""

import io
import re
import logging

from dataclasses import dataclass
from pathlib import PurePosixPath
from uuid import UUID

from pypdf import PdfReader

from app.ocr import docling_pages

MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_PDF_CHARACTERS = 200_000
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

logger = logging.getLogger(__name__)
EXTENSIONS = {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}

@dataclass(frozen=True, slots=True)
class UploadedSource:
    id: str
    filename: str
    media_type: str
    data: bytes
    pages: tuple[str, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)


def create_source(source_id: str, filename: str, media_type: str, data: bytes) -> UploadedSource:
    """Validate file identity, size, type, and PDF structure before retaining bytes."""
    try:
        normalized_id = str(UUID(source_id))
    except ValueError as error:
        raise ValueError("Invalid attachment ID.") from error

    if not data:
        raise ValueError("The attachment is empty.")
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise ValueError("Attachments must be 10 MB or smaller.")

    safe_filename = PurePosixPath(filename.replace("\\", "/")).name
    safe_filename = re.sub(r"[\x00-\x1f\x7f]", "", safe_filename).strip()
    if not safe_filename:
        raise ValueError("A filename is required.")

    if media_type == "application/pdf":
        if not data.startswith(b"%PDF-"):
            raise ValueError("The file does not appear to be a valid PDF.")
        try:
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported.")
            if not reader.pages or len(reader.pages) > MAX_PDF_PAGES:
                raise ValueError("PDFs must contain between 1 and 100 pages.")
            pypdf_pages = tuple(page.extract_text() or "" for page in reader.pages)
            try:
                pages = docling_pages(data, f"{normalized_id}.pdf", len(reader.pages))
            except Exception:
                logger.exception("Docling failed; using pypdf text")
                pages = pypdf_pages
            if sum(len(page) for page in pages) > MAX_PDF_CHARACTERS:
                raise ValueError("PDF text must be 200,000 characters or fewer.")
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("The PDF could not be read.") from error
        
    elif media_type in ALLOWED_IMAGE_TYPES:
        valid_signature = (
            (media_type == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n"))
            or (media_type == "image/jpeg" and data.startswith(b"\xff\xd8\xff"))
            or (media_type == "image/webp" and data.startswith(b"RIFF") and data[8:12] == b"WEBP")
        )
        if not valid_signature:
            raise ValueError("The file contents do not match the selected image type.")
        try:
            pages = docling_pages(data, f"{normalized_id}{EXTENSIONS[media_type]}", 1)
        except Exception:
            logger.exception("Docling OCR failed for image")
            pages = ("",)
    else:
        raise ValueError("Only PDF, JPEG, PNG, and WebP files are supported.")

    return UploadedSource(
        id=normalized_id,
        filename=safe_filename,
        media_type=media_type,
        data=data,
        pages=pages,
    )
