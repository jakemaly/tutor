import io

from pypdf import PdfWriter

from app.documents import create_source

SOURCE_ID = "123e4567-e89b-12d3-a456-426614174000"

def make_pdf(page_count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_pages_come_from_docling(monkeypatch):
    monkeypatch.setattr(
        "app.documents.docling_pages",
        lambda data, name, n: ("Page one text", "Page two text"),
    )

    source = create_source(SOURCE_ID, "notes.pdf", "application/pdf", make_pdf(2))

    assert source.pages == ("Page one text", "Page two text")


def test_pdf_falls_back_to_pypdf_when_docling_fails(monkeypatch):
    def boom(data, name, n):
        raise RuntimeError("docling broke")

    monkeypatch.setattr("app.documents.docling_pages", boom)

    source = create_source(SOURCE_ID, "notes.pdf", "application/pdf", make_pdf(2))

    assert len(source.pages) == 2  # upload still works, with pypdf's (blank) text