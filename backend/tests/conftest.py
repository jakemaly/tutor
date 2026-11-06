import pytest

@pytest.fixture(autouse=True)
def fake_docling(monkeypatch):
    monkeypatch.setattr("app.documents.docling_pages", lambda data, name, n: ("",) * n)

    