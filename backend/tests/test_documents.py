"""Schema limits, fixed-template rendering, and the temporary artifact store."""

from io import BytesIO

import pytest
from docx import Document
from pydantic import ValidationError
from pypdf import PdfReader

from app.artifacts import ArtifactStore
from app.documents import MAX_DOCUMENT_BYTES, render_document
from app.documents.schema import MAX_TOTAL_CHARS, StudyGuide, answer_key, filename_stem
from app.documents.wire import WireGuide, guide_to_dict, to_study_guide


def guide_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "title": "ε-δ Limit Proofs",
        "subtitle": "<b>not markup</b> & literal",
        "sections": [
            {
                "heading": "Core idea",
                "blocks": [
                    {"type": "paragraph", "text": "∀ε>0 ∃δ>0 with |x−a|<δ ⇒ |f(x)−L|<ε."},
                    {"type": "callout", "kind": "Watch out", "text": "δ never depends on x."},
                    {"type": "key_terms", "terms": [{"term": "ε", "definition": "Tolerance."}]},
                    {"type": "bullets", "items": ["one", "two"]},
                    {"type": "steps", "items": ["first", "second"]},
                    {
                        "type": "worked_example",
                        "problem": "Prove lim 3x+1 = 7 at 2.",
                        "steps": ["Take δ = ε/3."],
                        "answer": "7",
                    },
                    {
                        "type": "practice",
                        "questions": [
                            {"question": "Q one?", "answer": "A one"},
                            {"question": "Q two?"},
                            {"question": "Q three?", "answer": "A three"},
                        ],
                    },
                ],
            }
        ],
    }
    data.update(overrides)
    return data


def test_pdf_keeps_math_symbols_and_treats_markup_as_literal_text() -> None:
    document = render_document(StudyGuide.model_validate(guide_data()), "pdf")
    text = "".join(page.extract_text() for page in PdfReader(BytesIO(document.data)).pages)

    assert document.filename == "limit-proofs.pdf"
    assert document.media_type == "application/pdf"
    assert document.data.startswith(b"%PDF")
    assert "∀ε>0 ∃δ>0" in text
    assert "<b>not markup</b>" in text


def test_docx_contains_the_same_content() -> None:
    document = render_document(StudyGuide.model_validate(guide_data()), "docx")
    opened = Document(BytesIO(document.data))
    text = "\n".join(p.text for p in opened.paragraphs)

    assert document.filename == "limit-proofs.docx"
    assert "Core idea" in text and "Answer key" in text
    assert opened.core_properties.title == "ε-δ Limit Proofs"


def test_answer_key_keeps_question_numbers_and_skips_unanswered_questions() -> None:
    guide = StudyGuide.model_validate(guide_data())
    assert answer_key(guide) == [(1, "A one"), (3, "A three")]


@pytest.mark.parametrize("fmt", ["pdf", "docx"])
def test_control_characters_are_removed_instead_of_breaking_the_file(fmt: str) -> None:
    data = guide_data(title="Bad\x00 \x0b title")
    document = render_document(StudyGuide.model_validate(data), fmt)  # type: ignore[arg-type]
    assert document.data


def test_unsupported_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported format"):
        render_document(StudyGuide.model_validate(guide_data()), "html")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    [
        {"sections": []},
        {"title": ""},
        {"title": "x" * 121},
        {"unexpected": "field"},
        {"sections": [{"heading": "H", "blocks": [{"type": "script", "text": "x"}]}]},
        {"sections": [{"heading": "H", "blocks": [{"type": "paragraph", "text": "x", "css": 1}]}]},
        {"sections": [{"heading": "H", "blocks": []}]},
        {"sections": [{"heading": "H", "blocks": [{"type": "bullets", "items": ["x"] * 21}]}]},
    ],
)
def test_schema_rejects_out_of_bounds_or_unknown_structure(mutation: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        StudyGuide.model_validate(guide_data(**mutation))


def test_total_text_budget_is_enforced() -> None:
    paragraph = {"type": "paragraph", "text": "x" * 1500}
    sections = [{"heading": f"S{i}", "blocks": [paragraph] * 12} for i in range(12)]
    assert 12 * 12 * 1500 > MAX_TOTAL_CHARS
    with pytest.raises(ValidationError, match="limit is"):
        StudyGuide.model_validate(guide_data(sections=sections))


def test_maximum_size_guide_stays_under_the_output_ceiling() -> None:
    paragraph = {"type": "paragraph", "text": "word " * 280}
    sections = [{"heading": f"S{i}", "blocks": [paragraph] * 2} for i in range(12)]
    guide = StudyGuide.model_validate(guide_data(sections=sections))
    assert len(render_document(guide, "pdf").data) < MAX_DOCUMENT_BYTES


def test_filenames_are_ascii_and_never_empty() -> None:
    assert filename_stem(StudyGuide.model_validate(guide_data(title="../Intro: C++ & Q/A"))) == (
        "intro-c-q-a"
    )
    assert filename_stem(StudyGuide.model_validate(guide_data(title="εδ"))) == "study-guide"


def test_store_expires_and_evicts_oldest_first() -> None:
    now = [1000.0]
    store = ArtifactStore(ttl_seconds=60, max_items=2, clock=lambda: now[0])
    first = store.put("a.pdf", "application/pdf", b"a")
    second = store.put("b.pdf", "application/pdf", b"b")
    third = store.put("c.pdf", "application/pdf", b"c")

    assert store.get(first.id) is None
    assert store.get(second.id).data == b"b"
    assert store.get(third.id).data == b"c"
    now[0] += 61
    assert store.get(second.id) is None


def test_store_ids_are_unguessable_and_size_is_bounded() -> None:
    store = ArtifactStore(max_bytes=10)
    assert len(store.put("a", "text/plain", b"x").id) >= 32
    with pytest.raises(ValueError, match="too large"):
        store.put("b", "text/plain", b"x" * 11)


def test_a_complete_guide_round_trips_through_the_model_facing_shape() -> None:
    data = guide_data()

    assert guide_to_dict(WireGuide.model_validate(data)) == data
    assert to_study_guide(WireGuide.model_validate(data)) == StudyGuide.model_validate(data)


def test_fields_a_block_type_does_not_use_are_ignored() -> None:
    wire = WireGuide.model_validate(
        {
            "title": "T",
            "sections": [
                {
                    "heading": "H",
                    "blocks": [
                        # Models often fill unrelated fields; blanks and strays are dropped.
                        {"type": "bullets", "items": ["a"], "text": "stray", "kind": "Tip"},
                        {"type": "paragraph", "text": "p", "items": ["stray"], "answer": ""},
                        {"type": "practice", "questions": [{"question": "Q?", "answer": "  "}]},
                    ],
                }
            ],
        }
    )

    blocks = guide_to_dict(wire)["sections"][0]["blocks"]  # type: ignore[index]
    assert blocks == [
        {"type": "bullets", "items": ["a"]},
        {"type": "paragraph", "text": "p"},
        {"type": "practice", "questions": [{"question": "Q?"}]},
    ]
    assert to_study_guide(wire).sections[0].blocks[2].questions[0].answer is None


def test_missing_block_fields_are_reported_by_path() -> None:
    wire = WireGuide.model_validate(
        {"title": "T", "sections": [{"heading": "H", "blocks": [{"type": "key_terms"}]}]}
    )

    with pytest.raises(ValidationError) as caught:
        to_study_guide(wire)
    paths = [".".join(str(part) for part in error["loc"]) for error in caught.value.errors()]
    assert len(paths) == 1
    assert paths[0].startswith("sections.0.blocks.0") and paths[0].endswith("terms")


def test_object_items_from_a_model_become_plain_strings() -> None:
    items = ["a", {"term": "ε", "definition": "tolerance"}, {"text": "x"}, 3]
    wire = WireGuide.model_validate(
        {
            "title": "T",
            "sections": [{"heading": "H", "blocks": [{"type": "bullets", "items": items}]}],
        }
    )

    assert to_study_guide(wire).sections[0].blocks[0].items == ["a", "ε: tolerance", "x", "3"]
