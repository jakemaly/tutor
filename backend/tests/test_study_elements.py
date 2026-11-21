"""Validation tests for the allowlisted native study and preview tools."""

import pytest
from pydantic import ValidationError

from app.study_elements import (
    HtmlPreviewInput,
    StudyElementInput,
    make_present_html_preview,
    make_present_study_element,
)


@pytest.mark.parametrize(
    ("kind", "options"),
    [
        ("flashcard", [{"id": "answer", "label": "Mitochondria"}]),
        ("mcq", [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]),
        ("true_false", [{"id": "true", "label": "True"}, {"id": "false", "label": "False"}]),
        ("multi_select", [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]),
    ],
)
def test_study_element_tool_accepts_each_native_component(kind, options) -> None:
    element = StudyElementInput.model_validate(
        {"kind": kind, "question": "Which answer?", "options": options}
    )
    result = make_present_study_element().invoke(element.model_dump())

    assert result == {"presented": True}


@pytest.mark.parametrize(
    ("kind", "options"),
    [
        ("flashcard", [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]),
        ("mcq", [{"id": "a", "label": "A"}]),
        ("true_false", [{"id": "true", "label": "True"}, {"id": "false", "label": "No"}]),
        (
            "true_false",
            [
                {"id": "true", "label": "True"},
                {"id": "false", "label": "False"},
                {"id": "extra", "label": "False"},
            ],
        ),
        ("multi_select", [{"id": "same", "label": "A"}, {"id": "same", "label": "B"}]),
    ],
)
def test_study_element_rejects_invalid_component_options(kind, options) -> None:
    with pytest.raises(ValidationError):
        StudyElementInput.model_validate(
            {"kind": kind, "question": "Which answer?", "options": options}
        )


def test_html_preview_size_is_bounded_before_it_reaches_the_frame() -> None:
    with pytest.raises(ValidationError):
        HtmlPreviewInput.model_validate({"title": "Preview", "html": "x" * 28_001})

    assert make_present_html_preview().invoke({"title": "Preview", "html": "<p>Example</p>"}) == {
        "presented": True
    }
