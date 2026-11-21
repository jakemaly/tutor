"""Validation tests for the allowlisted native study and preview tools."""

import pytest
from pydantic import ValidationError

from app.study_elements import (
    HtmlPreviewInput,
    StudyElementInput,
    StudySetInput,
    make_present_html_preview,
    make_present_study_element,
    make_present_study_elements,
)


@pytest.mark.parametrize(
    "element",
    [
        {
            "kind": "flashcard",
            "question": "What organelle produces ATP?",
            "options": [{"id": "answer", "label": "Mitochondria"}],
        },
        {
            "kind": "mcq",
            "question": "Which pigment captures light?",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "correct_option_ids": ["a"],
        },
        {
            "kind": "true_false",
            "question": "Plants have mitochondria.",
            "options": [{"id": "true", "label": "True"}, {"id": "false", "label": "False"}],
            "correct_option_ids": ["true"],
        },
        {
            "kind": "multi_select",
            "question": "Select the cell's membrane-bound organelles.",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "correct_option_ids": ["a", "b"],
        },
    ],
)
def test_study_element_tool_accepts_each_native_component(element) -> None:
    validated = StudyElementInput.model_validate(element)
    result = make_present_study_element().invoke(validated.model_dump())

    assert result == {"presented": True}


@pytest.mark.parametrize(
    "element",
    [
        {
            "kind": "flashcard",
            "question": "Question?",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        },
        {
            "kind": "mcq",
            "question": "Question?",
            "options": [{"id": "a", "label": "A"}],
            "correct_option_ids": ["a"],
        },
        {
            "kind": "mcq",
            "question": "Question?",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "correct_option_ids": ["missing"],
        },
        {
            "kind": "true_false",
            "question": "Question?",
            "options": [{"id": "true", "label": "True"}, {"id": "false", "label": "No"}],
            "correct_option_ids": ["true"],
        },
        {
            "kind": "multi_select",
            "question": "Question?",
            "options": [{"id": "same", "label": "A"}, {"id": "same", "label": "B"}],
            "correct_option_ids": ["same"],
        },
    ],
)
def test_study_element_rejects_invalid_options_or_answer_keys(element) -> None:
    with pytest.raises(ValidationError):
        StudyElementInput.model_validate(element)


def test_study_set_tool_validates_multiple_items_together() -> None:
    elements = [
        {
            "kind": "flashcard",
            "question": "What is ATP?",
            "options": [{"id": "answer", "label": "A cellular energy carrier"}],
        },
        {
            "kind": "multi_select",
            "question": "Select the products of photosynthesis.",
            "options": [{"id": "a", "label": "Glucose"}, {"id": "b", "label": "Oxygen"}],
            "correct_option_ids": ["a", "b"],
        },
    ]
    study_set = StudySetInput.model_validate({"title": "Cell biology", "elements": elements})

    assert make_present_study_elements().invoke(study_set.model_dump()) == {"presented": True}


def test_html_preview_size_is_bounded_before_it_reaches_the_frame() -> None:
    with pytest.raises(ValidationError):
        HtmlPreviewInput.model_validate({"title": "Preview", "html": "x" * 28_001})

    assert make_present_html_preview().invoke({"title": "Preview", "html": "<p>Example</p>"}) == {
        "presented": True
    }
