"""Validated, bounded tools for presenting study UI and isolated HTML previews."""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

StudyElementKind = Literal["flashcard", "mcq", "true_false", "multi_select"]


class StudyOption(BaseModel):
    """A stable learner-facing choice, or the back of a flashcard."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    label: str = Field(min_length=1, max_length=500)

    @field_validator("label")
    @classmethod
    def label_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only answer text."""
        value = value.strip()
        if not value:
            raise ValueError("Option labels must not be blank.")
        return value


class StudyElementInput(BaseModel):
    """The small allowlisted component schema exposed to the tutor model."""

    model_config = ConfigDict(extra="forbid")

    kind: StudyElementKind
    question: str = Field(min_length=1, max_length=2000)
    options: list[StudyOption] = Field(min_length=1, max_length=8)
    correct_option_ids: list[str] = Field(default_factory=list, max_length=8)
    explanation: str | None = Field(default=None, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only prompts."""
        value = value.strip()
        if not value:
            raise ValueError("The study question must not be blank.")
        return value

    @field_validator("explanation")
    @classmethod
    def explanation_must_not_be_blank(cls, value: str | None) -> str | None:
        """Normalize an optional explanation without retaining whitespace-only text."""
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_options(self) -> "StudyElementInput":
        """Enforce component-specific choices, answer keys, and unique option IDs."""
        ids = [option.id for option in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError("Option IDs must be unique.")
        correct_ids = self.correct_option_ids
        if len(set(correct_ids)) != len(correct_ids):
            raise ValueError("Correct option IDs must be unique.")
        if not set(correct_ids).issubset(ids):
            raise ValueError("Correct option IDs must refer to an available option.")
        if self.kind == "flashcard" and len(self.options) != 1:
            raise ValueError("A flashcard must have exactly one answer.")
        if self.kind == "flashcard" and correct_ids:
            raise ValueError("Flashcards do not use correct option IDs.")
        if self.kind in {"mcq", "multi_select"} and len(self.options) < 2:
            raise ValueError("Questions must have at least two options.")
        if self.kind == "true_false":
            if len(self.options) != 2:
                raise ValueError("True/false questions must have exactly two options.")
            labels = {option.label.casefold() for option in self.options}
            if labels != {"true", "false"}:
                raise ValueError("True/false questions must have True and False options.")
        if self.kind in {"mcq", "true_false"} and len(correct_ids) != 1:
            raise ValueError("Single-answer questions must have exactly one correct option ID.")
        if self.kind == "multi_select" and not correct_ids:
            raise ValueError("Multi-select questions must have at least one correct option ID.")
        return self


class StudySetInput(BaseModel):
    """A bounded deck of native study elements presented together."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    elements: list[StudyElementInput] = Field(min_length=2, max_length=20)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only study set titles."""
        value = value.strip()
        if not value:
            raise ValueError("The study set title must not be blank.")
        return value


class HtmlPreviewInput(BaseModel):
    """Bound model-authored HTML before forwarding it to the isolated frame."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    html: str = Field(min_length=1, max_length=28_000)

    @field_validator("title", "html")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only preview content."""
        value = value.strip()
        if not value:
            raise ValueError("Preview content must not be blank.")
        return value


def make_present_study_element():
    """Create the native study-element tool for this graph."""

    @tool(args_schema=StudyElementInput)
    def present_study_element(
        kind: StudyElementKind,
        question: str,
        options: list[StudyOption],
        correct_option_ids: list[str] | None = None,
        explanation: str | None = None,
    ) -> dict[str, bool]:
        """Present and grade one flashcard or practice question inline."""
        return {"presented": True}

    return present_study_element


def make_present_study_elements():
    """Create the tool for showing a multi-item study set in one grouped deck."""

    @tool(args_schema=StudySetInput)
    def present_study_elements(title: str, elements: list[StudyElementInput]) -> dict[str, bool]:
        """Present a navigable deck of at least two validated flashcards or practice questions."""
        return {"presented": True}

    return present_study_elements


def make_present_html_preview():
    """Create the bounded HTML preview tool; the frontend must sandbox its output."""

    @tool(args_schema=HtmlPreviewInput)
    def present_html_preview(title: str, html: str) -> dict[str, bool]:
        """Show a visual HTML preview only when the learner explicitly requests one."""
        return {"presented": True}

    return present_html_preview
