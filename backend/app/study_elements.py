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

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only prompts."""
        value = value.strip()
        if not value:
            raise ValueError("The study question must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_options(self) -> "StudyElementInput":
        """Enforce component-specific choice counts and unique option IDs."""
        ids = [option.id for option in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError("Option IDs must be unique.")
        if self.kind == "flashcard" and len(self.options) != 1:
            raise ValueError("A flashcard must have exactly one answer.")
        if self.kind in {"mcq", "multi_select"} and len(self.options) < 2:
            raise ValueError("Questions must have at least two options.")
        if self.kind == "true_false":
            if len(self.options) != 2:
                raise ValueError("True/false questions must have exactly two options.")
            labels = {option.label.casefold() for option in self.options}
            if labels != {"true", "false"}:
                raise ValueError("True/false questions must have True and False options.")
        return self


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
        kind: StudyElementKind, question: str, options: list[StudyOption]
    ) -> dict[str, bool]:
        """Present one flashcard, multiple-choice, true/false, or multi-select item."""
        return {"presented": True}

    return present_study_element


def make_present_html_preview():
    """Create the bounded HTML preview tool; the frontend must sandbox its output."""

    @tool(args_schema=HtmlPreviewInput)
    def present_html_preview(title: str, html: str) -> dict[str, bool]:
        """Show a visual HTML preview only when the learner explicitly requests one."""
        return {"presented": True}

    return present_html_preview
