"""Validated study-guide structure: the only thing the model may author for a document.

The model fills in this data shape; fixed templates in render_pdf.py and render_docx.py decide
every font, margin, and color. Plain text only, with hard size limits, so a request can neither
inject markup/code nor produce an unbounded file.
"""

import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

MAX_TOTAL_CHARS = 40_000

# XML 1.0 forbids most control characters, and python-docx would raise on them.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f￾￿]")


def _clean(value: object) -> object:
    """Normalize line endings and drop control characters before length checks."""
    if not isinstance(value, str):
        return value
    return _CONTROL_CHARS.sub("", value.replace("\r\n", "\n").replace("\r", "\n"))


def _text(max_length: int):
    """A nonempty, trimmed, plain-text string type with a hard cap."""
    return Annotated[
        str,
        BeforeValidator(_clean),
        StringConstraints(strip_whitespace=True, min_length=1, max_length=max_length),
    ]


Text80, Text100, Text120, Text160 = _text(80), _text(100), _text(120), _text(160)
Text300, Text400, Text500, Text600, Text1500 = (
    _text(300),
    _text(400),
    _text(500),
    _text(600),
    _text(1500),
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paragraph(_Strict):
    type: Literal["paragraph"]
    text: Text1500


class Bullets(_Strict):
    type: Literal["bullets"]
    items: list[Text400] = Field(min_length=1, max_length=20)


class Steps(_Strict):
    """An ordered list, numbered by the template."""

    type: Literal["steps"]
    items: list[Text400] = Field(min_length=1, max_length=20)


class Term(_Strict):
    term: Text80
    definition: Text400


class KeyTerms(_Strict):
    type: Literal["key_terms"]
    terms: list[Term] = Field(min_length=1, max_length=20)


class WorkedExample(_Strict):
    type: Literal["worked_example"]
    problem: Text600
    steps: list[Text400] = Field(min_length=1, max_length=15)
    answer: Text300 | None = None


class Question(_Strict):
    question: Text500
    answer: Text500 | None = None


class PracticeQuestions(_Strict):
    """Questions for the reader; any answers are gathered into an answer key at the end."""

    type: Literal["practice"]
    questions: list[Question] = Field(min_length=1, max_length=15)


class Callout(_Strict):
    type: Literal["callout"]
    kind: Literal["Remember", "Tip", "Watch out"]
    text: Text500


Block = Annotated[
    Paragraph | Bullets | Steps | KeyTerms | WorkedExample | PracticeQuestions | Callout,
    Field(discriminator="type"),
]


class Section(_Strict):
    heading: Text100
    blocks: list[Block] = Field(min_length=1, max_length=12)


class StudyGuide(_Strict):
    title: Text120
    subtitle: Text160 | None = None
    sections: list[Section] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def _within_total_budget(self) -> "StudyGuide":
        total = sum(len(text) for text in _strings(self.model_dump()))
        if total > MAX_TOTAL_CHARS:
            raise ValueError(
                f"Document text is {total} characters; the limit is {MAX_TOTAL_CHARS}. "
                "Condense it to what the user asked for."
            )
        return self


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def answer_key(guide: StudyGuide) -> list[tuple[int, str]]:
    """Numbered answers for every practice question that has one, in document order."""
    key, number = [], 0
    for section in guide.sections:
        for block in section.blocks:
            if isinstance(block, PracticeQuestions):
                for question in block.questions:
                    number += 1
                    if question.answer:
                        key.append((number, question.answer))
    return key


def filename_stem(guide: StudyGuide) -> str:
    """An ASCII-only, header-safe file name derived from the title."""
    ascii_title = guide.title.encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:60].strip("-")
    return slug or "study-guide"
