"""What the model is shown: a flat, forgiving shape that is converted to the strict StudyGuide.

Models (and providers that check tool arguments themselves) handle a union of seven block
types badly: they mix fields between types, and the provider's error lists every branch. So
the tool advertises one block shape with a few plainly named fields. Anything a block type
does not use is ignored, then to_study_guide() builds the strict schema.py structure, which
still enforces every type, length, and size limit and names the exact bad field in its error.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.documents.schema import StudyGuide

BlockType = Literal[
    "paragraph", "bullets", "steps", "key_terms", "worked_example", "practice", "callout"
]

# The only fields each block type reads; everything else on a wire block is ignored.
_BLOCK_FIELDS: dict[str, tuple[str, ...]] = {
    "paragraph": ("text",),
    "bullets": ("items",),
    "steps": ("items",),
    "key_terms": ("terms",),
    "worked_example": ("problem", "steps", "answer"),
    "practice": ("questions",),
    "callout": ("kind", "text"),
}


class WireTerm(BaseModel):
    term: str = Field("", description="The word or symbol being defined.")
    definition: str = Field("", description="Its meaning, in one or two sentences.")


class WireQuestion(BaseModel):
    question: str = Field("", description="A question for the reader.")
    answer: str = Field("", description="Its answer; leave empty if none was given.")


class WireBlock(BaseModel):
    """One block. Set `type`, then only the fields that type uses."""

    type: BlockType
    text: str = Field("", description="paragraph or callout: the text.")
    kind: str = Field("", description="callout only: Remember, Tip, or Watch out.")
    items: list[Any] = Field(
        default_factory=list, description="bullets or steps: one plain string per item."
    )
    terms: list[WireTerm] = Field(default_factory=list, description="key_terms: the definitions.")
    problem: str = Field("", description="worked_example: the problem statement.")
    steps: list[Any] = Field(
        default_factory=list, description="worked_example: the solution steps, one string each."
    )
    answer: str = Field("", description="worked_example: the final answer.")
    questions: list[WireQuestion] = Field(
        default_factory=list, description="practice: questions for the reader."
    )


class WireSection(BaseModel):
    heading: str = Field(description="A short section heading.")
    blocks: list[WireBlock] = Field(description="1-12 blocks.")


class WireGuide(BaseModel):
    title: str
    subtitle: str = ""
    sections: list[WireSection] = Field(description="Usually 2-8 sections; at most 12.")


def _filled(value: object) -> bool:
    """Blank text and empty lists mean 'not provided'."""
    return bool(value.strip()) if isinstance(value, str) else bool(value)


def _item_text(item: object) -> str:
    """Keep the words when a model sends {"text": ...} or {"term": ..., "definition": ...}."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if item.get("text"):
            return str(item["text"])
        if item.get("term") or item.get("definition"):
            return ": ".join(str(item[key]) for key in ("term", "definition") if item.get(key))
        words = (str(v) for v in item.values() if isinstance(v, str | int | float))
        return " – ".join(word for word in words if word.strip())
    return str(item)


def _rows(rows: list, names: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {name: getattr(row, name) for name in names if _filled(getattr(row, name))} for row in rows
    ]


def _block_dict(block: WireBlock) -> dict[str, object]:
    data: dict[str, object] = {"type": block.type}
    for name in _BLOCK_FIELDS[block.type]:
        value = getattr(block, name)
        if name in ("items", "steps"):
            value = [text for text in map(_item_text, value) if text.strip()]
        elif name == "terms":
            value = _rows(value, ("term", "definition"))
        elif name == "questions":
            value = _rows(value, ("question", "answer"))
        if _filled(value):
            data[name] = value
    return data


def guide_to_dict(guide: WireGuide) -> dict[str, object]:
    """Keep only what each block type uses, leaving required-but-missing fields absent."""
    data: dict[str, object] = {
        "title": guide.title,
        "sections": [
            {"heading": section.heading, "blocks": [_block_dict(b) for b in section.blocks]}
            for section in guide.sections
        ],
    }
    if _filled(guide.subtitle):
        data["subtitle"] = guide.subtitle
    return data


def to_study_guide(guide: WireGuide) -> StudyGuide:
    """Validate against the strict schema; raises pydantic.ValidationError naming bad fields."""
    return StudyGuide.model_validate(guide_to_dict(guide))
