"""The create_study_document tool: validated structure in, temporary artifact out.

Only a bounded backend capability: the model supplies plain data (see documents/wire.py), which
is validated against the strict StudyGuide, and fixed templates render it. No model-authored
code runs and file bytes never enter the model's context; it receives a short confirmation
while the browser receives the artifact metadata.
"""

from typing import Any, Literal

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, ConfigDict, ValidationError

from app.artifacts import ArtifactBackend
from app.documents import render_document
from app.documents.schema import StudyGuide
from app.documents.wire import WireGuide, to_study_guide

TOOL_NAME = "create_study_document"


class CreateStudyDocumentInput(BaseModel):
    """Arguments the model must supply; extra fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["pdf", "docx"]
    guide: WireGuide


def _outline(guide: StudyGuide) -> list[dict[str, Any]]:
    return [
        {"heading": section.heading, "blocks": len(section.blocks)} for section in guide.sections
    ]


MAX_REPORTED_ERRORS = 8


def _describe(error: ValidationError) -> str:
    """List the first few bad fields by path so the model can fix exactly those."""
    problems = error.errors()
    lines = []
    for problem in problems[:MAX_REPORTED_ERRORS]:
        path = ".".join(str(part) for part in problem["loc"])
        lines.append(f"{path}: {problem['msg']}" if path else problem["msg"])
    more = " (more problems omitted)" if len(problems) > MAX_REPORTED_ERRORS else ""
    return "; ".join(lines) + more + ". Paths count from 0."


def make_create_study_document(store: ArtifactBackend):
    """Bind the tool to one store; the browser fetches the file from /api/artifacts/{id}."""

    @tool(
        TOOL_NAME,
        args_schema=CreateStudyDocumentInput,
        response_format="content_and_artifact",
    )
    def create_study_document(format: str, guide: WireGuide) -> tuple[str, dict[str, Any]]:  # noqa: A002
        """Create a downloadable study guide (PDF or Word) from content the user asked for.

        Use only after loading the study-document skill. Both arguments are required: `format`
        is "pdf" or "docx", and `guide` is a JSON object (never a string) with `title` and
        `sections`. Put in `guide` just the material the user asked to turn into a document,
        organized into sections; never the whole chat.
        """
        try:
            study_guide = to_study_guide(WireGuide.model_validate(guide))
            rendered = render_document(study_guide, format)  # type: ignore[arg-type]
            artifact = store.put(rendered.filename, rendered.media_type, rendered.data)
        except ValidationError as error:
            raise ToolException(f"Could not create the document: {_describe(error)}") from error
        except ValueError as error:
            raise ToolException(f"Could not create the document: {error}") from error

        metadata = {
            "id": artifact.id,
            "url": f"/api/artifacts/{artifact.id}",
            "filename": artifact.filename,
            "format": format,
            "title": study_guide.title,
            "size": len(artifact.data),
            "expires_at": artifact.expires_at,
            "outline": _outline(study_guide),
        }
        confirmation = (
            f"Created {artifact.filename}. It is shown to the user as a download card. "
            "Reply with one short sentence; do not repeat the document's contents."
        )
        return confirmation, metadata

    return create_study_document
