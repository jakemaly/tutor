"""Discover local skills, preload defaults, and expose the rest through one tool."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.tools import ToolException, tool

SKILLS_DIR = Path(__file__).with_name("skills")
BASE_PROMPT = "You are a helpful, clear assistant. Answer questions directly and accurately."


@dataclass(frozen=True)
class Skill:
    description: str
    instructions: str
    preload: bool = False


def discover_skills(directory: Path = SKILLS_DIR) -> dict[str, Skill]:
    """Read trusted skill files once at startup; names are the lookup allowlist."""
    skills = {}
    for path in sorted(directory.glob("*/SKILL.md")):
        try:
            _, frontmatter, body = path.read_text(encoding="utf-8").split("---", 2)
            metadata = yaml.safe_load(frontmatter)
            name, description = metadata["name"], metadata["description"]
            preload = metadata.get("preload", False)
            if (
                name != path.parent.name
                or not isinstance(description, str)
                or not description.strip()
                or type(preload) is not bool
                or not body.strip()
            ):
                raise ValueError(
                    "Use the folder name, nonempty description/body, and boolean preload"
                )
            skills[name] = Skill(description, body.strip(), preload)
        except (ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as error:
            raise ValueError(f"Invalid skill {path}: {error}") from error
    return skills


def build_system_prompt(skills: dict[str, Skill] | None = None) -> str:
    """Expose every description, but only preload bodies with preload: true."""
    skills = discover_skills() if skills is None else skills
    sections = [BASE_PROMPT, "Available skills:"]
    sections.extend(
        f"- {name} ({'preloaded' if skill.preload else 'invocation only'}): {skill.description}"
        for name, skill in skills.items()
    )
    sections.append(
        "Use the catalog to select skills relevant to the user's request. Before using an "
        "invocation-only skill, call load_skill(name) to read its instructions. Follow loaded "
        "skills when relevant. Do not reload instructions already present in this context. "
        "If no skill applies, answer directly."
    )
    sections.extend(
        f'<loaded_skill name="{name}">\n{skill.instructions}\n</loaded_skill>'
        for name, skill in skills.items()
        if skill.preload
    )
    return "\n\n".join(sections)


def make_load_skill(skills: dict[str, Skill]):
    """Keep filesystem paths out of model-supplied tool arguments."""

    @tool
    def load_skill(name: str) -> str:
        """Load a skill's full instructions by its exact name from the available skills catalog."""
        if name not in skills:
            raise ToolException(f"Unknown skill {name!r}. Available skills: {', '.join(skills)}")
        return f'<loaded_skill name="{name}">\n{skills[name].instructions}\n</loaded_skill>'

    return load_skill


if __name__ == "__main__":
    print(build_system_prompt())
