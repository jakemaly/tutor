"""Skill discovery, preload policy, and bounded catalog lookup."""

from pathlib import Path

import pytest
from langchain_core.tools import ToolException

from app.graph import SYSTEM_PROMPT
from app.skills import SKILLS_DIR, build_system_prompt, discover_skills, make_load_skill


def test_default_skill_is_preloaded_but_second_skill_is_catalog_only() -> None:
    skills = discover_skills()
    assert skills["socratic-questioning"].instructions in SYSTEM_PROMPT
    assert "worked-example (invocation only):" in SYSTEM_PROMPT
    assert skills["worked-example"].instructions not in SYSTEM_PROMPT
    assert skills["worked-example"].instructions in make_load_skill(skills).invoke(
        {"name": "worked-example"}
    )


def test_boolean_switch_controls_whether_new_skill_body_is_preloaded(tmp_path: Path) -> None:
    folder = tmp_path / "example"
    folder.mkdir()
    path = folder / "SKILL.md"
    content = (
        "---\nname: example\ndescription: Use for worked examples.\npreload: false\n---\n"
        "Unique instructions for this example."
    )
    path.write_text(content)
    catalog = build_system_prompt(discover_skills(tmp_path))
    assert "example (invocation only): Use for worked examples." in catalog
    assert "Unique instructions" not in catalog

    path.write_text(content.replace("preload: false", "preload: true"))
    assert "Unique instructions" in build_system_prompt(discover_skills(tmp_path))


def test_omitted_preload_defaults_to_invocation_only(tmp_path: Path) -> None:
    folder = tmp_path / "example"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: example\ndescription: An example.\n---\nInstructions."
    )
    assert not discover_skills(tmp_path)["example"].preload


@pytest.mark.parametrize("value", ['"false"', "1", "null"])
def test_preload_requires_a_yaml_boolean(tmp_path: Path, value: str) -> None:
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        f"---\nname: broken\ndescription: Example.\npreload: {value}\n---\nInstructions."
    )
    with pytest.raises(ValueError, match="Invalid skill.*broken/SKILL.md"):
        discover_skills(tmp_path)


def test_malformed_skill_identifies_the_file(tmp_path: Path) -> None:
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "SKILL.md").write_text("Missing frontmatter")
    with pytest.raises(ValueError, match="Invalid skill.*broken/SKILL.md"):
        discover_skills(tmp_path)


@pytest.mark.parametrize("name", ["missing", "../config.py", str(SKILLS_DIR)])
def test_tool_only_loads_catalog_names(name: str) -> None:
    with pytest.raises(ToolException, match="Unknown skill.*Available skills"):
        make_load_skill(discover_skills()).invoke({"name": name})
