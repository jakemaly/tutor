# Authoring tutor skills

Skills live in `backend/app/skills/<name>/SKILL.md`. They are trusted, developer-authored
instructions. All names and descriptions appear in the model's catalog; a per-file boolean
controls when the full instructions enter its context.

| Frontmatter | Full instructions | Example |
| --- | --- | --- |
| `preload: true` | In the system message from the start | `socratic-questioning` |
| `preload: false` (default) | Returned when the model calls `load_skill(name)` | `worked-example` |

“Invocation only” means the model may invoke the loader itself when the description matches;
users do not need to name the skill. It does not mean explicit user invocation is required.

## Add a skill

Create a folder and a UTF-8 `SKILL.md` with YAML frontmatter and Markdown instructions:

```markdown
---
name: worked-example
description: Give a complete step-by-step solution when a student explicitly requests one.
preload: false
---
# Worked example

Explain each step and why it follows. Check the result against the original problem.
```

The `name` must match the folder name. Use a short description that tells the model when
the skill applies. Both description and instructions must be nonempty. `preload` must be a
YAML boolean (`true` or `false`, without quotes). Omitting it means `false`.
`preload` is this app's frontmatter extension.

Restart the backend after editing skills. No Python registry or frontend change is needed.
The model endpoint must support streaming tool calls; tools are bound using the existing
OpenAI-compatible client, so this works with a managed API or a suitably configured GPU server.

## What reaches the model

```text
System message:
  Base assistant instructions
  Available skills:
    - socratic-questioning (preloaded): description
    - worked-example (invocation only): description
  Skill selection instructions
  <loaded_skill name="socratic-questioning">full instructions</loaded_skill>

User: Please give me a complete worked solution ...
Assistant tool call: load_skill(name="worked-example")
Tool result: <loaded_skill name="worked-example">full instructions</loaded_skill>
Assistant: Setup ... Steps ... Check ...
```

LangGraph routes requested calls to one `ToolNode`, then returns the tool result to the
model for its answer. The tool only accepts discovered catalog names; it does not read
arbitrary model-supplied paths or execute scripts. Unknown names return a recoverable tool
error. The model can answer directly when no new skill is needed.

The chat shows `Loading skill: <name>` followed by `Loaded skill: <name>`, or
`Skill not loaded: <name>` if the call fails or is interrupted. Only the name and load
status reach the browser; full instructions stay on the server.

Loaded instructions remain in model context during the current request. The chat retains
the visible trace, but the backend strips display-only skill calls/results from subsequent
requests and rebuilds text history. The model loads a skill again on a later request if
needed. Selection is a model judgment, not a deterministic guarantee.

## Inspect and test

From the repository root:

```sh
PYTHONPATH=backend uv run --project backend python -m app.skills
npm test
```

The preview is the exact system message used by `graph.py`; invocation-only bodies are
absent. Tools are exposed separately via the request's `tools` field. Tests cover the boolean
switch, catalog lookup, invalid names, and a streamed tool call through the full FastAPI →
LangGraph → tool → model → frontend response path.
