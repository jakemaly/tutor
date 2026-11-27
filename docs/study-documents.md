# Study documents (PDF / Word)

`create_study_document` turns *what the user asks to keep* into a downloadable study guide.
It is not a transcript exporter and not a document editor.

```text
User: "Make the limit-proof part a PDF"
 -> model calls load_skill("study-document"), picks the relevant material
 -> model calls create_study_document(format, guide)   # guide = plain data, no code
 -> backend converts it (app/study_documents/wire.py), validates it strictly
    (app/study_documents/schema.py), renders with a fixed template, stores the bytes briefly
    (app/artifacts.py)
 -> tool returns a one-line confirmation to the model; artifact metadata goes to the browser
 -> chat shows a card: outline, Preview (PDF only), Download -> GET /api/artifacts/{id}
```

Safety properties:

- The model supplies plain data; the server turns it into a strict `StudyGuide`; templates (`render_pdf.py`, `render_docx.py`)
  own every font, margin and color. No model-authored Python or markup is executed: text is
  escaped, control characters are stripped, unknown fields and block types are rejected.
- Hard limits: 12 sections, 12 blocks per section, 20 items per list, 40,000 characters total,
  5 MB per file.
- File bytes never pass through the model or the chat stream. The store is in memory, expires
  entries after 30 minutes and evicts oldest-first. Ids are unguessable capability URLs.
- Word files are always served as attachments; only PDFs can be previewed inline.

Two schemas, on purpose:

- `wire.py` is what the model is shown: one flat block shape (`type` plus a few plainly named
  fields) with no unions. Models and providers handle a seven-way union badly: they mix fields
  between block types, and a provider's error then lists every branch. The converter ignores
  fields a block type does not use and flattens object list items such as
  `{"term": ..., "definition": ...}` into plain strings.
- `schema.py` is the strict structure. Every type, length and size limit is enforced here, and
  an error names the exact field (for example `sections.2.blocks.0.bullets.items.1`) so the
  model can fix it.

Scope the model follows is defined in `backend/app/skills/study-document/SKILL.md`.

## Limits to know about

- **The store is per process.** With several backend instances or serverless functions (see
  `vercel.json`), a download can land on an instance that never saw the file and return 404.
  The app logs a warning at startup when it runs on Vercel with the default in-memory store.
  To fix it, implement `ArtifactBackend` (`put`/`get`, in `app/artifacts.py`) on shared storage
  with signed URLs and pass it as `create_app(artifacts=...)`. No other code changes.
- Both formats use a LaTeX-style font: Latin Modern Roman with Latin Modern Math's Greek letters
  and symbols merged in, bundled as renamed "Tutor Serif" files in `backend/app/study_documents/fonts`
  (renamed because the Latin Modern license requires it for modified copies; see
  `LICENSE-LatinModern.txt`). PDFs embed it, Word files carry it as an embedded font, and DejaVu
  Sans covers the odd character left. Math is plain Unicode; there is no LaTeX typesetting.
- There is no per-user quota or rate limit. All users share one store (50 files, 64 MB), so
  heavy use evicts older files early. Rate-limit `/api/chat` at your proxy or edge if the app
  is public.

## When the model gets the call wrong

Some providers (Groq, for example) check tool arguments against the JSON schema themselves and
abort the response with a `tool_use_failed` / "Tool call validation failed" error. Typical
causes are a missing `format` or `guide` sent as a string. LangGraph never sees that error, so
`app/graph.py` catches it, gives the model the provider's message once more (up to
`MAX_TOOL_CALL_RETRIES` times, not saved in the conversation) and asks it to call the tool again.
If every retry fails the chat shows "The model could not produce a valid document."

Other things to check when documents fail:

- **Rate limits.** Each step of the flow (load skill, create document, confirm) re-sends the
  system prompt and tool schemas, plus the generated guide. Small per-minute token limits (for
  example a free tier) can be exceeded. The real error is in the backend log under
  `Chat model request failed`.
- **Failed runs.** If a run ends early, any tool card that was already shown is closed as failed
  rather than left loading.
- **Expired files.** The card shows "expired" when the 30 minutes pass and, if the server no
  longer has a file, shows a message instead of an error page.
