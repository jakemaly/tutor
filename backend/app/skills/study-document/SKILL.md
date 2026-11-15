---
name: study-document
description: Turn the material the user asks for into a downloadable study guide (PDF or Word). Use when they ask to export, save, print, or make notes, a study guide, a cheat sheet, or a handout as a PDF or DOCX.
preload: false
---
# Study document

Build a document from **what the user asked for**, never from the whole conversation.

## 1. Decide the scope
- Identify the topic or part of the chat the user means ("the limit proofs", "that last
  explanation", "everything about determinants"). Use only that material.
- If it is genuinely unclear which part they mean, or whether they want PDF or Word, ask one
  short question first. Otherwise choose PDF.
- Leave out greetings, tutoring back-and-forth, hints you gave as questions, and anything the
  student did not ask to keep. Do not paste the transcript, and do not write "User:" or
  "Assistant:" lines.
- Include practice questions only if they asked for them. Include answers only if they asked
  for an answer key or the worked solutions were already given in the chat.

## 2. Organize it as a study guide
Write a short title and 2-8 sections, each with a clear `heading` and a list of `blocks`. Every
block is one object with a `type`; set only the fields shown for that type:
- `{"type": "paragraph", "text": "..."}`: a brief explanation in your own concise words.
- `{"type": "key_terms", "terms": [{"term": "ε", "definition": "..."}]}`: term/definition pairs.
- `{"type": "bullets", "items": ["...", "..."]}` or `"type": "steps"` for an ordered method.
  Items are plain strings, never objects; use `key_terms` for term/definition pairs.
- `{"type": "worked_example", "problem": "...", "steps": ["..."], "answer": "..."}`.
- `{"type": "practice", "questions": [{"question": "...", "answer": "..."}]}`: the answer is
  optional; answers are collected into an answer key.
- `{"type": "callout", "kind": "Remember", "text": "..."}`: `kind` is `Remember`, `Tip`, or
  `Watch out`; one short note.

## 3. Write plain text only
No Markdown, HTML, LaTeX, or code to run. Use Unicode for math: ε δ → ≤ ≥ ∑ √ ∈ ℝ ² ₁. The
fixed templates handle fonts, layout, and page numbers. Keep it concise; there are hard limits
on sections, items, and total length, and an over-long request is rejected.

## 4. Call the tool
Call `create_study_document(format, guide)` once. Always pass both arguments:
- `format` is `"pdf"` or `"docx"`.
- `guide` is a JSON object (never a quoted string) with `title` and `sections`, built from the
  block shapes above.

If the tool returns an error, fix exactly what it names and call it again; do not apologize or
change the content. When it succeeds, the user sees a download card. Reply with one short
sentence saying it is ready, and do not repeat the contents.
