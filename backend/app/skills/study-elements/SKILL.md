---
name: study-elements
description: Create interactive flashcards and practice questions using the native study-element tool.
preload: false
---
# Study elements

When a learner asks for flashcards or practice questions, call `present_study_element`
instead of writing a plain-text imitation of the interactive UI.

- Use `flashcard` for one prompt and one answer. Put the prompt in `question` and the
  answer in the sole option's `label`.
- Use `mcq` for a single-answer multiple-choice question, `true_false` for a binary
  question, and `multi_select` when more than one choice may be correct.
- Give each option a short, unique `id` and a concise `label`. Do not include correctness
  flags in the options.
- Present one item at a time unless the learner asks for a set; then call the tool once
  for each item.
- After a learner submits choices, respond to the question and selected option IDs in
  their turn. Explain the reasoning without assuming the UI has already graded it.
