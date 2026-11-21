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
  flags in the options; provide the correct option IDs separately.
- For one item, call `present_study_element`. For a requested set of two or more, call
  `present_study_elements` once with all items to create a grouped, navigable study deck.
- Provide one correct ID for `mcq` and `true_false`, and every correct ID for
  `multi_select`. Add a concise explanation when useful. Flashcards use their sole
  option as the answer and do not need correct IDs.
- The UI grades submitted choices inline. Do not expect another learner turn or send a
  follow-up chat response just to mark an answer correct or incorrect.
