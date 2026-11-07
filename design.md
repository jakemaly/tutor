# TMI Tutor — design and styling

**Status:** baseline styling set, v1
**Reference:** [TMI Tutor Figma baseline](https://www.figma.com/design/x7LLfyTdIFuzYc9kp1lUWz/Math-Tutor-Chatbot-UI?node-id=31-211)
**Figma foundations:** [`TMI Foundations` page](https://www.figma.com/design/x7LLfyTdIFuzYc9kp1lUWz/Math-Tutor-Chatbot-UI?node-id=56-53) · `TMI Core` variable collection

This document records the decisions behind the current home and conversation screens. The familiar assistant-ui chat shell is the starting point; the product’s identity comes from how the tutor helps students think, not from replacing standard chat conventions with an unfamiliar interface.

## Design position

- Keep the conversation readable, low-friction, and familiar: a simple top bar, student messages on the right, tutor responses in the thread, and a persistent composer.
- Make the tutor’s teaching intent visible. Ask for a question, an attempted step, or a place the student is stuck; respond to that work and leave a clear next move.
- Keep the visual tone adult and calm. Avoid childlike characters, celebratory streaks, grades, fake expertise, or decorative “AI” motifs.
- Use visual structure when it clarifies reasoning. A diagram or generated work surface should explain the nearby step and remain optional, legible, and available as text.

## Answers to the design questions

### 1. Designing for university students

Students arrive with different levels of prior knowledge, time pressure, devices, and kinds of work. Let them start from a full question or a partial attempt, keep the interface usable for self-paced sessions, and preserve enough conversational context to return to a specific step. Support prose, equations, code, diagrams, and images without forcing every subject into a lesson template.

Use plain, respectful language and adult visual tone. Make keyboard use, visible focus, readable math alternatives, and clear message roles part of the baseline. Do not infer a permanent ability level from one error. Offer supports that the student can request, reduce, or dismiss.

### 2. Credibility and approachability

Credibility comes from transparent tutoring behavior and calibrated claims, not a lab coat, avatar, authority badge, or polished gradients. A compact “How I tutor” explanation can set expectations: the tutor will look at the student’s reasoning and begin with a small hint. Show course sources when the tutor uses them, state uncertainty where it matters, and make it easy to ask for a check or a more direct explanation.

Keep the basic chatbot hierarchy because students already know how to read it: student contribution on the right, tutor response on the left, composer at the bottom. Differentiate through the tutor’s response structure and visual language. In this baseline, Alike is reserved for the welcoming home heading; SF Pro or the platform UI sans-serif is used for controls and conversation text. The distinction adds warmth without making technical explanations harder to scan.

### 3. Value beyond a general chatbot

Use the interface to invite active work instead of presenting a broad blank prompt alone. The selected home screen’s suggested starts (“Check my reasoning,” “Explain a concept step by step,” “Quiz me on derivatives”) communicate useful study actions. Over time, make these adaptable to the active subject and context rather than generic task chips.

Useful product-specific controls to explore:

- **Help intent:** “Check my reasoning,” “Give me a hint,” “Explain the concept,” and “Show a worked example.” Keep these concrete, optional, and reversible.
- **Hint ladder:** let the student request a smaller nudge, another hint, or a direct explanation. Label the current help level; do not imply that one level is objectively right for everyone.
- **Study artifacts:** keep a pasted derivation, code sample, image, or course excerpt attached to the turn it concerns, so the tutor can refer to the exact line or step.
- **Generative UI:** render a graph, annotated equation, proof outline, code trace, or process diagram only when it helps explain the current step. Offer a compact/text alternative and allow the artifact to be collapsed.

### 4. Supporting the reasoning → nudge loop

Give the student’s attempt visual ownership, then have the tutor address one specific step and end with one answerable next question. A subtle **Your turn** cue can mark that handoff. Useful follow-ups include “Check this step,” “Smaller nudge,” and “Related example.” These make the learning loop easier to continue while preserving productive effort.

Avoid answer leakage by making answer reveal a deliberate choice, not the default state. Avoid correctness badges, mastery meters, or progress animations unless they reflect a real, validated assessment. In the interface, distinguish an observed issue in a step from a verified overall result.

### 5. Visuals across subjects and levels

Use a visual when it explains the same content as the nearby text: for example, a graph beside a calculus question, an argument map beside a writing question, or a process diagram beside a biology explanation. Label the relevant region and provide a text equivalent. Decorative motion, unrelated graphs, and duplicate representations add load without helping the student reason.

Novices may benefit from labeled, segmented, or annotated views. More experienced students may prefer a compact representation or may already know the scaffold; let them collapse it. Generate visuals based on the current subject and student work rather than showing a fixed dashboard of diagrams.

These are design hypotheses, not claims that a palette or font improves learning. They draw on evidence about university AI tutoring ([Kestin et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12179260/)), tutor transparency ([Bo et al., 2024](https://ojs.aaai.org/index.php/HCOMP/article/view/31597)), active learning ([Chi & Wylie, 2014](https://education.asu.edu/sites/default/files/lcl/chiwylie2014icap_2.pdf)), hint-based tutoring ([Bastani et al., 2025](https://www.pnas.org/doi/10.1073/pnas.2422633122)), useful feedback ([Hattie & Timperley, 2007](https://assess.ucr.edu/media/746/download#page=1)), multimedia learning ([Mayer, 2017](https://onlinelibrary.wiley.com/doi/10.1111/jcal.12197)), and expertise reversal ([Kalyuga, 2007](https://doi.org/10.1016/j.chb.2007.02.018)). Accessibility follows [WCAG 2.2](https://www.w3.org/TR/wcag/).

## Styling set

The app tokens live in [`frontend/src/design-tokens.css`](frontend/src/design-tokens.css). Figma mirrors them in `TMI Core` with CSS code syntax on each variable. Text styles use the names below. Keep the CSS and Figma values aligned when changing a token.

### Color

| Token | Value | Use |
|---|---:|---|
| `color/surface/base` | `#FFFFFF` | App and composer surface |
| `color/surface/student` | `#F1F1F1` | Neutral student message bubble |
| `color/surface/code` | `#F7F7F7` | Code block surface |
| `color/surface/hover` | `#FAFAFA` | Quiet hover surface |
| `color/text/primary` | `#1E1E1E` | Main text and controls |
| `color/text/secondary` | `#757575` | Placeholder, action, and metadata text |
| `color/border/default` | `#D6D6D6` | Composer and control outlines |
| `color/border/subtle` | `#E1E1E1` | Quiet outlines on the home state |
| `color/action/neutral` | `#A6A6A6` | Send control background; use a dark icon for contrast |
| `color/action/neutral-hover` | `#969696` | Send control hover |
| `color/focus` | `#006D77` | Keyboard focus outline |
| `color/brand/teal` | `#10A9BA` | TMI mark and restrained brand accents |
| `color/brand/coral` | `#FF5A3C` | TMI mark and restrained brand accents |
| `color/feedback/error-surface` | `#FFF7F4` | Error message surface |
| `color/feedback/error-border` | `#EAD2CC` | Error message outline |
| `color/feedback/error-text` | `#783D31` | Error message text |

Teal and coral identify the TMI mark; they are not tutor/student role colors or correctness signals. Do not use either as small text on white. The primary and secondary text colors meet WCAG AA contrast on the white surface; never use color alone to communicate a state.

### Typography

| Figma text style | Font | Size / line height | Use |
|---|---|---:|---|
| `TMI/Display/Home` | Alike Regular | 66 / 82.5 px | Home heading |
| `TMI/Body/Tutor` | SF Pro Regular | 19 / 29 px | Tutor response |
| `TMI/Body/Student` | SF Pro Regular | 16 / 22 px | Student message |
| `TMI/Composer/Placeholder` | SF Pro Regular | 22 / 31 px | Composer placeholder |
| `TMI/Control/Prompt` | SF Pro Regular | 18 / 25.2 px | Suggested prompt |
| `TMI/Control/Brand` | SF Pro Semibold | 19 / 25 px | App name |
| `TMI/Control/Model` | SF Pro Semibold | 18 / 25 px | Selected model |
| `TMI/Control/Secondary` | SF Pro Medium | 14 / 20 px | Secondary control labels |
| `TMI/Meta/ModelDetail` | SF Pro Regular | 16 / 22 px | Reasoning level or model detail |
| `TMI/Meta/Action` | SF Pro Regular | 14 / 20 px | Message actions |
| `TMI/Meta/Disclaimer` | SF Pro Regular | 12 / 16 px | Accuracy note |

The web app self-hosts Alike for the display face under the SIL Open Font License 1.1. The font’s copyright notice and full license are reproduced in the appendix below. Conversation UI uses the native system sans-serif stack, which resolves to SF Pro on Apple platforms, Segoe UI on Windows, and `system-ui` on Linux. Equations and code use a monospace face only where their content requires it.

### Layout and shape

| Token | Value |
|---|---:|
| `space/1` … `space/8` | 4, 8, 12, 16, 20, 24, 32, 48 px |
| `radius/prompt` | 15 px |
| `radius/control` | 14 px |
| `radius/new-chat` | 18 px |
| `radius/student` | 21 px |
| `radius/composer` | 24 px |
| `radius/code` | 10 px |
| `radius/error` | 11 px |
| `radius/full` | 999 px |
| `layout/topbar-height` | 62 px |
| `layout/content-width` | 1040 px |
| `layout/composer-height` | 124 px |
| `layout/student-message-max` | 410 px |

Keep the message column comfortable to read and the composer anchored below the active thread. At narrow widths, reduce the display heading and horizontal margins, allow the student bubble to use more of the screen, and preserve the same message order and focus behavior.

## Distinctive visual constants

1. **Editorial welcome, familiar conversation.** A single Alike heading gives the entry point its own voice; the chat remains a recognizable assistant-ui conversation.
2. **TMI color used as punctuation.** The teal/coral mark may appear as a small brand signifier or focus accent. Most of the interface stays neutral so the student’s work and the tutor’s reasoning remain the visual focus.
3. **Visible learning handoff.** Small labels such as `Hint 1` and `Your turn` can make the tutor’s move legible without turning the thread into a game or progress dashboard.
4. **Work-linked generative UI.** A graph or annotation belongs beside the exact reasoning step it explains, with a text equivalent and a way to collapse it.

Only the first two are visual styling decisions in the current baseline. The handoff and generated work artifacts are interaction patterns to prototype next; the current app does not yet implement them.

## Baseline fidelity and current implementation

The selected Figma pair is the source for the type hierarchy, whitespace, neutral student bubble, rounded composer, and home copy. The React app keeps its existing assistant-ui structure and API behavior. It adopts the shared styling tokens and distinct home/conversation composer prompts. The model picker and prompt suggestion components visible in Figma remain design references and are not implemented by this styling pass.

## Third-party font license

The Alike font is copyright 2009 The Alike Project Authors (https://github.com/cyrealtype/Alike), with Reserved Font Names “Alike” and “Alike Angular.” The following is the full license text distributed with the font.

```text
Copyright 2009 The Alike Project Authors (https://github.com/cyrealtype/Alike) with Reserved Font Name "Alike" and "Alike Angular".

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
https://scripts.sil.org/OFL


-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```
