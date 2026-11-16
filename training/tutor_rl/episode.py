"""Tutoring episodes: tutor and simulated student alternate, then the student is tested.

PROVISIONAL INTERFACE. Issue #46 says to build on the "Sprint 1 episode/model interfaces";
none exist in the repo yet. Replace `ChatModel`, `Problem`, and `Episode` with those once they
are confirmed, keeping `run_episodes` and `rewards.score_episodes` as the PedagogicalRL logic.

`run_episodes` advances many dialogues in lockstep so each model call is batched: the tutor
policy generates one turn for every open dialogue at once, and the student endpoint receives
its requests concurrently. `run_episode` is the single-dialogue special case.

Paper setup (section 5.1, appendix A.2): each dialogue starts from an empty history and only the
problem. Either the student opens with an attempt, or the tutor opens and elicits one; the
choice is random per problem and shared by every rollout of that problem. At most 16 messages.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from tutor_rl.config import EpisodeConfig
from tutor_rl.prompts import (
    END_TOKEN,
    STUDENT_FINAL,
    STUDENT_PRE_DIALOG,
    STUDENT_SYSTEM,
    TUTOR_SYSTEM,
)

Role = Literal["system", "user", "assistant"]
Speaker = Literal["tutor", "student"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class ChatModel(Protocol):
    """Anything that turns a chat history into one reply (vLLM, OpenAI-compatible API, fake).

    Models may also define `generate_batch(conversations, *, n=1) -> list[list[str]]` to answer
    several conversations in one call; `generate_many` falls back to looping `generate`.
    A tutor may define `was_truncated(messages, text) -> bool` (hit the token limit).
    """

    def generate(self, messages: Sequence[Message], *, n: int = 1) -> list[str]: ...


def generate_many(
    model: ChatModel, conversations: Sequence[Sequence[Message]], *, n: int = 1
) -> list[list[str]]:
    if not conversations:
        return []
    batch = getattr(model, "generate_batch", None)
    if batch is not None:
        return batch(conversations, n=n)
    return [model.generate(c, n=n) for c in conversations]


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    answer: str  # ground-truth final answer as written in the dataset (LaTeX allowed)
    source: str = ""  # Big-Math sub-source, e.g. "cn_k12"
    solve_rate: float | None = None  # dataset's Llama-3.1-8B solve rate (64 samples)


@dataclass
class Turn:
    speaker: Speaker
    content: str


@dataclass
class TutorStep:
    """One tutor generation: what the tutor saw and exactly what it wrote.

    `context_turns` is how many of `Episode.turns` preceded the generation, so the trainer can
    rebuild the prompt with `Episode.tutor_view(context_turns)`. `output` is the raw text,
    including the end token, which is what the policy is optimized on. `truncated` means the
    turn hit the token limit without a stop token (appendix B's length penalty).
    """

    context_turns: int
    output: str
    truncated: bool = False


@dataclass
class Episode:
    problem: Problem
    initiator: Speaker = "student"
    turns: list[Turn] = field(default_factory=list)
    ended_by_tutor: bool = False
    final_attempts: list[str] = field(default_factory=list)
    tutor_steps: list[TutorStep] = field(default_factory=list)

    def tutor_view(self, context_turns: int | None = None) -> list[Message]:
        """The tutor sees itself as the assistant; this is what the trainer optimizes."""
        turns = self.turns if context_turns is None else self.turns[:context_turns]
        msgs = [Message("system", TUTOR_SYSTEM.format(problem=self.problem.question))]
        msgs += [Message("assistant" if t.speaker == "tutor" else "user", t.content) for t in turns]
        return msgs

    def student_view(self) -> list[Message]:
        msgs = [Message("system", STUDENT_SYSTEM.format(problem=self.problem.question))]
        msgs += [
            Message("assistant" if t.speaker == "student" else "user", t.content)
            for t in self.turns
        ]
        return msgs

    def final_prompt(self) -> list[Message]:
        return self.student_view() + [Message("user", STUDENT_FINAL)]

    def transcript(self) -> str:
        """The judges' format, as in the paper's judge examples ("- Student: ...")."""
        names = {"tutor": "Teacher", "student": "Student"}
        return "\n".join(f"- {names[t.speaker]}: {t.content}" for t in self.turns)

    def to_dict(self) -> dict:
        return {
            "problem": vars(self.problem),
            "initiator": self.initiator,
            "turns": [vars(t) for t in self.turns],
            "ended_by_tutor": self.ended_by_tutor,
            "final_attempts": self.final_attempts,
            "tutor_steps": [vars(s) for s in self.tutor_steps],
        }


def pre_dialog_prompt(problem: Problem) -> list[Message]:
    """The student's chain-of-thought attempt before any dialogue (paper Figure 9)."""
    return [Message("user", STUDENT_PRE_DIALOG.format(problem=problem.question))]


def pick_initiators(n_problems: int, cfg: EpisodeConfig, rng: random.Random) -> list[Speaker]:
    """One opener per problem: fixed by the config, or a fair coin flip for "random"."""
    if cfg.initiator in ("student", "tutor"):
        return [cfg.initiator] * n_problems
    if cfg.initiator != "random":
        raise ValueError(f"episode.initiator must be random, student or tutor: {cfg.initiator}")
    return [rng.choice(("student", "tutor")) for _ in range(n_problems)]


def run_episodes(
    problems: Sequence[Problem],
    tutor: ChatModel,
    student: ChatModel,
    cfg: EpisodeConfig,
    *,
    initiators: Sequence[Speaker] | None = None,
    final_student: ChatModel | None = None,
) -> list[Episode]:
    """Roll out one dialogue per problem, in lockstep.

    `initiators` gives who opens each dialogue (default: the student, unless the config fixes
    it). `final_student` answers the post-dialog test (default: `student`); it exists so the
    final solutions can have a larger token budget than the chat turns.
    """
    if initiators is None:
        initiators = [cfg.initiator if cfg.initiator == "tutor" else "student"] * len(problems)
    episodes = [Episode(p, initiator=who) for p, who in zip(problems, initiators, strict=True)]
    open_eps = list(episodes)
    truncated = getattr(tutor, "was_truncated", None)
    for turn_index in range(cfg.max_turns):
        if not open_eps:
            break
        # Each dialogue alternates from its own opener; group this turn's speakers.
        for speaker in ("student", "tutor"):
            due = [ep for ep in open_eps if _speaker_at(ep, turn_index) == speaker]
            if not due:
                continue
            if speaker == "student":
                replies = generate_many(student, [ep.student_view() for ep in due])
                for ep, reply in zip(due, replies, strict=True):
                    ep.turns.append(Turn("student", reply[0]))
                continue
            views = [ep.tutor_view() for ep in due]
            replies = generate_many(tutor, views)
            for ep, view, reply in zip(due, views, replies, strict=True):
                raw = reply[0]
                cut = bool(truncated(view, raw)) if truncated else False
                ep.tutor_steps.append(TutorStep(len(ep.turns), raw, cut))
                if END_TOKEN in raw:
                    content = raw.replace(END_TOKEN, "").strip()
                    if content:
                        ep.turns.append(Turn("tutor", content))
                    ep.ended_by_tutor = True
                else:
                    ep.turns.append(Turn("tutor", raw))
        open_eps = [ep for ep in open_eps if not ep.ended_by_tutor]

    # Post-dialog test: the student solves alone, with the conversation in context.
    finals = generate_many(
        final_student or student,
        [ep.final_prompt() for ep in episodes],
        n=cfg.student_final_attempts,
    )
    for ep, attempts in zip(episodes, finals, strict=True):
        ep.final_attempts = list(attempts)
    return episodes


def _speaker_at(episode: Episode, turn_index: int) -> Speaker:
    other: Speaker = "tutor" if episode.initiator == "student" else "student"
    return episode.initiator if turn_index % 2 == 0 else other


def run_episode(
    problem: Problem, tutor: ChatModel, student: ChatModel, cfg: EpisodeConfig
) -> Episode:
    return run_episodes([problem], tutor, student, cfg)[0]
