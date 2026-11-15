"""Tutoring episodes: tutor and simulated student alternate, then the student is tested.

PROVISIONAL INTERFACE. Issue #46 says to build on the "Sprint 1 episode/model interfaces";
none exist in the repo yet. Replace `ChatModel`, `Problem`, and `Episode` with those once they
are confirmed, keeping `run_episodes` and `rewards.score_episodes` as the PedagogicalRL logic.

`run_episodes` advances many dialogues in lockstep so each model call is batched: the tutor
policy generates one turn for every open dialogue at once, and the student endpoint receives
its requests concurrently. `run_episode` is the single-dialogue special case.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from tutor_rl.config import EpisodeConfig
from tutor_rl.prompts import END_TOKEN, STUDENT_FINAL, STUDENT_SYSTEM, TUTOR_SYSTEM

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class ChatModel(Protocol):
    """Anything that turns a chat history into one reply (vLLM, OpenAI-compatible API, fake).

    Models may also define `generate_batch(conversations, *, n=1) -> list[list[str]]` to answer
    several conversations in one call; `generate_many` falls back to looping `generate`.
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
    answer: str  # normalized final answer, e.g. "72"


@dataclass
class Turn:
    speaker: Literal["tutor", "student"]
    content: str


@dataclass
class TutorStep:
    """One tutor generation: what the tutor saw and exactly what it wrote.

    `context_turns` is how many of `Episode.turns` preceded the generation, so the trainer can
    rebuild the prompt with `Episode.tutor_view(context_turns)`. `output` is the raw text,
    including the end token, which is what the policy is optimized on.
    """

    context_turns: int
    output: str


@dataclass
class Episode:
    problem: Problem
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
        return self.student_view() + [
            Message("user", STUDENT_FINAL.format(problem=self.problem.question))
        ]

    def transcript(self) -> str:
        return "\n".join(f"- {t.speaker.capitalize()}: {t.content}" for t in self.turns)

    def to_dict(self) -> dict:
        return {
            "problem": vars(self.problem),
            "turns": [vars(t) for t in self.turns],
            "ended_by_tutor": self.ended_by_tutor,
            "final_attempts": self.final_attempts,
            "tutor_steps": [vars(s) for s in self.tutor_steps],
        }


def run_episodes(
    problems: Sequence[Problem], tutor: ChatModel, student: ChatModel, cfg: EpisodeConfig
) -> list[Episode]:
    """Roll out one dialogue per problem, in lockstep. The student opens, as in PedagogicalRL."""
    episodes = [Episode(p) for p in problems]
    open_eps = list(episodes)
    speaker: Literal["tutor", "student"] = "student"
    for _ in range(cfg.max_turns):
        if not open_eps:
            break
        if speaker == "student":
            replies = generate_many(student, [ep.student_view() for ep in open_eps])
            for ep, reply in zip(open_eps, replies, strict=True):
                ep.turns.append(Turn("student", reply[0]))
        else:
            replies = generate_many(tutor, [ep.tutor_view() for ep in open_eps])
            still_open = []
            for ep, reply in zip(open_eps, replies, strict=True):
                raw = reply[0]
                ep.tutor_steps.append(TutorStep(len(ep.turns), raw))
                if END_TOKEN in raw:
                    content = raw.replace(END_TOKEN, "").strip()
                    if content:
                        ep.turns.append(Turn("tutor", content))
                    ep.ended_by_tutor = True
                else:
                    ep.turns.append(Turn("tutor", raw))
                    still_open.append(ep)
            open_eps = still_open
        speaker = "tutor" if speaker == "student" else "student"

    # Post-dialogue test: the student solves alone, with the conversation in context.
    finals = generate_many(
        student, [ep.final_prompt() for ep in episodes], n=cfg.student_final_attempts
    )
    for ep, attempts in zip(episodes, finals, strict=True):
        ep.final_attempts = list(attempts)
    return episodes


def run_episode(
    problem: Problem, tutor: ChatModel, student: ChatModel, cfg: EpisodeConfig
) -> Episode:
    return run_episodes([problem], tutor, student, cfg)[0]
