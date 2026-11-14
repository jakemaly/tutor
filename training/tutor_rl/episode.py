"""One tutoring episode: tutor and simulated student alternate, then the student is tested.

PROVISIONAL INTERFACE. Issue #46 says to build on the "Sprint 1 episode/model interfaces";
none exist in the repo yet. Replace `ChatModel`, `Problem`, and `Episode` with those once they
are confirmed, keeping `run_episode` and `rewards.score_episode` as the PedagogicalRL logic.
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
    """Anything that turns a chat history into one reply (vLLM, OpenAI-compatible API, fake)."""

    def generate(self, messages: Sequence[Message], *, n: int = 1) -> list[str]: ...


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
class Episode:
    problem: Problem
    turns: list[Turn] = field(default_factory=list)
    ended_by_tutor: bool = False
    final_attempts: list[str] = field(default_factory=list)

    def tutor_view(self) -> list[Message]:
        """The tutor sees itself as the assistant; this is what the trainer optimizes."""
        msgs = [Message("system", TUTOR_SYSTEM.format(problem=self.problem.question))]
        msgs += [
            Message("assistant" if t.speaker == "tutor" else "user", t.content) for t in self.turns
        ]
        return msgs

    def student_view(self) -> list[Message]:
        msgs = [Message("system", STUDENT_SYSTEM.format(problem=self.problem.question))]
        msgs += [
            Message("assistant" if t.speaker == "student" else "user", t.content)
            for t in self.turns
        ]
        return msgs

    def transcript(self) -> str:
        return "\n".join(f"- {t.speaker.capitalize()}: {t.content}" for t in self.turns)


def run_episode(
    problem: Problem, tutor: ChatModel, student: ChatModel, cfg: EpisodeConfig
) -> Episode:
    """Roll out one dialogue. The student opens, as in PedagogicalRL's classroom."""
    ep = Episode(problem)
    speaker: Literal["tutor", "student"] = "student"
    while len(ep.turns) < cfg.max_turns:
        if speaker == "student":
            reply = student.generate(ep.student_view())[0]
        else:
            reply = tutor.generate(ep.tutor_view())[0]
            if END_TOKEN in reply:
                reply = reply.replace(END_TOKEN, "").strip()
                if reply:
                    ep.turns.append(Turn("tutor", reply))
                ep.ended_by_tutor = True
                break
        ep.turns.append(Turn(speaker, reply))
        speaker = "tutor" if speaker == "student" else "student"

    # Post-dialogue test: the student solves alone, with the conversation in context.
    final_prompt = ep.student_view() + [
        Message("user", STUDENT_FINAL.format(problem=problem.question))
    ]
    ep.final_attempts = student.generate(final_prompt, n=cfg.student_final_attempts)
    return ep
