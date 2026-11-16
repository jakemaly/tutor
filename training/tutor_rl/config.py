"""Typed run configuration loaded from a TOML file in training/configs/."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EpisodeConfig:
    max_turns: int = 16  # total tutor + student messages (paper: 16)
    student_final_attempts: int = 8  # K post-dialog solutions for the solve rate (paper: K=8)
    # Who opens: "random" picks student- or tutor-initiated per problem, the same for every
    # rollout of that problem (paper, appendix A.2); "student" or "tutor" fixes it.
    initiator: str = "random"


@dataclass(frozen=True)
class RewardConfig:
    """r = r_sol + (r_ped - 1) * lam + r_templ; with hard=True a rejected dialogue gets -lam."""

    lam: float = 0.75
    hard: bool = False
    judges: tuple[str, ...] = ("leakage", "helpfulness")  # prompts in prompts.JUDGES
    samples_per_judge: int = 2  # paper: "sample twice from each"
    template: bool = True  # appendix B: end-of-conversation bonus and truncation penalty
    end_bonus: float = 0.1
    length_penalty: float = 0.5


@dataclass(frozen=True)
class RunConfig:
    """The whole file; sections without a dataclass yet are kept as plain dicts."""

    path: Path
    run_name: str
    seed: int
    episode: EpisodeConfig
    reward: RewardConfig
    tutor: dict
    student: dict
    judge: dict
    train: dict
    data: dict
    eval: dict
    text: str  # the TOML as written, saved next to every checkpoint
    eval_judge: dict = field(default_factory=dict)  # optional separate judge for evaluation

    @property
    def output_dir(self) -> Path:
        return Path(self.train["output_dir"])


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    reward = dict(raw.get("reward", {}))
    if "judges" in reward:
        reward["judges"] = tuple(reward["judges"])
    return RunConfig(
        path=path,
        run_name=raw["run_name"],
        seed=raw.get("seed", 0),
        episode=EpisodeConfig(**raw.get("episode", {})),
        reward=RewardConfig(**reward),
        tutor=raw["tutor"],
        student=raw["student"],
        judge=raw["judge"],
        train=raw["train"],
        data=raw["data"],
        eval=raw.get("eval", {}),
        eval_judge=raw.get("eval_judge", {}),
        text=text,
    )
