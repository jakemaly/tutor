"""Typed run configuration loaded from a TOML file in training/configs/."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpisodeConfig:
    max_turns: int = 8
    student_final_attempts: int = 2


@dataclass(frozen=True)
class RewardConfig:
    leak_penalty: float = 1.0
    end_on_rejection: bool = True


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

    @property
    def output_dir(self) -> Path:
        return Path(self.train["output_dir"])


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    return RunConfig(
        path=path,
        run_name=raw["run_name"],
        seed=raw.get("seed", 0),
        episode=EpisodeConfig(**raw.get("episode", {})),
        reward=RewardConfig(**raw.get("reward", {})),
        tutor=raw["tutor"],
        student=raw["student"],
        judge=raw["judge"],
        train=raw["train"],
        data=raw["data"],
        eval=raw.get("eval", {}),
        text=text,
    )
