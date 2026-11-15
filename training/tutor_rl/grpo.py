"""Multi-turn GRPO for the tutor (steps 1-5 of train.py's docstring).

TRL's GRPOTrainer scores one completion per prompt; here an episode has several tutor turns,
so the loss is written directly (as the authors did in eth-lre/PedagogicalRL src/grpo/trainer.py):
every tutor turn of an episode is a (prompt, completion) sample that shares the episode's group
advantage, student and system tokens are never scored, and the loss is averaged over all tutor
tokens in the step. Rollouts are on-policy with one update per batch, so the importance ratio is
1 and needs no clipping. KL is the k3 estimator against the base model (adapter disabled).
"""

import json
import random
import shutil
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from tutor_rl.config import RunConfig
from tutor_rl.episode import ChatModel, Episode, Problem, run_episodes
from tutor_rl.policy import HFTutorPolicy, load_adapter_weights
from tutor_rl.rewards import group_advantages, score_episodes, summarize


@dataclass
class Sample:
    prompt_ids: list[int]
    completion_ids: list[int]
    advantage: float


def build_samples(
    policy: HFTutorPolicy, episodes: list[Episode], advantages: list[float]
) -> list[Sample]:
    samples = []
    for ep, adv in zip(episodes, advantages, strict=True):
        if abs(adv) < 1e-8:  # no learning signal (the whole group scored the same)
            continue
        for step in ep.tutor_steps:
            prompt = policy.prompt_ids(ep.tutor_view(step.context_turns))
            completion = policy.completion_ids(prompt, step.output)
            if completion:
                samples.append(Sample(prompt, completion, adv))
    return samples


def completion_logprobs(model, sample: Sample, temperature: float) -> torch.Tensor:
    """Log-probability of each completion token; logits only for the completion positions."""
    device = next(model.parameters()).device
    ids = torch.tensor([sample.prompt_ids + sample.completion_ids], device=device)
    n = len(sample.completion_ids)
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=n + 1).logits[0, :-1]
    logits = logits.float() / temperature
    targets = torch.tensor(sample.completion_ids, device=device)
    return torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)


class GRPOTrainer:
    def __init__(
        self,
        cfg: RunConfig,
        policy: HFTutorPolicy,
        student: ChatModel,
        judge: ChatModel,
        train_problems: list[Problem],
        run_dir: Path,
        data_dir: Path | None = None,
    ) -> None:
        self.cfg, self.policy, self.student, self.judge = cfg, policy, student, judge
        self.problems = train_problems
        self.run_dir = Path(run_dir)
        self.data_dir = Path(data_dir) if data_dir else self.run_dir / "data"
        t = cfg.train
        self.params = [p for p in policy.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.params, lr=t["learning_rate"], betas=(0.9, 0.99), weight_decay=0.0
        )
        warmup = max(1, t.get("warmup_steps", 0))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: min(1.0, (s + 1) / warmup)
        )
        self.step_count = 0
        (self.run_dir / "rollouts").mkdir(parents=True, exist_ok=True)

    # ----- data --------------------------------------------------------------------------
    def next_problems(self) -> list[Problem]:
        """Deterministic: epoch-wise shuffles from the run seed, resumable from the step."""
        k = self.cfg.train["problems_per_step"]
        out = []
        for i in range(self.step_count * k, (self.step_count + 1) * k):
            epoch, pos = divmod(i, len(self.problems))
            order = list(range(len(self.problems)))
            random.Random(self.cfg.seed + epoch).shuffle(order)
            out.append(self.problems[order[pos]])
        return out

    # ----- one step ----------------------------------------------------------------------
    def step(self) -> dict:
        t, start = self.cfg.train, time.time()
        group = t["episodes_per_problem"]
        problems = [p for p in self.next_problems() for _ in range(group)]

        torch.manual_seed(self.cfg.seed + self.step_count)
        self.policy.records.clear()
        self.policy.use_adapter = True
        episodes = run_episodes(problems, self.policy, self.student, self.cfg.episode)
        scores = score_episodes(episodes, self.judge, self.cfg.reward)
        advantages = []
        for g in range(0, len(scores), group):
            advantages += group_advantages([s.reward for s in scores[g : g + group]])
        rollout_time = time.time() - start

        samples = build_samples(self.policy, episodes, advantages)
        update = self._update(samples) if samples else {"skipped_update": True}
        self.step_count += 1

        metrics = {
            "step": self.step_count,
            **summarize(episodes, scores),
            "reward_std_within_group": statistics.fmean(
                statistics.pstdev([s.reward for s in scores[g : g + group]])
                for g in range(0, len(scores), group)
            ),
            "samples": len(samples),
            "tutor_tokens": sum(len(s.completion_ids) for s in samples),
            "lr": self.scheduler.get_last_lr()[0],
            "rollout_seconds": round(rollout_time, 1),
            "step_seconds": round(time.time() - start, 1),
            "max_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1)
            if torch.cuda.is_available()
            else 0.0,
            **update,
        }
        self._log(metrics, episodes, scores, advantages)
        return metrics

    def _update(self, samples: list[Sample]) -> dict:
        model, beta = self.policy.model, self.cfg.train.get("beta", 0.0)
        temperature = self.policy.cfg.get("temperature", 1.0)
        total_tokens = sum(len(s.completion_ids) for s in samples)
        model.train()
        pg_sum = kl_sum = 0.0
        for s in samples:
            if beta > 0:
                with torch.no_grad(), model.disable_adapter():
                    ref = completion_logprobs(model, s, temperature)
            logp = completion_logprobs(model, s, temperature)
            ratio = torch.exp(logp - logp.detach())  # = 1, carries the policy gradient
            pg = -(ratio * s.advantage).sum()
            loss = pg
            if beta > 0:
                diff = ref - logp
                kl = (torch.exp(diff) - diff - 1).sum()
                loss = loss + beta * kl
                kl_sum += kl.item()
            (loss / total_tokens).backward()
            pg_sum += pg.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.train["max_grad_norm"])
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "pg_loss": pg_sum / total_tokens,
            "kl": kl_sum / total_tokens,
            "grad_norm": float(grad_norm),
        }

    def _log(self, metrics, episodes, scores, advantages) -> None:
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        path = self.run_dir / "rollouts" / f"step-{self.step_count:05d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ep, sc, adv in zip(episodes, scores, advantages, strict=True):
                row = {**ep.to_dict(), **vars(sc), "advantage": adv, "step": self.step_count}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ----- loop and checkpoints ----------------------------------------------------------
    def train(
        self,
        on_step: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_save: Callable[[Path], None] | None = None,
    ) -> Path:
        """Run to max_steps. `on_save(path)` runs after each checkpoint (e.g. a Hub upload)."""
        max_steps, every = self.cfg.train["max_steps"], self.cfg.train["save_steps"]
        while self.step_count < max_steps:
            metrics = self.step()
            if on_step:
                on_step(metrics)
            if self.step_count % every == 0 or self.step_count == max_steps:
                path = self.save(self.checkpoint_path())
                if on_save:
                    on_save(path)
            if should_stop and should_stop():
                break
        final = self.save(self.run_dir / "final", with_optimizer=False)
        if on_save:
            on_save(final)
        return final

    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoints" / f"step-{self.step_count:05d}"

    def save(self, path: Path, with_optimizer: bool = True) -> Path:
        """Adapter + config + data manifest + trainer state, so a checkpoint stands alone."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.policy.model.save_pretrained(path / "adapter")
        self.policy.tokenizer.save_pretrained(path / "adapter")
        (path / "config.toml").write_text(self.cfg.text, encoding="utf-8")
        manifest = self.data_dir / "manifest.json"
        if manifest.exists():
            shutil.copy(manifest, path / "data_manifest.json")
        state = {"step": self.step_count, "run_name": self.cfg.run_name}
        (path / "trainer_state.json").write_text(json.dumps(state, indent=2))
        if with_optimizer:
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                path / "optimizer.pt",
            )
        return path

    def resume(self, checkpoint: Path | None = None) -> int:
        """Load the newest (or given) checkpoint's adapter and optimizer; returns the step."""
        checkpoint = checkpoint or latest_checkpoint(self.run_dir)
        if checkpoint is None:
            return 0
        load_adapter_weights(self.policy, Path(checkpoint) / "adapter")
        opt = torch.load(Path(checkpoint) / "optimizer.pt", weights_only=False)
        self.optimizer.load_state_dict(opt["optimizer"])
        self.scheduler.load_state_dict(opt["scheduler"])
        self.step_count = json.loads((Path(checkpoint) / "trainer_state.json").read_text())["step"]
        # Steps logged after this checkpoint are redone, so drop their metrics rows.
        metrics = self.run_dir / "metrics.jsonl"
        if metrics.exists():
            rows = metrics.read_text(encoding="utf-8").splitlines()
            keep = [r for r in rows if r.strip() and json.loads(r)["step"] <= self.step_count]
            metrics.write_text("".join(r + "\n" for r in keep), encoding="utf-8")
        return self.step_count


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted((Path(run_dir) / "checkpoints").glob("step-*/optimizer.pt"))
    return ckpts[-1].parent if ckpts else None
