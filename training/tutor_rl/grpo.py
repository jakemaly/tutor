"""Multi-turn GRPO for the tutor (paper section 5.3 and appendix A.2).

TRL's GRPOTrainer scores one completion per prompt; here an episode has several tutor turns,
so the loss is written directly: every tutor turn of an episode is a (prompt, completion) sample
that shares the episode's group advantage, student and system tokens are never scored, no
discounting, and the loss is averaged over all tutor tokens in the batch.

Each batch of rollouts gets `num_iterations` optimizer steps (the paper's mu = 2). The first
pass is exactly on-policy (ratio 1); later passes use the PPO clipped ratio against the
log-probs recorded on the first pass. KL is the k3 estimator against the base model (adapter
disabled), computed once per batch.
"""

import json
import logging
import random
import shutil
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tutor_rl.config import RunConfig
from tutor_rl.episode import ChatModel, Episode, Problem, pick_initiators, run_episodes
from tutor_rl.policy import HFTutorPolicy, load_adapter_weights
from tutor_rl.rewards import group_advantages, score_dict, score_episodes, summarize


@dataclass
class Sample:
    prompt_ids: list[int]
    completion_ids: list[int]
    advantage: float
    old_logp: torch.Tensor | None = field(default=None, repr=False)  # filled on the first pass
    ref_logp: torch.Tensor | None = field(default=None, repr=False)


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


def make_logger(run_dir: Path) -> logging.Logger:
    """A per-run logger writing to logs/train.log (uploaded with every checkpoint)."""
    log = logging.getLogger(f"tutor_rl.{run_dir.resolve()}")
    log.setLevel(logging.INFO)
    log.propagate = False
    path = run_dir / "logs" / "train.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    target = str(path.resolve())
    for h in list(log.handlers):  # reopen if the file was moved aside (a fresh, non-resumed run)
        if getattr(h, "baseFilename", None) == target and not path.exists():
            log.removeHandler(h)
            h.close()
    if not any(getattr(h, "baseFilename", None) == target for h in log.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
    return log


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
        final_student: ChatModel | None = None,
    ) -> None:
        self.cfg, self.policy, self.student, self.judge = cfg, policy, student, judge
        self.final_student = final_student
        self.problems = train_problems
        self.run_dir = Path(run_dir)
        self.data_dir = Path(data_dir) if data_dir else self.run_dir / "data"
        t = cfg.train
        self.iterations = int(t.get("num_iterations", 1))
        self.params = [p for p in policy.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.params,
            lr=t["learning_rate"],
            betas=(t.get("adam_beta1", 0.9), t.get("adam_beta2", 0.999)),
            weight_decay=t.get("weight_decay", 0.0),
        )
        # A plain function (not a bound method), so the scheduler state saves and loads cleanly.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda update: self._lr_factor(update)
        )
        self.step_count = 0
        self.log = make_logger(self.run_dir)
        (self.run_dir / "rollouts").mkdir(parents=True, exist_ok=True)

    def _lr_factor(self, update: int) -> float:
        """Linear warmup, then constant or linear decay to 0 (TRL's default) over all updates."""
        t = self.cfg.train
        warmup = t.get("warmup_steps", 0) * self.iterations
        total = max(1, t["max_steps"] * self.iterations)
        if warmup and update < warmup:
            return (update + 1) / warmup
        if t.get("lr_scheduler", "linear") == "constant":
            return 1.0
        return max(0.0, (total - update) / max(1, total - warmup))

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
        unique = self.next_problems()
        rng = random.Random(self.cfg.seed * 1_000_003 + self.step_count)
        openers = pick_initiators(len(unique), self.cfg.episode, rng)
        problems = [p for p in unique for _ in range(group)]
        initiators = [who for who in openers for _ in range(group)]

        torch.manual_seed(self.cfg.seed + self.step_count)
        self.policy.records.clear()
        self.policy.use_adapter = True
        episodes = run_episodes(
            problems,
            self.policy,
            self.student,
            self.cfg.episode,
            initiators=initiators,
            final_student=self.final_student,
        )
        dialog_time = time.time() - start
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
            "zero_signal_groups": sum(
                len({s.reward for s in scores[g : g + group]}) == 1
                for g in range(0, len(scores), group)
            )
            / len(unique),
            "samples": len(samples),
            "tutor_tokens": sum(len(s.completion_ids) for s in samples),
            "lr": self.scheduler.get_last_lr()[0],
            "dialog_seconds": round(dialog_time, 1),
            "judge_seconds": round(rollout_time - dialog_time, 1),
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
        """`num_iterations` optimizer steps over the whole batch (one full-batch step each)."""
        model, beta = self.policy.model, self.cfg.train.get("beta", 0.0)
        eps = self.cfg.train.get("epsilon", 0.2)
        temperature = self.policy.cfg.get("temperature", 1.0)
        total_tokens = sum(len(s.completion_ids) for s in samples)
        model.train()
        out: dict = {}
        for it in range(self.iterations):
            pg_sum = kl_sum = clipped = 0.0
            for s in samples:
                if beta > 0 and s.ref_logp is None:
                    with torch.no_grad(), model.disable_adapter():
                        s.ref_logp = completion_logprobs(model, s, temperature)
                logp = completion_logprobs(model, s, temperature)
                if s.old_logp is None:  # first pass: the policy that sampled the batch
                    s.old_logp = logp.detach()
                ratio = torch.exp(logp - s.old_logp)
                unclipped = ratio * s.advantage
                clipped_obj = torch.clamp(ratio, 1 - eps, 1 + eps) * s.advantage
                pg = -torch.min(unclipped, clipped_obj).sum()
                clipped += (clipped_obj < unclipped).sum().item()
                loss = pg
                if beta > 0:
                    diff = s.ref_logp - logp
                    kl = (torch.exp(diff) - diff - 1).sum()
                    loss = loss + beta * kl
                    kl_sum += kl.item()
                (loss / total_tokens).backward()
                pg_sum += pg.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.train["max_grad_norm"])
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if it == 0:
                out.update(
                    pg_loss=pg_sum / total_tokens,
                    kl=kl_sum / total_tokens,
                    grad_norm=float(grad_norm),
                )
            else:
                out[f"clip_frac_iter{it + 1}"] = clipped / total_tokens
                out[f"grad_norm_iter{it + 1}"] = float(grad_norm)
        for s in samples:  # free GPU memory before the next rollout
            s.old_logp = s.ref_logp = None
        if torch.cuda.is_available():
            # Return the update's activation blocks so the next rollout's batched generation
            # (a different shape) does not fragment around them.
            torch.cuda.empty_cache()
        out["policy_updates"] = self.scheduler.last_epoch
        return out

    def _log(self, metrics, episodes, scores, advantages) -> None:
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        path = self.run_dir / "rollouts" / f"step-{self.step_count:05d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ep, sc, adv in zip(episodes, scores, advantages, strict=True):
                row = {**ep.to_dict(), **score_dict(sc), "advantage": adv, "step": self.step_count}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        m = metrics
        self.log.info(
            "step %d/%d reward %.3f solve %.3f accept %.3f leak %.3f helpful-reject %.3f "
            "turns %.1f ended %.2f trunc %.2f kl %.5f grad %.3f samples %d tokens %d "
            "dialog %.0fs judge %.0fs step %.0fs gpu %.1fGB",
            m["step"], self.cfg.train["max_steps"], m["reward"], m["solve_rate"],
            m["accept_rate"], m["leak_rate"], m.get("helpfulness_reject_rate", 0.0),
            m["mean_turns"], m["ended_by_tutor_rate"], m["truncated_rate"], m.get("kl", 0.0),
            m.get("grad_norm", 0.0), m["samples"], m["tutor_tokens"], m["dialog_seconds"],
            m["judge_seconds"], m["step_seconds"], m["max_gpu_gb"],
        )  # fmt: skip
        if m.get("skipped_update"):
            self.log.warning("step %d: every group scored the same, no update", m["step"])

    # ----- loop and checkpoints ----------------------------------------------------------
    def train(
        self,
        on_step: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_save: Callable[[Path], None] | None = None,
    ) -> Path:
        """Run to max_steps. `on_save(path)` runs after each checkpoint (e.g. a Hub upload)."""
        max_steps, every = self.cfg.train["max_steps"], self.cfg.train["save_steps"]
        self.log.info(
            "training %s from step %d to %d", self.cfg.run_name, self.step_count, max_steps
        )
        while self.step_count < max_steps:
            try:
                metrics = self.step()
            except Exception:
                self.log.exception("step %d failed", self.step_count + 1)
                raise
            if on_step:
                on_step(metrics)
            if self.step_count % every == 0 or self.step_count == max_steps:
                path = self.save(self.checkpoint_path())
                self.log.info("saved %s", path.relative_to(self.run_dir))
                if on_save:
                    on_save(path)
            if should_stop and should_stop():
                break
        final = self.save(self.run_dir / "final", with_optimizer=False)
        self.log.info("saved final adapter")
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
        self.log.info("resumed from %s at step %d", checkpoint, self.step_count)
        return self.step_count


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted((Path(run_dir) / "checkpoints").glob("step-*/optimizer.pt"))
    return ckpts[-1].parent if ckpts else None
