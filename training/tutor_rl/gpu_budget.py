"""One GPU, one memory budget: everything together stays under `[gpu] max_fraction` (80%).

The student, judge, and (during evaluation) the Gemma leak judge are vLLM servers, each holding
a fixed `gpu_memory_utilization` share. The tutor is trained in this process; it gets whatever
the servers leave under the target, enforced as a hard PyTorch cap
(`torch.cuda.set_per_process_memory_fraction`). The rest of the GPU stays free as headroom, so
spikes (CUDA graphs, kernels' workspaces, fragmentation) do not end in out-of-memory errors.

    training:    student + judge              + tutor = max_fraction
    evaluation:  student + judge + eval_judge + tutor = max_fraction
"""

from dataclasses import dataclass

DEFAULT_MAX_FRACTION = 0.80
DEFAULT_MIN_TUTOR = 0.24  # Qwen3.5-9B bf16 weights alone are ~18.5 GB (~19% of 96 GB)


@dataclass(frozen=True)
class GPUBudget:
    max_fraction: float
    servers: dict[str, float]  # vLLM server name -> gpu_memory_utilization
    tutor: float  # this process's cap

    def markdown(self, total_gib: float | None = None) -> str:
        def size(f: float) -> str:
            return f"{f:.0%}" + (f" ({f * total_gib:.1f} GiB)" if total_gib else "")

        rows = [f"| {name} (vLLM) | {size(f)} |" for name, f in self.servers.items()]
        rows.append(f"| tutor (PyTorch cap) | {size(self.tutor)} |")
        rows.append(f"| **total** | **{size(self.max_fraction)}** |")
        rows.append(f"| left free | {size(1 - self.max_fraction)} |")
        return "| part | GPU share |\n| --- | ---: |\n" + "\n".join(rows)


def plan(cfg, *, evaluating: bool = False) -> GPUBudget:
    """The budget for training, or for evaluation (adds the eval judge if one is configured)."""
    gpu = getattr(cfg, "gpu", {}) or {}
    target = float(gpu.get("max_fraction", DEFAULT_MAX_FRACTION))
    minimum = float(gpu.get("min_tutor_fraction", DEFAULT_MIN_TUTOR))
    servers = {"student": float(cfg.student["gpu_memory_utilization"])}
    if not cfg.judge.get("share_student", False):
        servers["judge"] = float(cfg.judge["gpu_memory_utilization"])
    if evaluating and cfg.eval_judge:
        servers["eval_judge"] = float(cfg.eval_judge["gpu_memory_utilization"])
    tutor = round(target - sum(servers.values()), 4)
    if not 0 < target <= 0.95:
        raise ValueError(f"[gpu] max_fraction must be in (0, 0.95], got {target}")
    if tutor < minimum:
        phase = "evaluation" if evaluating else "training"
        raise ValueError(
            f"GPU budget for {phase} does not fit: the vLLM servers take "
            f"{sum(servers.values()):.0%} ({servers}) of the {target:.0%} target, leaving the "
            f"tutor {tutor:.0%}; it needs at least {minimum:.0%}. Lower a server's "
            "gpu_memory_utilization or raise [gpu] max_fraction."
        )
    return GPUBudget(target, servers, tutor)


def cap_tutor(budget: GPUBudget, device: int = 0) -> None:
    """Limit this process's PyTorch allocations to the tutor's share (call before loading it).

    Lowering the cap later (for evaluation) is safe as long as memory in use fits: cached but
    unused blocks are released first.
    """
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(budget.tutor, device)
