"""One GPU, one memory budget: everything together stays under `[gpu] max_fraction` (80%).

The student and judge are vLLM servers, each holding a fixed `gpu_memory_utilization` share. The
tutor runs in this process and gets whatever the servers leave under the target, enforced as a
hard PyTorch cap (`torch.cuda.set_per_process_memory_fraction`). The rest of the GPU stays free,
so spikes (CUDA graphs, kernel workspaces, fragmentation) do not end in out-of-memory errors.

    student + judge + tutor = max_fraction
"""

from dataclasses import dataclass

DEFAULT_MAX_FRACTION = 0.80
DEFAULT_MIN_TUTOR = 0.24  # Qwen3.5-9B bf16 weights alone are ~18.5 GB (~19% of 96 GB)


@dataclass(frozen=True)
class GPUBudget:
    max_fraction: float
    servers: dict[str, float]  # vLLM server name -> gpu_memory_utilization
    tutor: float  # this process's cap

    def summary(self) -> str:
        parts = [f"{name} {f:.0%}" for name, f in self.servers.items()]
        return ", ".join([*parts, f"tutor {self.tutor:.0%}"]) + f" = {self.max_fraction:.0%}"


def plan(cfg) -> GPUBudget:
    """The servers' shares plus the tutor's cap, checked to fit under the target."""
    gpu = getattr(cfg, "gpu", {}) or {}
    target = float(gpu.get("max_fraction", DEFAULT_MAX_FRACTION))
    minimum = float(gpu.get("min_tutor_fraction", DEFAULT_MIN_TUTOR))
    servers = {"student": float(cfg.student["gpu_memory_utilization"])}
    if not cfg.judge.get("share_student", False):
        servers["judge"] = float(cfg.judge["gpu_memory_utilization"])
    tutor = round(target - sum(servers.values()), 4)
    if not 0 < target <= 0.95:
        raise ValueError(f"[gpu] max_fraction must be in (0, 0.95], got {target}")
    if tutor < minimum:
        raise ValueError(
            f"GPU budget does not fit: the vLLM servers take {sum(servers.values()):.0%} "
            f"({servers}) of the {target:.0%} target, leaving the tutor {tutor:.0%}; it needs at "
            f"least {minimum:.0%}. Lower a server's gpu_memory_utilization or raise "
            "[gpu] max_fraction."
        )
    return GPUBudget(target, servers, tutor)


def cap_tutor(budget: GPUBudget, device: int = 0) -> None:
    """Limit this process's PyTorch allocations to the tutor's share (call before loading it)."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(budget.tutor, device)
