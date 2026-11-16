"""Training and evaluation charts for the notebook (matplotlib, light surface).

Each training panel is one metric (small multiples, no legends needed): raw values as faint dots,
a moving average as the line. Colours: the validated categorical slots 1 (blue, ours/trained)
and 2 (orange, base); paper reference points are neutral grey.
"""

import math

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"
INK, INK_MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b66", "#e6e5e0", "#fcfcfb"

# (metric key, panel title, section) in reading order, four panels per row.
TRAINING_PANELS = [
    ("reward", "Reward r", "Reward"),
    ("solve_rate", "Post-dialog solve rate r_sol", "Reward"),
    ("accept_rate", "Judges accept (r_ped = 1)", "Reward"),
    ("template_reward", "Template reward r_templ", "Reward"),
    ("leakage_reject_rate", "Leakage judge rejects", "Pedagogy"),
    ("helpfulness_reject_rate", "Helpfulness judge rejects", "Pedagogy"),
    ("mean_tutor_words", "Words per tutor turn", "Pedagogy"),
    ("ended_by_tutor_rate", "Tutor ends the dialogue", "Pedagogy"),
    ("mean_turns", "Messages per dialogue", "Dialogue"),
    ("truncated_rate", "Dialogues with a cut-off turn", "Dialogue"),
    ("zero_signal_groups", "Groups with no signal (all equal)", "Dialogue"),
    ("reward_std_within_group", "Reward std within group", "Dialogue"),
    ("kl", "KL to base model", "Optimization"),
    ("grad_norm", "Grad norm (before clipping)", "Optimization"),
    ("clip_frac_iter2", "Clipped tokens, 2nd pass", "Optimization"),
    ("step_seconds", "Seconds per step", "Optimization"),
]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def moving_average(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1) : i + 1] if not math.isnan(v)]
        out.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return out


def training_dashboard(rows: list[dict], window: int = 10):
    """A 4x4 grid of training curves from metrics.jsonl rows."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(15, 10.5), facecolor=SURFACE)
    steps = [r["step"] for r in rows]
    for ax, (key, title, section) in zip(axes.flat, TRAINING_PANELS, strict=True):
        _style(ax)
        values = [float(r.get(key, math.nan)) for r in rows]
        ax.set_title(f"{section} · {title}", loc="left", fontsize=9, color=INK)
        if not rows or all(math.isnan(v) for v in values):
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center",
                    color=INK_MUTED, fontsize=8)  # fmt: skip
            continue
        ax.scatter(steps, values, s=10, color=BLUE, alpha=0.3, linewidths=0)
        smooth = moving_average(values, window)
        ax.plot(steps, smooth, color=BLUE, linewidth=2)
        last = next((v for v in reversed(smooth) if not math.isnan(v)), None)
        if last is not None:  # one direct label: the latest smoothed value
            label = f"{last:.0f}" if abs(last) >= 100 else f"{last:.3g}"
            ax.annotate(label, (steps[-1], last), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK, va="center")  # fmt: skip
    for ax in axes[-1]:
        ax.set_xlabel("step", fontsize=8, color=INK_MUTED)
    fig.suptitle(f"Training curves (dots: each step; line: {window}-step average)",
                 x=0.01, ha="left", fontsize=11, color=INK)  # fmt: skip
    fig.tight_layout()
    plt.close(fig)
    return fig


LABELLED_REFERENCE = {
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B-Instruct",
    "GPT-4o-2024-11-20",
    "LearnLM 2.0 Flash",
    "SocraticLM",
    "Qwen2.5-7B-RL λ=0.0",
    "Qwen2.5-7B-RL λ=0.75",
    "Qwen2.5-7B-RL-hard λ=1.0",
}


def eval_tradeoff(report: dict):
    """Paper Figure 1 style: delta solve rate vs. leaked solutions, paper models for reference."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    _style(ax)
    for name, (delta, leak) in report.get("paper_reference", {}).items():
        ax.scatter(leak, delta, s=40, color=GREY, alpha=0.7, linewidths=0)
        if name in LABELLED_REFERENCE:  # the rest stay unlabeled so labels never collide
            ax.annotate(name, (leak, delta), xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color=INK_MUTED)  # fmt: skip
    ours = report["results"]
    colors = {"base": ORANGE, "trained": BLUE}
    if "base" in ours and "trained" in ours:
        b, t = ours["base"], ours["trained"]
        ax.annotate("", xy=(t["leak_rate"], t["delta_solve_rate"]),
                    xytext=(b["leak_rate"], b["delta_solve_rate"]),
                    arrowprops={"arrowstyle": "->", "color": INK_MUTED, "lw": 1.2})  # fmt: skip
    for i, (name, r) in enumerate(ours.items()):
        color = colors.get(name, BLUE if i else ORANGE)
        ax.scatter(r["leak_rate"], r["delta_solve_rate"], s=90, color=color,
                   edgecolors=SURFACE, linewidths=2, zorder=3)  # fmt: skip
        ax.annotate(f"Qwen3.5-9B {name}", (r["leak_rate"], r["delta_solve_rate"]),
                    xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK,
                    weight="bold", bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE,
                                         "ec": "none", "alpha": 0.85})  # fmt: skip
    ax.text(0.99, 0.01, "Grey: paper Table 2 (Qwen2.5 tutors). Unlabeled grey: other RL λ values "
            "and Qwen2.5-14B; see the table below.", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7, color=INK_MUTED)  # fmt: skip
    ax.set_xlabel("Leaked solutions (%)  ← better", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("Δ solve rate (pp)  better →", fontsize=9, color=INK_MUTED)
    ax.set_title(f"Student gain vs. answer leakage (leak judge: {report.get('leak_judge')})",
                 loc="left", fontsize=10, color=INK)  # fmt: skip
    fig.tight_layout()
    plt.close(fig)
    return fig


EVAL_PANELS = [
    ("delta_solve_rate", "Δ solve rate (pp) ↑"),
    ("leak_rate", "Leaked solutions (%) ↓"),
    ("accept_rate", "Training judges accept ↑"),
    ("mean_turns", "Messages per dialogue"),
]


def eval_bars(report: dict):
    """One small panel per metric, a bar per policy (base orange, trained blue)."""
    import matplotlib.pyplot as plt

    names = list(report["results"])
    colors = [ORANGE if n == "base" else BLUE for n in names]
    fig, axes = plt.subplots(1, len(EVAL_PANELS), figsize=(13, 3.2), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, EVAL_PANELS, strict=True):
        _style(ax)
        ax.grid(False, axis="x")
        values = [report["results"][n].get(key, math.nan) for n in names]
        bars = ax.bar(names, values, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, values, strict=True):
            ax.annotate(f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}",
                        (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)  # fmt: skip
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    plt.close(fig)
    return fig
