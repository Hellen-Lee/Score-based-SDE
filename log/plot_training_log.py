"""
从 traininglog.txt 解析训练记录并绘制常用实验图（损失曲线、损失分布等）。
用法: python plot_training_log.py [--log PATH] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 与日志中的小节标题对应
SECTION_MAP = {
    "欧拉-丸山法": "Euler–Maruyama",
    "随机化欧拉法": "Stochastic Euler",
}

_ITER_LOSS = re.compile(r"Iteration\s+(\d+)\s+Loss\s+([-\d.eE+]+)")


def parse_training_log(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    runs: list[dict] = []
    for line in text.splitlines():
        s = line.strip()
        if s in SECTION_MAP:
            runs.append(
                {
                    "title": SECTION_MAP[s],
                    "label_zh": s,
                    "iterations": [],
                    "losses": [],
                }
            )
            continue
        m = _ITER_LOSS.search(line)
        if m and runs:
            runs[-1]["iterations"].append(int(m.group(1)))
            runs[-1]["losses"].append(float(m.group(2)))
    return runs


def plot_loss_curves(runs: list[dict], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2563eb", "#dc2626"]
    for i, run in enumerate(runs):
        it = np.array(run["iterations"])
        lo = np.array(run["losses"])
        label = run["title"]
        ax.plot(it, lo, color=colors[i % len(colors)], linewidth=1.8, label=label, alpha=0.95)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss: MNIST score-based SDE (two sampling methods)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_histograms(runs: list[dict], out_path: Path) -> None:
    """各次运行中记录的 Loss 值分布（日志每 500 step 一条，反映训练过程损失水平分布）。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2563eb", "#dc2626"]
    for i, run in enumerate(runs):
        lo = np.array(run["losses"])
        label = run["title"]
        ax.hist(
            lo,
            bins=18,
            alpha=0.55,
            color=colors[i % len(colors)],
            label=label,
            density=True,
        )
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of logged training losses (same checkpoints as curves)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot figures from traininglog.txt")
    p.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).resolve().parent / "traininglog.txt",
        help="Path to traininglog.txt",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for output PNG files",
    )
    args = p.parse_args()
    runs = parse_training_log(args.log)
    if not runs:
        raise SystemExit(f"No runs parsed from {args.log}")
    for r in runs:
        if not r["iterations"]:
            raise SystemExit(f"Run {r['title']} has no iteration/loss rows.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(runs, args.out_dir / "training_loss_curves.png")
    plot_loss_histograms(runs, args.out_dir / "training_loss_histogram.png")
    print(f"Wrote: {args.out_dir / 'training_loss_curves.png'}")
    print(f"Wrote: {args.out_dir / 'training_loss_histogram.png'}")


if __name__ == "__main__":
    main()
