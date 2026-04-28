"""Parse weights/eventnet_train.log and save a training history chart."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LOG_PATH = Path("weights/eventnet_train.log")
OUT_PATH = Path("weights/eventnet_training.png")

RUN_HEADERS = [
    (r"Starting EventNet training",        "EventNet (2-class)"),
    (r"Starting HeatmapEventNet training", "HeatmapEventNet (2-class)"),
    (r"TCNEventNet params=\d+",            "TCNEventNet v1 (4-class)"),
    (r"TCNEventNet params=\d+",            "TCNEventNet v2 (4-class)"),
]

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+train=([\d.]+)"
    r"(?:\s+val=([\d.]+))?"
    r"(?:\s+acc=([\d.]+))?"
    r"(?:\s+\[hit=([\d.]+)\s+bounce=([\d.]+)\s+net=([\d.]+)\s+none=([\d.]+)\])?"
)


@dataclass
class RunData:
    name: str
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_epochs: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    acc_epochs: list[int] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    hit_f1: list[float] = field(default_factory=list)
    bounce_f1: list[float] = field(default_factory=list)
    net_f1: list[float] = field(default_factory=list)
    none_f1: list[float] = field(default_factory=list)


def _parse_runs(lines: list[str]) -> list[RunData]:
    header_patterns = [re.compile(p) for p, _ in RUN_HEADERS]
    runs: list[RunData] = []
    current: RunData | None = None
    header_counts: dict[int, int] = {}

    for line in lines:
        line = line.strip()
        for idx, pat in enumerate(header_patterns):
            if pat.search(line):
                count = header_counts.get(idx, 0)
                header_counts[idx] = count + 1
                if idx < 2 or count == 0:
                    name = RUN_HEADERS[idx][1]
                else:
                    name = "TCNEventNet v2 (4-class)"
                current = RunData(name=name)
                runs.append(current)
                break

        if current is None:
            continue

        m = EPOCH_RE.search(line)
        if not m:
            continue

        ep = int(m.group(1))
        train = float(m.group(2))
        current.epochs.append(ep)
        current.train_loss.append(train)

        if m.group(3):
            current.val_epochs.append(ep)
            current.val_loss.append(float(m.group(3)))

        if m.group(4):
            current.acc_epochs.append(ep)
            current.accuracy.append(float(m.group(4)))

        if m.group(5):
            current.hit_f1.append(float(m.group(5)))
            current.bounce_f1.append(float(m.group(6)))
            current.net_f1.append(float(m.group(7)))
            current.none_f1.append(float(m.group(8)))

    return runs


def plot(runs: list[RunData], out: Path) -> None:
    n = len(runs)
    fig = plt.figure(figsize=(16, 4 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.45, wspace=0.3)

    for row, run in enumerate(runs):
        ax_loss = fig.add_subplot(gs[row, 0])
        ax_acc  = fig.add_subplot(gs[row, 1])

        ax_loss.set_title(f"{run.name} — Loss", fontsize=10)
        if run.epochs:
            ax_loss.plot(run.epochs, run.train_loss, label="train", linewidth=1.2)
        if run.val_epochs:
            ax_loss.plot(run.val_epochs, run.val_loss, label="val", linewidth=1.2, linestyle="--")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(fontsize=8)

        ax_acc.set_title(f"{run.name} — Accuracy", fontsize=10)
        if run.acc_epochs:
            ax_acc.plot(run.acc_epochs, run.accuracy, color="tab:green", linewidth=1.5, label="accuracy")
            best_acc = max(run.accuracy)
            ax_acc.axhline(best_acc, color="tab:green", linestyle=":", alpha=0.6,
                           label=f"best={best_acc:.3f}")

        if run.hit_f1:
            ep = run.acc_epochs[: len(run.hit_f1)]
            ax_acc.plot(ep, run.hit_f1,    label="hit",    linewidth=0.9, linestyle="--")
            ax_acc.plot(ep, run.bounce_f1, label="bounce", linewidth=0.9, linestyle="--")
            ax_acc.plot(ep, run.net_f1,    label="net",    linewidth=0.9, linestyle="--")
            ax_acc.plot(ep, run.none_f1,   label="none",   linewidth=0.9, linestyle="--")

        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Score")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.grid(alpha=0.3)
        ax_acc.legend(fontsize=8)

    fig.suptitle("EventNet Training History", fontsize=13, y=1.01)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main() -> None:
    text = LOG_PATH.read_text(errors="replace")
    lines = text.splitlines()
    runs = _parse_runs(lines)
    if not runs:
        raise RuntimeError("No training runs found in log file")
    for r in runs:
        best = max(r.accuracy) if r.accuracy else float("nan")
        print(f"  {r.name}: {len(r.epochs)} epochs, best acc={best:.3f}")
    plot(runs, OUT_PATH)


if __name__ == "__main__":
    main()
