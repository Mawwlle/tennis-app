"""TrackNet training loop."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracknet.model import TrackNet

matplotlib.use("Agg")  # headless — no display needed

log = logging.getLogger(__name__)


def focal_bce_loss(pred: Tensor, target: Tensor, gamma: float = 2.0, pos_weight: float = 50.0) -> Tensor:
    pw = torch.tensor(pos_weight, device=pred.device)
    bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw, reduction="none")
    p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
    return (((1 - p_t) ** gamma) * bce).mean()


def _detection_metrics(
    pred_logits: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    dist_tol: int = 5,
) -> dict[str, float]:
    tp = fp = fn = 0
    preds = torch.sigmoid(pred_logits).squeeze(1).cpu().numpy()
    gts = target.squeeze(1).cpu().numpy()

    for pred_map, gt_map in zip(preds, gts):
        ball_in_gt = gt_map.max() > 0.1
        if (pred_map > threshold).any():
            py, px = divmod(int(pred_map.argmax()), pred_map.shape[1])
            if ball_in_gt:
                gy, gx = divmod(int(gt_map.argmax()), gt_map.shape[1])
                if ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 <= dist_tol:
                    tp += 1
                else:
                    fp += 1; fn += 1
            else:
                fp += 1
        elif ball_in_gt:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def train_epoch(
    model: TrackNet,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs} [train]", leave=False, unit="batch")
    for frames, heatmaps in bar:
        frames   = frames.to(device)
        heatmaps = heatmaps.to(device)
        optimizer.zero_grad()
        loss = focal_bce_loss(model(frames), heatmaps)
        loss.backward()
        optimizer.step()
        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


@torch.no_grad()
def evaluate(
    model: TrackNet,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_metrics: list[dict[str, float]] = []
    total_loss = 0.0
    for frames, heatmaps in tqdm(loader, desc="             [val] ", leave=False, unit="batch"):
        frames   = frames.to(device)
        heatmaps = heatmaps.to(device)
        logits = model(frames)
        total_loss += focal_bce_loss(logits, heatmaps).item()
        all_metrics.append(_detection_metrics(logits, heatmaps))

    avg = lambda key: float(np.mean([m[key] for m in all_metrics]))
    return {
        "val_loss": total_loss / len(loader),
        "precision": avg("precision"),
        "recall": avg("recall"),
        "f1": avg("f1"),
    }


def _save_plot(
    train_losses: list[tuple[int, float]],
    val_losses: list[tuple[int, float]],
    val_metrics: list[tuple[int, float, float, float]],
    output_path: Path,
) -> None:
    """Save training curves to a PNG file.

    train_losses:  [(epoch, loss), ...]
    val_losses:    [(epoch, loss), ...]
    val_metrics:   [(epoch, f1, precision, recall), ...]
    """
    epochs_train, losses_train = zip(*train_losses)
    epochs_val,   losses_val   = zip(*val_losses)
    epochs_m, f1s, precs, recs = zip(*val_metrics)

    fig = plt.figure(figsize=(14, 14))
    fig.suptitle("TrackNet Training", fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(3, 1, hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # — Loss —
    ax1.plot(epochs_train, losses_train, label="train loss", color="steelblue", linewidth=1.5)
    ax1.plot(epochs_val,   losses_val,   label="val loss",   color="tomato",    linewidth=1.5, marker="o", markersize=4)
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # — Metrics —
    ax2.plot(epochs_m, f1s,   label="F1",        color="seagreen",  linewidth=1.5, marker="o", markersize=4)
    ax2.plot(epochs_m, precs, label="Precision",  color="goldenrod", linewidth=1.5, marker="s", markersize=4)
    ax2.plot(epochs_m, recs,  label="Recall",     color="orchid",    linewidth=1.5, marker="^", markersize=4)
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # — Combined: loss (left axis) + metrics (right axis) —
    ax3.set_title("Overview", fontsize=11)
    ax3_r = ax3.twinx()

    ax3.plot(epochs_train, losses_train, color="steelblue", linewidth=1.2, alpha=0.7, label="train loss")
    ax3.plot(epochs_val,   losses_val,   color="tomato",    linewidth=1.2, alpha=0.7, label="val loss")
    ax3_r.plot(epochs_m, f1s,   color="seagreen",  linewidth=1.5, marker="o", markersize=5, label="F1")
    ax3_r.plot(epochs_m, precs, color="goldenrod", linewidth=1.2, marker="s", markersize=4, label="Precision", alpha=0.8)
    ax3_r.plot(epochs_m, recs,  color="orchid",    linewidth=1.2, marker="^", markersize=4, label="Recall",    alpha=0.8)

    for ep in epochs_m:
        ax3.axvline(ep, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    ax3.set_ylabel("Loss", color="steelblue")
    ax3.tick_params(axis="y", labelcolor="steelblue")
    ax3_r.set_ylabel("Score", color="seagreen")
    ax3_r.tick_params(axis="y", labelcolor="seagreen")
    ax3_r.set_ylim(0, 1)
    ax3.set_xlabel("Epoch")
    ax3.grid(True, alpha=0.2)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def run_training(
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )

    model     = TrackNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_f1   = 0.0
    best_path = output_dir / "tracknet_best.pt"
    plot_path = output_dir / "train_plot.png"

    train_losses: list[tuple[int, float]] = []
    val_losses:   list[tuple[int, float]] = []
    val_metrics:  list[tuple[int, float, float, float]] = []

    log.info("Starting training: %d epochs, device=%s", epochs, device)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, epochs)
        train_losses.append((epoch, train_loss))

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_loader, device)
            scheduler.step(metrics["val_loss"])
            f1 = metrics["f1"]

            val_losses.append((epoch, metrics["val_loss"]))
            val_metrics.append((epoch, f1, metrics["precision"], metrics["recall"]))

            log.info(
                "Epoch %3d/%d  train=%.4f  val=%.4f  P=%.3f  R=%.3f  F1=%.3f",
                epoch, epochs, train_loss, metrics["val_loss"],
                metrics["precision"], metrics["recall"], f1,
            )
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)
                log.info("  ✓ new best F1=%.3f  saved → %s", best_f1, best_path)

            _save_plot(train_losses, val_losses, val_metrics, plot_path)
        else:
            log.info("Epoch %3d/%d  train=%.4f", epoch, epochs, train_loss)

    log.info("Done. Best F1=%.3f  weights → %s", best_f1, best_path)
    log.info("Plot saved → %s", plot_path)
    return best_path
