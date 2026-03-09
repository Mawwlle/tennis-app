"""TrackNet training loop."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracknet.model import TrackNet

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

    log.info("Starting training: %d epochs, device=%s", epochs, device)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, epochs)

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_loader, device)
            scheduler.step(metrics["val_loss"])
            f1 = metrics["f1"]
            log.info(
                "Epoch %3d/%d  train=%.4f  val=%.4f  P=%.3f  R=%.3f  F1=%.3f",
                epoch, epochs, train_loss, metrics["val_loss"],
                metrics["precision"], metrics["recall"], f1,
            )
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)
                log.info("  ✓ new best F1=%.3f  saved → %s", best_f1, best_path)
        else:
            log.info("Epoch %3d/%d  train=%.4f", epoch, epochs, train_loss)

    log.info("Done. Best F1=%.3f  weights → %s", best_f1, best_path)
    return best_path
