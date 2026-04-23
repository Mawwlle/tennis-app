"""HeatmapEventNet training loop.

Uses BCEWithLogitsLoss with per-sample weights (TTNet smooth labeling).
Each sample carries a weight in [0, 1] that decays with temporal distance
from the annotated event frame — see eventnet/heatmap.py for details.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from eventnet.dataset import IDX_TO_LABEL, NUM_CLASSES
from eventnet.model import HeatmapEventNet, HeatmapEventNetConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def _weighted_bce(logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
    """BCEWithLogitsLoss weighted per-sample.

    Args:
        logits:  (B, C) raw logits
        targets: (B, C) soft targets in [0, 1]  (one-hot × smooth_weight)
        weights: (B,)   per-sample weight

    Returns:
        Scalar loss.
    """
    loss_per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B, C)
    loss_per_sample = loss_per_elem.mean(dim=1)                                             # (B,)
    return (loss_per_sample * weights).sum() / weights.sum()


def train_epoch(
    model: HeatmapEventNet,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs} [train]", leave=False, unit="batch")
    for heatmaps, targets, weights in bar:
        heatmaps = heatmaps.to(device)
        targets  = targets.to(device)
        weights  = weights.to(device)
        optimizer.zero_grad()
        loss = _weighted_bce(model(heatmaps), targets, weights)
        loss.backward()
        optimizer.step()
        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


@torch.no_grad()
def evaluate(
    model: HeatmapEventNet,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float | str]:
    model.eval()
    correct = 0
    total   = 0
    total_loss = 0.0
    per_class_correct: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}
    per_class_total:   dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}

    for heatmaps, targets, weights in loader:
        heatmaps = heatmaps.to(device)
        targets  = targets.to(device)
        weights  = weights.to(device)
        logits   = model(heatmaps)
        total_loss += _weighted_bce(logits, targets, weights).item()

        # For accuracy: argmax of logits vs argmax of one-hot target
        pred_cls   = logits.argmax(dim=1)
        target_cls = targets.argmax(dim=1)
        correct += int((pred_cls == target_cls).sum().item())
        total   += len(target_cls)

        for c in range(NUM_CLASSES):
            mask = target_cls == c
            per_class_correct[c] += int((pred_cls[mask] == c).sum().item())
            per_class_total[c]   += int(mask.sum().item())

    per_class_acc = {
        c: per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0.0
        for c in range(NUM_CLASSES)
    }
    label_tags = "  ".join(f"{IDX_TO_LABEL[c]}={per_class_acc[c]:.2f}" for c in range(NUM_CLASSES))
    return {
        "val_loss": total_loss / len(loader),
        "accuracy": correct / total if total > 0 else 0.0,
        "label_tags": label_tags,
        **{f"acc_{c}": per_class_acc[c] for c in range(NUM_CLASSES)},
    }


def run_training(
    train_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    cfg: HeatmapEventNetConfig,
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
            logging.FileHandler(output_dir / "eventnet_train.log"),
        ],
    )

    model     = HeatmapEventNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_acc  = 0.0
    best_path = output_dir / "eventnet_best.pt"

    n_params = sum(p.numel() for p in model.parameters())
    log.info("HeatmapEventNet  params=%d  device=%s  epochs=%d", n_params, device, epochs)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, epochs)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_loader, device)
            acc = metrics["accuracy"]
            log.info(
                "Epoch %3d/%d  train=%.4f  val=%.4f  acc=%.3f  [%s]",
                epoch, epochs, train_loss, metrics["val_loss"], acc,
                metrics["label_tags"],
            )
            if acc > best_acc:
                best_acc = acc
                torch.save({"state_dict": model.state_dict(), "cfg": cfg.model_dump()}, best_path)
                log.info("  ✓ new best acc=%.3f  saved → %s", best_acc, best_path)
        else:
            log.info("Epoch %3d/%d  train=%.4f", epoch, epochs, train_loss)

    log.info("Done. Best acc=%.3f  weights → %s", best_acc, best_path)
    return best_path
