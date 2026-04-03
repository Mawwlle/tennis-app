"""TCNEventNet training loop."""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from eventnet.model import TCNEventNet, TCNEventNetConfig


log = logging.getLogger(__name__)


def train_epoch(
    model: TCNEventNet,
    loader: DataLoader[tuple[Tensor, int]],
    optimizer: torch.optim.Optimizer,
    class_weights: Tensor,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs} [train]", leave=False, unit="batch")
    for feats, labels in bar:
        feats  = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(feats), labels, weight=class_weights)
        loss.backward()
        optimizer.step()
        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


@torch.no_grad()
def evaluate(
    model: TCNEventNet,
    loader: DataLoader[tuple[Tensor, int]],
    class_weights: Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total   = 0
    total_loss = 0.0
    per_class_correct: dict[int, int] = {i: 0 for i in range(4)}
    per_class_total:   dict[int, int] = {i: 0 for i in range(4)}

    for feats, labels in loader:
        feats  = feats.to(device)
        labels = labels.to(device)
        logits = model(feats)
        total_loss += F.cross_entropy(logits, labels, weight=class_weights).item()
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total   += len(labels)
        for c in range(4):
            mask = labels == c
            per_class_correct[c] += int((preds[mask] == c).sum().item())
            per_class_total[c]   += int(mask.sum().item())

    per_class_acc = {
        c: per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0.0
        for c in range(4)
    }
    return {
        "val_loss": total_loss / len(loader),
        "accuracy": correct / total if total > 0 else 0.0,
        **{f"acc_{c}": per_class_acc[c] for c in range(4)},
    }


def run_training(
    train_loader: DataLoader[tuple[Tensor, int]],
    val_loader: DataLoader[tuple[Tensor, int]],
    cfg: TCNEventNetConfig,
    class_weights: Tensor,
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

    model     = TCNEventNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    class_weights = class_weights.to(device)
    best_acc  = 0.0
    best_path = output_dir / "eventnet_best.pt"

    n_params = sum(p.numel() for p in model.parameters())
    log.info("TCNEventNet  params=%d  device=%s  epochs=%d", n_params, device, epochs)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, class_weights, device, epoch, epochs)

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_loader, class_weights, device)
            scheduler.step(metrics["val_loss"])
            acc = metrics["accuracy"]
            log.info(
                "Epoch %3d/%d  train=%.4f  val=%.4f  acc=%.3f  "
                "[hit=%.2f bounce=%.2f net=%.2f none=%.2f]",
                epoch, epochs, train_loss, metrics["val_loss"], acc,
                metrics["acc_0"], metrics["acc_1"], metrics["acc_2"], metrics["acc_3"],
            )
            if acc > best_acc:
                best_acc = acc
                torch.save({"state_dict": model.state_dict(), "cfg": cfg.model_dump()}, best_path)
                log.info("  ✓ new best acc=%.3f  saved → %s", best_acc, best_path)
        else:
            log.info("Epoch %3d/%d  train=%.4f", epoch, epochs, train_loss)

    log.info("Done. Best acc=%.3f  weights → %s", best_acc, best_path)
    return best_path
