"""SegNet training loop."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from segnet.dataset import SegNetDataset
from segnet.model import SegNet


def _atomic_save(obj: object, path: Path) -> None:
    """Save to a temp file then rename — prevents corrupted checkpoints on crash."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_training(
    dataset: SegNetDataset,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    device: torch.device,
    weights_dir: Path,
) -> None:
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    print(f"Train: {n_train}  Val: {n_val}")

    model = SegNet(num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    weights_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _atomic_save(model.state_dict(), weights_dir / "segnet_best.pt")
            print(f"  → saved (val_loss={val_loss:.4f})")

    _atomic_save(model.state_dict(), weights_dir / "segnet_last.pt")
