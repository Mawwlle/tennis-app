from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from model.network import GameStateClassifier, build_model
from model.schemas import ModelConfig, TrainingConfig


def train_epoch(
    model: GameStateClassifier,
    dataloader: DataLoader[tuple[Tensor, int]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    for frames, labels in dataloader:
        frames = frames.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits: Tensor = model(frames)
        loss: Tensor = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        count += frames.size(0)

    return total_loss / max(count, 1)


def evaluate(
    model: GameStateClassifier,
    dataloader: DataLoader[tuple[Tensor, int]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            logits: Tensor = model(frames)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / max(total, 1)
    return {"accuracy": accuracy}


def run_training(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_loader: DataLoader[tuple[Tensor, int]],
    val_loader: DataLoader[tuple[Tensor, int]],
    output_dir: Path,
) -> Path:
    device = torch.device(training_config.device)
    model = build_model(model_config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    best_accuracy = 0.0

    for epoch in range(training_config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        accuracy = metrics["accuracy"]

        print(f"Epoch {epoch + 1}/{training_config.epochs} — loss: {train_loss:.4f}, val_acc: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_path)

    print(f"Best val accuracy: {best_accuracy:.4f}")
    return best_path
