import torch
from torch import Tensor, nn

from model.schemas import ModelConfig


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class GameStateClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            Conv3DBlock(3, 32),
            Conv3DBlock(32, 64),
            Conv3DBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, num_frames, 3, H, W) -> (B, 3, num_frames, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.blocks(x)
        x = self.pool(x)
        return self.fc(x)


def build_model(config: ModelConfig | None = None) -> GameStateClassifier:
    if config is None:
        config = ModelConfig()
    return GameStateClassifier(config)


def load_model(path: str, config: ModelConfig, device: torch.device) -> GameStateClassifier:
    model = build_model(config)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Model file not found: {path}") from exc
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
