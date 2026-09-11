"""
Production-grade Residual CNN Regressor for Satellite Infrared Imagery.
Estimates contemporaneous tropical cyclone intensity (wind speed in knots).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions, batch normalization, and skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResidualSatelliteRegressor(nn.Module):
    """
    Residual CNN Regressor for Satellite Infrared Imagery.
    Input: (B, 1, 128, 128)
    Output: (B,) wind speed in knots
    Total Parameters: 620,961
    """
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super(ResidualSatelliteRegressor, self).__init__()
        self.in_channels = in_channels

        # Stem: (B, 1, 128, 128) -> (B, 32, 64, 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: (B, 32, 64, 64) -> (B, 64, 32, 32)
        self.stage1 = ResidualBlock(32, 64, stride=2)

        # Stage 2: (B, 64, 32, 32) -> (B, 128, 16, 16)
        self.stage2 = ResidualBlock(64, 128, stride=2)

        # Stage 3: (B, 128, 16, 16) -> (B, 128, 8, 8)
        self.stage3 = ResidualBlock(128, 128, stride=2)

        # Global average pooling: (B, 128, 8, 8) -> (B, 128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Non-linear regression head: 128 -> 128 -> 32 -> 1
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] != self.in_channels:
            if self.in_channels == 1 and x.shape[1] == 3:
                x = x[:, :1, :, :]
            elif self.in_channels == 3 and x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        pred = self.head(out).squeeze(-1)
        return pred


def get_model(in_channels: int = 1, dropout: float = 0.3) -> ResidualSatelliteRegressor:
    return ResidualSatelliteRegressor(in_channels=in_channels, dropout=dropout)
