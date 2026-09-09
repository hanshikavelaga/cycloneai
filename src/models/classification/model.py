import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A simple residual block helper.
    Consists of two Conv2d layers with Batch Normalization, ReLU, and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SimpleSatelliteCNN(nn.Module):
    """
    Baseline Lightweight CNN Architecture for Satellite Imagery.
    Provides a simple, fast benchmark before evaluating deeper residual models.
    """
    def __init__(self, in_channels=1, num_classes=3):
        super(SimpleSatelliteCNN, self).__init__()
        self.in_channels = in_channels
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                          # (B, 32, 32, 32)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                          # (B, 64, 8, 8)
            
            nn.AdaptiveAvgPool2d((2, 2))                                     # (B, 64, 2, 2)
        )
        
        self.flatten_dim = 64 * 2 * 2  # 256
        self.fc_shared = nn.Linear(self.flatten_dim, 64)
        
        # Classification Head
        self.fc_class = nn.Linear(64, num_classes)
        # Intensity Regression Head (Wind speed in knots)
        self.fc_reg = nn.Linear(64, 1)

    def forward(self, x):
        # Auto-adapt if input channels differ (e.g. 1-ch expanded or 3-ch grayscale)
        if x.shape[1] != self.in_channels:
            if self.in_channels == 1 and x.shape[1] == 3:
                x = x[:, :1, :, :]
            elif self.in_channels == 3 and x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
                
        feat = self.features(x)
        flat = torch.flatten(feat, 1)
        shared = F.relu(self.fc_shared(flat))
        
        logits = self.fc_class(shared)
        reg = self.fc_reg(shared).squeeze(-1)
        return logits, reg


class ResidualSatelliteCNN(nn.Module):
    """
    Production-grade Multi-Task Residual CNN for Tropical Cyclone Satellite Imagery.
    Features residual feature extraction blocks, dropout regularization,
    wind speed intensity regression, and IMD genesis stage classification.
    """
    def __init__(self, in_channels=1, num_classes=3):
        super(ResidualSatelliteCNN, self).__init__()
        self.in_channels = in_channels
        
        # Stem Convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Residual Stages
        self.stage1 = ResidualBlock(32, 64, stride=2)    # (B, 64, 32, 32)
        self.stage2 = ResidualBlock(64, 128, stride=2)   # (B, 128, 16, 16)
        self.stage3 = ResidualBlock(128, 128, stride=2)  # (B, 128, 8, 8)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        self.fc_shared = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        
        # Classification Head (IMD Categories)
        self.fc_class = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )
        
        # Regression Head (Wind Speed in knots)
        self.fc_reg = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Auto-adapt channel dimensions if needed
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
        shared = self.fc_shared(out)
        
        class_logits = self.fc_class(shared)
        intensity_pred = self.fc_reg(shared).squeeze(-1)
        
        return class_logits, intensity_pred


# Retain legacy class name for backwards compatibility with existing backend and seeder scripts
CycloneMultiTaskCNN = ResidualSatelliteCNN


if __name__ == "__main__":
    # Unit tests for architectures
    dummy_x = torch.randn(4, 1, 128, 128)
    
    baseline = SimpleSatelliteCNN(in_channels=1, num_classes=3)
    b_logits, b_reg = baseline(dummy_x)
    print("SimpleSatelliteCNN Baseline Test:")
    print(f"  Input: {dummy_x.shape} -> Logits: {b_logits.shape}, Reg: {b_reg.shape}")
    print(f"  Parameters: {sum(p.numel() for p in baseline.parameters()):,}")
    
    residual = ResidualSatelliteCNN(in_channels=1, num_classes=3)
    r_logits, r_reg = residual(dummy_x)
    print("\nResidualSatelliteCNN Architecture Test:")
    print(f"  Input: {dummy_x.shape} -> Logits: {r_logits.shape}, Reg: {r_reg.shape}")
    print(f"  Parameters: {sum(p.numel() for p in residual.parameters()):,}")
    print("\nAll models compiled and verified successfully!")
