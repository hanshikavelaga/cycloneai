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


class CycloneMultiTaskCNN(nn.Module):
    """
    Multi-Task Convolutional Neural Network (CNN) for Tropical Cyclones.
    Takes a 3-channel (IR, WV, VIS) image of size 128x128.
    Outputs:
    - Pattern Classification: 5 classes (No Cyclone, Shear, Curved Band, CDO, Eye).
    - Intensity Regression: 2 continuous targets (Wind Speed in knots, Dvorak T-number).
    """
    def __init__(self):
        super(CycloneMultiTaskCNN, self).__init__()
        # Input: (Batch, 3, 128, 128)
        
        # Convolutional Backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Shape: (Batch, 32, 64, 64)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks to learn complex cloud structures
        self.layer1 = ResidualBlock(32, 64, stride=2)    # Shape: (Batch, 64, 32, 32)
        self.layer2 = ResidualBlock(64, 128, stride=2)   # Shape: (Batch, 128, 16, 16)
        self.layer3 = ResidualBlock(128, 128, stride=2)  # Shape: (Batch, 128, 8, 8)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Shape: (Batch, 128, 1, 1)
        self.fc_shared = nn.Linear(128, 128)
        
        # Branch 1: Classification Head (5 pattern categories)
        self.fc_class = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(64, 5)
        )
        
        # Branch 2: Intensity Regression Head (Wind speed, Dvorak T-number)
        self.fc_reg = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # Shared Backbone Feature Extractor
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        shared_features = self.relu(self.fc_shared(out))
        
        # Branch predictions
        class_logits = self.fc_class(shared_features)
        intensity_preds = self.fc_reg(shared_features)
        
        return class_logits, intensity_preds


if __name__ == "__main__":
    # Test compilation & forward pass with dummy tensor
    model = CycloneMultiTaskCNN()
    dummy_input = torch.randn(2, 3, 128, 128)
    logits, reg = model(dummy_input)
    
    print("Compilation Test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Classification logits shape: {logits.shape} (Expected: [2, 5])")
    print(f"Intensity predictions shape: {reg.shape} (Expected: [2, 2])")
