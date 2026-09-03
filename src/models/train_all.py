import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

# Add parent directory to path to enable absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.classification.model import CycloneMultiTaskCNN
from src.models.genesis.model import CycloneGenesisLSTM
from src.models.track_prediction.model import CycloneTrackPredictorLSTM

# Output Weight Directories
MODEL_DIRS = {
    "classification": "./models/classification",
    "genesis": "./models/genesis",
    "track_prediction": "./models/track_prediction"
}

for d in MODEL_DIRS.values():
    os.makedirs(d, exist_ok=True)


def train_classification_model():
    print("\n--- Initializing & Seeding Classification CNN Model weights ---")
    model = CycloneMultiTaskCNN()
    
    # Generate 16 mock image samples (Shape: 3, 128, 128)
    # 5 classes: 0=No Cyclone, 1=Shear, 2=Curved Band, 3=CDO, 4=Eye
    dummy_images = torch.randn(16, 3, 128, 128)
    dummy_classes = torch.randint(0, 5, (16,))
    dummy_reg = torch.randn(16, 2)  # [wind_speed, t_number]
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train for 2 epochs to verify gradient steps
    model.train()
    for epoch in range(2):
        optimizer.zero_grad()
        class_logits, reg_outputs = model(dummy_images)
        
        loss_cls = criterion_cls(class_logits, dummy_classes)
        loss_reg = criterion_reg(reg_outputs, dummy_reg)
        loss = loss_cls + loss_reg
        
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/2 - Combined Loss: {loss.item():.4f}")
        
    save_path = os.path.join(MODEL_DIRS["classification"], "multi_task_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to: {save_path}")


def train_genesis_model():
    print("\n--- Initializing & Seeding Genesis LSTM Model weights ---")
    model = CycloneGenesisLSTM(input_size=7, hidden_size=32)
    
    # Dummy sequences (Batch=16, Sequence=5 steps, Features=7)
    dummy_seqs = torch.randn(16, 5, 7)
    dummy_labels = torch.randint(0, 2, (16, 1)).float()  # 0 or 1
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(2):
        optimizer.zero_grad()
        probs = model(dummy_seqs)
        loss = criterion(probs, dummy_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/2 - Genesis Loss: {loss.item():.4f}")
        
    save_path = os.path.join(MODEL_DIRS["genesis"], "genesis_lstm.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to: {save_path}")


def train_track_model():
    print("\n--- Initializing & Seeding Track Predictor LSTM Model weights ---")
    model = CycloneTrackPredictorLSTM(input_size=4, hidden_size=64, num_forecast_steps=4)
    
    # Dummy inputs: Batch=16, Sequence=4 steps, Features=4 [lat, lon, wind, press]
    dummy_input = torch.randn(16, 4, 4)
    dummy_target = torch.randn(16, 4, 4)  # Future: 4 steps * 4 features
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(2):
        optimizer.zero_grad()
        predictions = model(dummy_input)
        loss = criterion(predictions, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/2 - Track Forecast Loss: {loss.item():.4f}")
        
    save_path = os.path.join(MODEL_DIRS["track_prediction"], "track_lstm.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to: {save_path}")


if __name__ == "__main__":
    # Ensure torch is installed and runnable
    try:
        train_classification_model()
        train_genesis_model()
        train_track_model()
        print("\nAll models initialized, trained for 2 epochs, and checkpoints generated successfully.")
    except Exception as e:
        print(f"Error training models: {e}")
        sys.exit(1)
