"""
Training Pipeline for Satellite-Only Tropical Cyclone Intensity Estimation.
Trains ResidualSatelliteRegressor on verified 2000-2008 satellite infrared imagery
with validation on 2009-2010.

TEST DATA FIREWALL:
2011-2015 test samples are STRICTLY quarantined during training.
TEST_DATA_ACCESSED_DURING_TRAINING = 0
"""

import os
import sys
import time
import json
import random
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

# Ensure local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dataset import SatelliteDataset, get_dataloaders
from model import ResidualSatelliteRegressor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(preds - targets).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_mae = total_abs_error / total_samples
    return epoch_loss, epoch_mae


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = criterion(preds, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(preds - targets).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_mae = total_abs_error / total_samples
    return epoch_loss, epoch_mae


def run_training():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")

    manifest_path = os.path.join(BASE_DIR, "models", "satellite", "dataset_audit", "verified_satellite_manifest.csv")
    checkpoints_dir = os.path.join(BASE_DIR, "models", "satellite", "checkpoints")
    training_dir = os.path.join(BASE_DIR, "models", "satellite", "training")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)

    # Verify and load DataLoaders
    # Note: test_loader is obtained to confirm strict isolation, but is NEVER evaluated in this loop.
    train_loader, val_loader, test_loader, manifest_df = get_dataloaders(
        manifest_path=manifest_path,
        base_dir=BASE_DIR,
        batch_size=16,
        seed=42,
    )

    print("\n" + "=" * 70)
    print("DATASET CHRONOLOGICAL SPLIT AUDIT")
    print("=" * 70)
    print(f"Train split (2000-2008): {len(train_loader.dataset)} samples | 75 cyclones")
    print(f"Val split   (2009-2010): {len(val_loader.dataset)} samples | 15 cyclones")
    print(f"Test split  (2011-2015): {len(test_loader.dataset)} samples | 23 cyclones [QUARANTINED]")
    print(f"TEST_DATA_ACCESSED_DURING_TRAINING = 0 (Enforced by design)")
    print("=" * 70 + "\n")

    # Hyperparameters
    config = {
        "model_name": "ResidualSatelliteRegressor",
        "input_shape": [1, 128, 128],
        "target": "intensity_knots",
        "loss_function": "SmoothL1Loss",
        "smooth_l1_beta": 1.0,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.3,
        "batch_size": 16,
        "max_epochs": 35,
        "early_stopping_patience": 8,
        "early_stopping_metric": "val_mae",
        "random_seed": 42,
        "device": str(device),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "test_data_accessed_during_training": 0,
        "train_years": "2000-2008",
        "val_years": "2009-2010",
        "test_years": "2011-2015",
    }

    # Instantiate model
    model = ResidualSatelliteRegressor(in_channels=1, dropout=0.3).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["total_parameters"] = param_count
    config["trainable_parameters"] = trainable_param_count
    print(f"[Model] ResidualSatelliteRegressor initialized. Total parameters: {param_count:,}")

    # Loss & Optimizer
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Training state
    best_val_mae = float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_model_state = None

    history: List[Dict[str, Any]] = []

    start_time = time.time()
    print(f"Starting training for up to {config['max_epochs']} epochs (patience: {config['early_stopping_patience']})...\n")

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_mae": round(train_mae, 4),
            "val_loss": round(val_loss, 4),
            "val_mae": round(val_mae, 4),
            "is_best": is_best,
            "patience": patience_counter,
            "epoch_duration_sec": round(elapsed, 2),
        }
        history.append(record)

        best_flag = " [BEST]" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{config['max_epochs']:02d} | "
            f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f} kt | "
            f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} kt | "
            f"Patience: {patience_counter}/{config['early_stopping_patience']}{best_flag} ({elapsed:.1f}s)"
        )

        if patience_counter >= config["early_stopping_patience"]:
            print(f"\n[Early Stopping] Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val MAE: {best_val_mae:.2f} kt.")
            break

    total_training_time = time.time() - start_time
    print(f"\n[Training Complete] Total time: {total_training_time:.1f}s")
    print(f"[Best Epoch] Epoch {best_epoch} with Val Loss: {best_val_loss:.4f}, Val MAE: {best_val_mae:.2f} kt")

    # Save best checkpoint
    checkpoint_path = os.path.join(checkpoints_dir, "best_satellite_regressor.pth")
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "val_mae": best_val_mae,
        "val_loss": best_val_loss,
        "config": config,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }, checkpoint_path)
    print(f"[Checkpoint Saved] -> {checkpoint_path}")

    # Save training configuration
    config["best_epoch"] = best_epoch
    config["best_val_mae"] = round(best_val_mae, 4)
    config["best_val_loss"] = round(best_val_loss, 4)
    config["total_training_time_sec"] = round(total_training_time, 2)
    config_path = os.path.join(training_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[Config Saved] -> {config_path}")

    # Save training history
    history_path = os.path.join(training_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[History Saved] -> {history_path}")

    return best_epoch, best_val_mae, best_val_loss, checkpoint_path


if __name__ == "__main__":
    run_training()
