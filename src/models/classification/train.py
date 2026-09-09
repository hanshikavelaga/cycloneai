"""
CycloneAI Satellite CNN Training & Evaluation Pipeline
=====================================================
Multi-task training loop for intensity regression and IMD category classification.
Features class-weighted loss, reproducible seeding, early stopping, and model comparison.
"""

import os
import sys
import time
import json
import random
import argparse

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error
)

from src.models.classification.dataset import get_dataloaders, CLASS_TO_IDX, IDX_TO_CLASS
from src.models.classification.model import SimpleSatelliteCNN, ResidualSatelliteCNN


def set_seed(seed=42):
    """Sets random seeds across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset):
    """
    Computes balanced inverse-frequency class weights for cross-entropy loss:
    w_c = N / (num_classes * N_c), normalized to mean 1.0.
    Ensures each class contributes equal weighted loss mass (1/K) during optimization,
    preventing majority-class gradient domination and class collapse.
    """
    counts = dataset.data['class_idx'].value_counts().sort_index()
    total = len(dataset)
    num_classes = len(CLASS_TO_IDX)
    
    raw_weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        w = total / (num_classes * max(c, 1))
        raw_weights.append(w)
        
    raw_weights = np.array(raw_weights, dtype=np.float32)
    # Normalize weights so mean is 1.0
    norm_weights = raw_weights / raw_weights.mean()
    return torch.tensor(norm_weights, dtype=torch.float32)


def train_one_epoch(model, dataloader, optimizer, criterion_reg, criterion_cls, device, reg_weight=0.01):
    model.train()
    total_loss, total_loss_reg, total_loss_cls = 0.0, 0.0, 0.0
    all_preds_cls, all_targets_cls = [], []
    all_preds_reg, all_targets_reg = [], []

    for images, winds, cats, _ in dataloader:
        images = images.to(device)
        winds = winds.to(device)
        cats = cats.to(device)

        optimizer.zero_grad()
        logits, reg = model(images)

        loss_reg = criterion_reg(reg, winds)
        loss_cls = criterion_cls(logits, cats)
        # Balanced gradient scaling: MSE (10-25) * 0.1 ~ 1.0-2.5 comparable with CE (~1.0)
        loss = (reg_weight * loss_reg) + loss_cls

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_loss_reg += loss_reg.item() * batch_size
        total_loss_cls += loss_cls.item() * batch_size

        preds_cls = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds_cls.extend(preds_cls)
        all_targets_cls.extend(cats.detach().cpu().numpy())
        all_preds_reg.extend(reg.detach().cpu().numpy())
        all_targets_reg.extend(winds.detach().cpu().numpy())

    n = len(dataloader.dataset)
    acc = accuracy_score(all_targets_cls, all_preds_cls)
    mae = mean_absolute_error(all_targets_reg, all_preds_reg)
    return total_loss / n, total_loss_reg / n, total_loss_cls / n, acc, mae


def evaluate(model, dataloader, criterion_reg, criterion_cls, device, reg_weight=0.01):
    model.eval()
    total_loss, total_loss_reg, total_loss_cls = 0.0, 0.0, 0.0
    all_preds_cls, all_targets_cls = [], []
    all_preds_reg, all_targets_reg = [], []

    with torch.no_grad():
        for images, winds, cats, _ in dataloader:
            images = images.to(device)
            winds = winds.to(device)
            cats = cats.to(device)

            logits, reg = model(images)
            loss_reg = criterion_reg(reg, winds)
            loss_cls = criterion_cls(logits, cats)
            loss = (reg_weight * loss_reg) + loss_cls

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_loss_reg += loss_reg.item() * batch_size
            total_loss_cls += loss_cls.item() * batch_size

            preds_cls = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds_cls.extend(preds_cls)
            all_targets_cls.extend(cats.cpu().numpy())
            all_preds_reg.extend(reg.cpu().numpy())
            all_targets_reg.extend(winds.cpu().numpy())

    n = len(dataloader.dataset)
    acc = accuracy_score(all_targets_cls, all_preds_cls)
    bal_acc = balanced_accuracy_score(all_targets_cls, all_preds_cls)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets_cls, all_preds_cls, average='macro', zero_division=0)
    mae = mean_absolute_error(all_targets_reg, all_preds_reg)
    rmse = np.sqrt(mean_squared_error(all_targets_reg, all_preds_reg))

    metrics = {
        'loss': total_loss / n,
        'loss_reg': total_loss_reg / n,
        'loss_cls': total_loss_cls / n,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'mae': mae,
        'rmse': rmse,
        'preds_cls': all_preds_cls,
        'targets_cls': all_targets_cls,
        'preds_reg': all_preds_reg,
        'targets_reg': all_targets_reg
    }
    return metrics



def train_model(model_name, epochs=30, batch_size=16, lr=1e-3, seed=42, save_best=True):
    set_seed(seed)
    device = torch.device("cpu")
    print("=" * 65)
    print(f"TRAINING MODEL: {model_name.upper()} on CPU (Seed={seed})")
    print("=" * 65)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    mapping_csv = os.path.join(repo_root, "data", "processed", "image_metadata_mapping.csv")
    models_dir = os.path.join(repo_root, "models", "classification")
    reports_dir = os.path.join(repo_root, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # DataLoaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        mapping_csv, batch_size=batch_size, repo_root=repo_root, in_channels=1
    )

    # Instantiate selected model architecture
    if model_name.lower() == "baseline":
        model = SimpleSatelliteCNN(in_channels=1, num_classes=len(CLASS_TO_IDX)).to(device)
    elif model_name.lower() == "residual":
        model = ResidualSatelliteCNN(in_channels=1, num_classes=len(CLASS_TO_IDX)).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Loss Functions with Class Weighting
    class_weights = compute_class_weights(train_ds).to(device)
    print(f"Computed Balanced Class Weights: {class_weights.tolist()}")
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_reg = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_score = -float('inf')
    best_metrics = None
    best_epoch = 0
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_lreg, tr_lcls, tr_acc, tr_mae = train_one_epoch(
            model, train_loader, optimizer, criterion_reg, criterion_cls, device
        )
        val_metrics = evaluate(model, val_loader, criterion_reg, criterion_cls, device)
        scheduler.step(val_metrics['loss'])

        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'val_loss': val_metrics['loss'],
            'train_acc': tr_acc,
            'val_acc': val_metrics['accuracy'],
            'val_bal_acc': val_metrics['balanced_accuracy'],
            'val_f1': val_metrics['f1'],
            'train_mae': tr_mae,
            'val_mae': val_metrics['mae']
        })

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {tr_loss:.4f} (Acc: {tr_acc*100:.1f}%, MAE: {tr_mae:.2f}) | "
              f"Val Loss: {val_metrics['loss']:.4f} (Acc: {val_metrics['accuracy']*100:.1f}%, Bal: {val_metrics['balanced_accuracy']*100:.1f}%, F1: {val_metrics['f1']:.3f}, MAE: {val_metrics['mae']:.2f} kts)")

        # Scientifically principled multi-objective validation selection
        # Rewards high Balanced Accuracy & Macro F1 across minority classes while keeping MAE low
        val_score = val_metrics['balanced_accuracy'] + val_metrics['f1'] - 0.05 * val_metrics['mae']
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_metrics = val_metrics

            if save_best:
                ckpt_path = os.path.join(models_dir, f"{model_name}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_score': best_score,
                    'val_metrics': {k: v for k, v in val_metrics.items() if not isinstance(v, list)},
                    'architecture': model.__class__.__name__,
                    'in_channels': 1,
                    'num_classes': len(CLASS_TO_IDX),
                    'class_mapping': CLASS_TO_IDX
                }, ckpt_path)

    elapsed_time = time.time() - start_time
    print(f"\nTraining finished in {elapsed_time:.1f}s. Best Epoch: {best_epoch} (Val F1: {best_metrics['f1']:.4f}, MAE: {best_metrics['mae']:.2f} kts)")

    # Save training curve history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(reports_dir, f"training_history_{model_name}.csv")
    history_df.to_csv(history_path, index=False)

    return {
        'model_name': model_name,
        'parameters': param_count,
        'training_time_s': round(elapsed_time, 2),
        'best_epoch': best_epoch,
        'val_loss': round(best_metrics['loss'], 4),
        'val_accuracy': round(best_metrics['accuracy'], 4),
        'val_balanced_accuracy': round(best_metrics['balanced_accuracy'], 4),
        'val_precision': round(best_metrics['precision'], 4),
        'val_recall': round(best_metrics['recall'], 4),
        'val_f1': round(best_metrics['f1'], 4),
        'val_mae_kts': round(best_metrics['mae'], 2),
        'val_rmse_kts': round(best_metrics['rmse'], 2)
    }



def compare_models():
    """Trains and compares candidate architectures on the validation set."""
    print("\n" + "=" * 70)
    print("CYCLONEAI: EXECUTING MODEL COMPARISON (PHASE 12)")
    print("=" * 70)

    # 1. Baseline Model
    res_baseline = train_model("baseline", epochs=25, batch_size=16)

    # 2. Residual Architecture Model
    res_residual = train_model("residual", epochs=25, batch_size=16)

    comparison_df = pd.DataFrame([res_baseline, res_residual])
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out_comparison = os.path.join(repo_root, "reports", "model_comparison.csv")
    comparison_df.to_csv(out_comparison, index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS (Validation Set 2011-2012):")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison table to: {out_comparison}")


if __name__ == "__main__":
    compare_models()
