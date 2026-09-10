"""
Plot Generation Module for Satellite-Only Tropical Cyclone Intensity Estimation.
Generates publication-quality diagnostic visualizations:
1. training_curves.png
2. prediction_vs_actual.png
3. residual_distribution.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))


def plot_training_curves(history_path: str, output_path: str):
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_mae = [r["train_mae"] for r in history]
    val_mae = [r["val_mae"] for r in history]

    best_epoch_idx = np.argmin(val_mae)
    best_epoch = epochs[best_epoch_idx]
    best_val_mae = val_mae[best_epoch_idx]

    plt.figure(figsize=(13, 5), dpi=300)

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss (Smooth L1)", color="#1f77b4", lw=2)
    plt.plot(epochs, val_loss, label="Val Loss (Smooth L1)", color="#ff7f0e", lw=2, linestyle="--")
    plt.axvline(best_epoch, color="#2ca02c", linestyle=":", label=f"Best Model (Epoch {best_epoch})")
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Loss", fontsize=11, fontweight="bold")
    plt.title("Training and Validation Loss", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", frameon=True)

    # Subplot 2: MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mae, label="Train MAE (kt)", color="#1f77b4", lw=2)
    plt.plot(epochs, val_mae, label="Val MAE (kt)", color="#d62728", lw=2, linestyle="--")
    plt.axvline(best_epoch, color="#2ca02c", linestyle=":", label=f"Best Epoch {best_epoch} ({best_val_mae:.2f} kt)")
    plt.axhline(best_val_mae, color="#2ca02c", linestyle=":", alpha=0.6)
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Absolute Error (knots)", fontsize=11, fontweight="bold")
    plt.title("Intensity Estimation MAE", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Created] Training Curves -> {output_path}")


def plot_prediction_vs_actual(test_preds_path: str, test_metrics_path: str, output_path: str):
    df = pd.read_csv(test_preds_path)
    with open(test_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    reg = metrics["regression"]
    actual = df["actual_intensity_knots"].values
    pred = df["predicted_intensity_knots"].values

    plt.figure(figsize=(7, 7), dpi=300)

    # Error bands
    min_val = min(actual.min(), pred.min()) - 3
    max_val = max(actual.max(), pred.max()) + 3
    line_x = np.linspace(min_val, max_val, 100)

    plt.fill_between(line_x, line_x - 5, line_x + 5, color="gray", alpha=0.15, label="±5 kt Margin")
    plt.plot(line_x, line_x, color="black", linestyle="--", lw=1.5, label="Identity (1:1)")

    # Scatter points colored by error
    abs_err = np.abs(actual - pred)
    sc = plt.scatter(actual, pred, c=abs_err, cmap="plasma", s=65, edgecolors="k", linewidths=0.5, zorder=4)
    cbar = plt.colorbar(sc)
    cbar.set_label("Absolute Error (knots)", fontsize=10, fontweight="bold")

    # Inset stats box
    stats_text = (
        f"Test Observations: {len(actual)}\n"
        f"MAE: {reg['mae']:.2f} kt\n"
        f"RMSE: {reg['rmse']:.2f} kt\n"
        f"R²: {reg['r2']:.3f}\n"
        f"Pearson r: {reg['pearson_r']:.3f}\n"
        f"±5 kt Accuracy: {reg['accuracy_within_5kt']*100:.1f}%\n"
        f"±10 kt Accuracy: {reg['accuracy_within_10kt']*100:.1f}%"
    )
    plt.gca().text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
        fontsize=9,
        fontfamily="monospace",
    )

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Actual Ground-Truth Intensity (knots)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Intensity (knots)", fontsize=11, fontweight="bold")
    plt.title("Satellite CNN: Predicted vs Actual Wind Speed (Test 2011–2015)", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Created] Prediction vs Actual -> {output_path}")


def plot_residual_distribution(test_preds_path: str, test_metrics_path: str, output_path: str):
    df = pd.read_csv(test_preds_path)
    with open(test_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    residuals = (df["predicted_intensity_knots"] - df["actual_intensity_knots"]).values
    reg = metrics["regression"]

    plt.figure(figsize=(8, 5), dpi=300)
    counts, bins, patches = plt.hist(residuals, bins=12, color="#3498db", edgecolor="black", alpha=0.7, density=True)

    # Fit Gaussian curve
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    x = np.linspace(residuals.min() - 2, residuals.max() + 2, 200)
    p = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    plt.plot(x, p, "r-", lw=2, label=f"Normal Fit (μ={mu:.2f}, σ={sigma:.2f})")

    plt.axvline(0, color="black", linestyle="--", lw=1.5, label="Zero Error (Unbiased)")
    plt.axvline(mu, color="red", linestyle=":", lw=1.5, label=f"Mean Bias = {mu:.2f} kt")

    stats_text = (
        f"Mean Bias: {reg['mean_bias_kt']:.2f} kt\n"
        f"Std Dev (σ): {sigma:.2f} kt\n"
        f"MAE: {reg['mae']:.2f} kt\n"
        f"Std Ratio (σ_pred/σ_true): {reg['std_ratio']:.3f}"
    )
    plt.gca().text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
        fontsize=9,
        fontfamily="monospace",
    )

    plt.xlabel("Residual Error: Predicted - Actual (knots)", fontsize=11, fontweight="bold")
    plt.ylabel("Probability Density", fontsize=11, fontweight="bold")
    plt.title("Satellite CNN Test Residual Distribution (2011–2015)", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Created] Residual Distribution -> {output_path}")


def generate_all_plots():
    history_path = os.path.join(BASE_DIR, "models", "satellite", "training", "training_history.json")
    test_preds_path = os.path.join(BASE_DIR, "models", "satellite", "evaluation", "test_predictions.csv")
    test_metrics_path = os.path.join(BASE_DIR, "models", "satellite", "evaluation", "test_metrics.json")
    plots_dir = os.path.join(BASE_DIR, "models", "satellite", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_training_curves(history_path, os.path.join(plots_dir, "training_curves.png"))
    plot_prediction_vs_actual(test_preds_path, test_metrics_path, os.path.join(plots_dir, "prediction_vs_actual.png"))
    plot_residual_distribution(test_preds_path, test_metrics_path, os.path.join(plots_dir, "residual_distribution.png"))


if __name__ == "__main__":
    generate_all_plots()
