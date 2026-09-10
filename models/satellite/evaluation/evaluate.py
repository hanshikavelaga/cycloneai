"""
Evaluation Suite for Satellite-Only Tropical Cyclone Intensity Estimation.
Performs frozen single-pass evaluation on the held-out test set (2011-2015)
and records comprehensive regression, classification, and baseline metrics.

FINAL_TEST_EVALUATION = 1
"""

import os
import sys
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
TRAINING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "training"))
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

from dataset import SatelliteDataset
from model import ResidualSatelliteRegressor


# IMD Category Definition
CATEGORIES = ["LPA (<17 kts)", "Depression (17-27 kts)", "Deep Depression (28-33 kts)"]


def intensity_to_imd_category(wind_kts: float) -> str:
    """Deterministic mapping from continuous knots to IMD stage."""
    if wind_kts < 17.0:
        return "LPA (<17 kts)"
    elif wind_kts < 28.0:
        return "Depression (17-27 kts)"
    else:
        return "Deep Depression (28-33 kts)"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_mean: float,
    meta_list: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Computes all regression, classification, and baseline metrics.
    """
    # 1. Regression Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Correlations
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)

    # Threshold Accuracies
    abs_errors = np.abs(y_true - y_pred)
    acc_within_3kt = float(np.mean(abs_errors <= 3.0))
    acc_within_5kt = float(np.mean(abs_errors <= 5.0))
    acc_within_10kt = float(np.mean(abs_errors <= 10.0))

    # Standard Deviation Ratio
    std_true = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 1.0
    std_pred = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 1.0
    std_ratio = std_pred / std_true if std_true > 0 else 1.0

    # Mean Bias
    mean_bias = float(np.mean(y_pred - y_true))

    # Baseline comparison (predicting train_mean for all samples)
    baseline_preds = np.full_like(y_true, fill_value=train_mean)
    baseline_mae = float(mean_absolute_error(y_true, baseline_preds))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline_preds)))
    baseline_r2 = float(r2_score(y_true, baseline_preds))

    mae_improvement_kt = baseline_mae - mae
    mae_improvement_pct = (mae_improvement_kt / baseline_mae) * 100.0 if baseline_mae > 0 else 0.0

    # 2. Derived IMD Classification
    cat_true = [intensity_to_imd_category(val) for val in y_true]
    cat_pred = [intensity_to_imd_category(val) for val in y_pred]

    acc = float(accuracy_score(cat_true, cat_pred))
    balanced_acc = float(balanced_accuracy_score(cat_true, cat_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        cat_true, cat_pred, labels=CATEGORIES, zero_division=0
    )
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    per_class = {}
    for idx, cat_name in enumerate(CATEGORIES):
        per_class[cat_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1[idx]),
            "support": int(support[idx]),
        }

    # Confusion matrix
    cm = confusion_matrix(cat_true, cat_pred, labels=CATEGORIES)
    cm_df = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)

    # 3. Compile predictions table
    pred_records = []
    for i in range(len(y_true)):
        meta = meta_list[i]
        pred_records.append({
            "cyclone_id": meta.get("cyclone_id", ""),
            "cyclone_name": meta.get("cyclone_name", ""),
            "year": meta.get("year", 0),
            "timestamp_utc": meta.get("timestamp_utc", ""),
            "latitude": meta.get("latitude", 0.0),
            "longitude": meta.get("longitude", 0.0),
            "filename": meta.get("filename", ""),
            "actual_intensity_knots": float(y_true[i]),
            "predicted_intensity_knots": round(float(y_pred[i]), 2),
            "error_knots": round(float(y_pred[i] - y_true[i]), 2),
            "abs_error_knots": round(float(abs_errors[i]), 2),
            "actual_category": cat_true[i],
            "predicted_category": cat_pred[i],
            "correct_category": bool(cat_true[i] == cat_pred[i]),
        })
    preds_df = pd.DataFrame(pred_records)

    metrics = {
        "n_samples": len(y_true),
        "regression": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p_value": float(pearson_p),
            "spearman_rho": round(float(spearman_rho), 4),
            "spearman_p_value": float(spearman_p),
            "accuracy_within_3kt": round(acc_within_3kt, 4),
            "accuracy_within_5kt": round(acc_within_5kt, 4),
            "accuracy_within_10kt": round(acc_within_10kt, 4),
            "mean_bias_kt": round(mean_bias, 4),
            "std_ratio": round(std_ratio, 4),
            "std_true": round(std_true, 4),
            "std_pred": round(std_pred, 4),
        },
        "baseline_comparison": {
            "train_mean_knots": round(train_mean, 2),
            "baseline_mae": round(baseline_mae, 4),
            "baseline_rmse": round(baseline_rmse, 4),
            "baseline_r2": round(baseline_r2, 4),
            "mae_reduction_kt": round(mae_improvement_kt, 4),
            "mae_reduction_pct": round(mae_improvement_pct, 2),
        },
        "derived_classification": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": per_class,
        },
    }

    return metrics, preds_df, cm_df


def run_evaluation():
    manifest_path = os.path.join(BASE_DIR, "models", "satellite", "dataset_audit", "verified_satellite_manifest.csv")
    checkpoint_path = os.path.join(BASE_DIR, "models", "satellite", "checkpoints", "best_satellite_regressor.pth")
    eval_dir = os.path.join(BASE_DIR, "models", "satellite", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained checkpoint not found at {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"[Checkpoint Loaded] Epoch: {ckpt['epoch']}, Best Val MAE from training: {ckpt['val_mae']:.2f} kt")

    # Load model
    model = ResidualSatelliteRegressor(in_channels=1, dropout=0.3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load Manifest
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_mean = float(train_df["intensity_knots"].mean())
    print(f"[Baseline] Training set mean intensity: {train_mean:.2f} knots")

    # Helper to predict over dataset
    def predict_split(split_df: pd.DataFrame):
        dataset = SatelliteDataset(split_df, base_dir=BASE_DIR, is_train=False, augment=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        y_true_list = []
        y_pred_list = []
        meta_all = []

        with torch.no_grad():
            for images, targets, metas in loader:
                preds = model(images)
                y_true_list.extend(targets.numpy().tolist())
                y_pred_list.extend(preds.numpy().tolist())

                # Unpack batch dict of lists into list of dicts
                b_len = targets.size(0)
                for i in range(b_len):
                    m = {k: metas[k][i] for k in metas}
                    meta_all.append(m)

        return np.array(y_true_list), np.array(y_pred_list), meta_all

    # 1. Evaluate Validation Split (2009-2010, 30 obs)
    print("\nEvaluating Validation Split (2009-2010)...")
    val_true, val_pred, val_metas = predict_split(val_df)
    val_metrics, val_preds_df, val_cm_df = compute_metrics(val_true, val_pred, train_mean, val_metas)

    val_metrics_path = os.path.join(eval_dir, "validation_metrics.json")
    with open(val_metrics_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Validation MAE: {val_metrics['regression']['mae']:.2f} kt | RMSE: {val_metrics['regression']['rmse']:.2f} kt | R2: {val_metrics['regression']['r2']:.3f}")
    print(f"Validation Metrics saved -> {val_metrics_path}")

    # 2. Evaluate Frozen Held-Out Test Set (2011-2015, 46 obs) - EXACTLY ONCE
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION: HELD-OUT 2011-2015 SET (N=46, 23 CYCLONES)")
    print("FINAL_TEST_EVALUATION = 1")
    print("=" * 70)
    test_true, test_pred, test_metas = predict_split(test_df)
    test_metrics, test_preds_df, test_cm_df = compute_metrics(test_true, test_pred, train_mean, test_metas)

    test_metrics["split"] = "test (2011-2015)"
    test_metrics["final_test_evaluation_count"] = 1
    test_metrics["cyclones_count"] = test_df["cyclone_id"].nunique()

    test_metrics_path = os.path.join(eval_dir, "test_metrics.json")
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test Metrics saved -> {test_metrics_path}")

    test_preds_path = os.path.join(eval_dir, "test_predictions.csv")
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Test Predictions saved -> {test_preds_path}")

    cm_path = os.path.join(eval_dir, "confusion_matrix.csv")
    test_cm_df.to_csv(cm_path)
    print(f"Test Confusion Matrix saved -> {cm_path}")

    print("\n--- TEST EVALUATION SUMMARY ---")
    print(f"MAE:               {test_metrics['regression']['mae']:.2f} kt (Baseline: {test_metrics['baseline_comparison']['baseline_mae']:.2f} kt)")
    print(f"RMSE:              {test_metrics['regression']['rmse']:.2f} kt (Baseline: {test_metrics['baseline_comparison']['baseline_rmse']:.2f} kt)")
    print(f"R2:                {test_metrics['regression']['r2']:.3f}")
    print(f"Pearson r:         {test_metrics['regression']['pearson_r']:.3f} (p={test_metrics['regression']['pearson_p_value']:.4e})")
    print(f"Spearman rho:      {test_metrics['regression']['spearman_rho']:.3f}")
    print(f"Acc +/- 5 kt:      {test_metrics['regression']['accuracy_within_5kt']*100:.1f}%")
    print(f"Acc +/- 10 kt:     {test_metrics['regression']['accuracy_within_10kt']*100:.1f}%")
    print(f"Std Ratio:         {test_metrics['regression']['std_ratio']:.3f}")
    print(f"Classification Acc:{test_metrics['derived_classification']['accuracy']*100:.1f}%")
    print(f"Balanced Acc:      {test_metrics['derived_classification']['balanced_accuracy']*100:.1f}%")
    print(f"Macro F1:          {test_metrics['derived_classification']['macro_f1']*100:.1f}%")
    print("=" * 70 + "\n")

    return val_metrics, test_metrics


if __name__ == "__main__":
    run_evaluation()
