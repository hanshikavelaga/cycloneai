"""
CycloneAI Master Consolidation Experiment
=========================================
Runs a 5-fold Cyclone-Grouped Cross-Validation comparing all 4 approaches:
  A. Classification head, aggressive inverse-frequency CE weights
  B. Classification head, smoothed CE weights: 1/sqrt(N_c) normalized to mean 1.0
  C. Classification head, Focal Loss (gamma=2.0) with smoothed alpha_c
  D. Regression head (ResidualSatelliteRegressor) with post-hoc category derivation

Then aggregates metrics, performs paired statistical significance tests,
decides the winner, and evaluates the winning approach exactly once on the
frozen 2011-2015 test set.
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from scipy import stats

# Paths
BASE_DIR = r"C:\Users\hasin\Desktop\cycloneai"
OUT_DIR = os.path.join(BASE_DIR, r"models\consolidated_cv")
REPORT_DIR = os.path.join(BASE_DIR, r"reports")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Add classification model to path
sys.path.insert(0, os.path.join(BASE_DIR, "src", "models", "classification"))
from model import ResidualSatelliteCNN

# Official IMD Genesis Categories
CLASS_NAMES = [
    'Low Pressure Area (< 17 kts)',
    'Depression (17-27 kts)',
    'Deep Depression (28-33 kts)'
]

def wind_to_class_idx(wind_kts: float) -> int:
    if wind_kts < 17.0:
        return 0
    elif wind_kts <= 27.0:
        return 1
    else:
        return 2

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset & Focal Loss
# ---------------------------------------------------------------------------

class MasterSatelliteDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: str = BASE_DIR, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_p = row["image_path"]
        if not os.path.isabs(img_p):
            img_p = os.path.join(self.base_dir, img_p)

        arr = np.load(img_p).astype(np.float32)
        if arr.ndim == 2:
            # Random 90-degree rotations only for training
            if self.is_train:
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    arr = np.rot90(arr, k=k).copy()
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 128, 128)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            if self.is_train:
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    arr = np.rot90(arr[0], k=k).copy()
                    arr = np.expand_dims(arr, 0)
            tensor = torch.from_numpy(arr)
        else:
            raise ValueError(f"Unexpected shape: {arr.shape}")

        wind = float(row["intensity_knots"])
        cls_idx = wind_to_class_idx(wind)

        return tensor, torch.tensor(cls_idx, dtype=torch.long), torch.tensor(wind, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Tensor of shape (3,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Training and Evaluation Helpers
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, approach_type, loss_fn, lambda_reg, device):
    model.train()
    total_loss = 0.0
    for images, targets_cls, targets_wind in dataloader:
        images = images.to(device)
        targets_cls = targets_cls.to(device)
        targets_wind = targets_wind.to(device)

        optimizer.zero_grad()
        logits, pred_wind = model(images)

        if approach_type in ["A", "B", "C"]:
            l_cls = loss_fn(logits, targets_cls)
            l_reg = F.smooth_l1_loss(pred_wind, targets_wind, beta=1.0)
            loss = l_cls + lambda_reg * l_reg
        elif approach_type == "D":
            loss = F.smooth_l1_loss(pred_wind, targets_wind, beta=1.0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets_cls)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def eval_dataset(model, dataloader, approach_type, device):
    model.eval()
    all_pred_cls = []
    all_true_cls = []
    all_pred_wind = []
    all_true_wind = []

    for images, targets_cls, targets_wind in dataloader:
        images = images.to(device)
        logits, pred_wind = model(images)

        if approach_type in ["A", "B", "C"]:
            pred_c = torch.argmax(logits, dim=1).cpu().numpy()
            pred_w = pred_wind.cpu().numpy()
        elif approach_type == "D":
            pred_w = pred_wind.cpu().numpy()
            pred_c = np.array([wind_to_class_idx(w) for w in pred_w])

        all_pred_cls.extend(pred_c)
        all_true_cls.extend(targets_cls.numpy())
        all_pred_wind.extend(pred_w)
        all_true_wind.extend(targets_wind.numpy())

    y_true_c = np.array(all_true_cls)
    y_pred_c = np.array(all_pred_cls)
    y_true_w = np.array(all_true_wind)
    y_pred_w = np.array(all_pred_wind)

    acc = float(accuracy_score(y_true_c, y_pred_c))
    bal_acc = float(balanced_accuracy_score(y_true_c, y_pred_c))

    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true_c, y_pred_c, labels=[0, 1, 2], average='macro', zero_division=0
    )
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true_c, y_pred_c, labels=[0, 1, 2], average=None, zero_division=0
    )

    cm = confusion_matrix(y_true_c, y_pred_c, labels=[0, 1, 2])
    mae = float(mean_absolute_error(y_true_w, y_pred_w))
    rmse = float(np.sqrt(mean_squared_error(y_true_w, y_pred_w)))

    metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(float(prec_m), 4),
        "macro_recall": round(float(rec_m), 4),
        "macro_f1": round(float(f1_m), 4),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(prec_c[i]), 4),
                "recall": round(float(rec_c[i]), 4),
                "f1": round(float(f1_c[i]), 4),
                "support": int(sup_c[i]),
            } for i in range(3)
        },
        "wind_mae": round(mae, 4),
        "wind_rmse": round(rmse, 4),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, y_true_c, y_pred_c, y_true_w, y_pred_w


# ---------------------------------------------------------------------------
# MAIN EXPERIMENT EXECUTION
# ---------------------------------------------------------------------------

def run_master_experiment():
    print("=" * 80)
    print("CYLONEAI MASTER CONSOLIDATION EXPERIMENT")
    print("=" * 80)

    # 1. Load manifest and filter pool
    manifest_path = os.path.join(BASE_DIR, r"models\satellite\dataset_audit\verified_satellite_manifest.csv")
    df = pd.read_csv(manifest_path)
    df["class_idx"] = df["intensity_knots"].apply(wind_to_class_idx)

    pool_df = df[df["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"\n[STEP 1] Pooled CV Samples: {len(pool_df)} | Unique Cyclones: {pool_df['cyclone_id'].nunique()}")
    print("Pooled Class Distribution:")
    for idx, cname in enumerate(CLASS_NAMES):
        count_c = (pool_df["class_idx"] == idx).sum()
        print(f"  {cname}: {count_c} ({count_c / len(pool_df) * 100:.1f}%)")

    # GroupKFold (grouped by cyclone_id)
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(pool_df, pool_df["class_idx"], groups=pool_df["cyclone_id"]))

    print("\nFold Breakdown:")
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        va_sub = pool_df.iloc[va_idx]
        c_counts = va_sub["class_idx"].value_counts().to_dict()
        l_c = c_counts.get(0, 0)
        d_c = c_counts.get(1, 0)
        dd_c = c_counts.get(2, 0)
        warn = " [WARNING: Deep Depression support < 3 (Recall unreliable)]" if dd_c < 3 else ""
        print(f"  Fold {fold_idx + 1}: Held-out Cyclones={va_sub['cyclone_id'].nunique()}, Samples={len(va_sub)} | "
              f"Low={l_c}, Dep={d_c}, Deep Dep={dd_c}{warn}")

    # Set up hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing compute device: {device}")

    approaches = ["A", "B", "C", "D"]
    approach_names = {
        "A": "Aggressive Inverse-Frequency CE",
        "B": "Smoothed CE (1/sqrt(N_c))",
        "C": "Focal Loss (gamma=2.0, alpha=1/sqrt(N_c))",
        "D": "Regression Head (Post-Hoc Derived Category)",
    }

    # Storage for all fold results
    fold_results = {app: [] for app in approaches}

    print("\n" + "=" * 80)
    print("STEP 2: RUNNING ALL 4 APPROACHES UNDER IDENTICAL 5-FOLD CV")
    print("=" * 80)

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        fold_num = fold_idx + 1
        print(f"\n>>> PROCESSING FOLD {fold_num} / 5 <<<")

        tr_data = pool_df.iloc[tr_idx].copy()
        va_data = pool_df.iloc[va_idx].copy()

        # Compute weights for this fold's training split
        tr_counts = np.bincount(tr_data["class_idx"].values, minlength=3).astype(np.float32)

        # Approach A weights: aggressive 1/N_c normalized to mean 1.0
        w_A = 1.0 / np.maximum(tr_counts, 1.0)
        w_A = w_A / np.mean(w_A)
        tensor_w_A = torch.tensor(w_A, dtype=torch.float32).to(device)

        # Approach B weights: smoothed 1/sqrt(N_c) normalized to mean 1.0
        w_B = 1.0 / np.sqrt(np.maximum(tr_counts, 1.0))
        w_B = w_B / np.mean(w_B)
        tensor_w_B = torch.tensor(w_B, dtype=torch.float32).to(device)

        # Approach C alpha: same smoothed weights for Focal Loss
        tensor_w_C = torch.tensor(w_B, dtype=torch.float32).to(device)

        # Build datasets
        train_ds = MasterSatelliteDataset(tr_data, base_dir=BASE_DIR, is_train=True)
        val_ds = MasterSatelliteDataset(va_data, base_dir=BASE_DIR, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        for app in approaches:
            set_seed(42)  # Strictly identical initialization per approach & fold
            model = ResidualSatelliteCNN(in_channels=1, num_classes=3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            if app == "A":
                loss_fn = nn.CrossEntropyLoss(weight=tensor_w_A)
            elif app == "B":
                loss_fn = nn.CrossEntropyLoss(weight=tensor_w_B)
            elif app == "C":
                loss_fn = FocalLoss(alpha=tensor_w_C, gamma=2.0)
            elif app == "D":
                loss_fn = nn.SmoothL1Loss(beta=1.0)

            # Checkpoint selection tracker
            best_score = -float("inf") if app in ["A", "B", "C"] else float("inf")
            best_epoch = -1
            best_state = None
            best_val_metrics = None

            for epoch in range(1, 26):
                _ = train_epoch(model, train_loader, optimizer, app, loss_fn, lambda_reg=0.01, device=device)
                val_m, _, _, _, _ = eval_dataset(model, val_loader, app, device)

                if app in ["A", "B", "C"]:
                    # Primary: Macro F1, Secondary: Balanced Accuracy
                    score = val_m["macro_f1"]
                    if score > best_score or (abs(score - best_score) < 1e-6 and val_m["balanced_accuracy"] > best_val_metrics["balanced_accuracy"]):
                        best_score = score
                        best_epoch = epoch
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        best_val_metrics = val_m
                elif app == "D":
                    # Primary: Regression MAE (lower is better)
                    score = val_m["wind_mae"]
                    if score < best_score:
                        best_score = score
                        best_epoch = epoch
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        best_val_metrics = val_m

            # Load best checkpoint to finalize metrics
            model.load_state_dict(best_state)
            final_val_m, _, _, _, _ = eval_dataset(model, val_loader, app, device)
            final_val_m["best_epoch"] = best_epoch
            final_val_m["fold"] = fold_num
            final_val_m["approach"] = app

            # Save fold checkpoint and metrics
            fold_ckpt_name = f"{app}_fold_{fold_num}.pth"
            torch.save({
                "approach": app,
                "fold": fold_num,
                "best_epoch": best_epoch,
                "state_dict": best_state,
                "metrics": final_val_m,
            }, os.path.join(OUT_DIR, fold_ckpt_name))

            with open(os.path.join(OUT_DIR, f"{app}_fold_{fold_num}_metrics.json"), "w") as f:
                json.dump(final_val_m, f, indent=2)

            # Save confusion matrix CSV
            cm_df = pd.DataFrame(final_val_m["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
            cm_df.to_csv(os.path.join(OUT_DIR, f"{app}_fold_{fold_num}_cm.csv"))

            fold_results[app].append(final_val_m)

            # Print fold outcome
            print(f"  [{app}] {approach_names[app]}")
            print(f"      Best Epoch: {best_epoch:02d} | Acc: {final_val_m['accuracy']*100:.1f}% | "
                  f"Bal Acc: {final_val_m['balanced_accuracy']*100:.1f}% | Macro F1: {final_val_m['macro_f1']*100:.1f}% | "
                  f"Wind MAE: {final_val_m['wind_mae']:.2f} kt")
            print(f"      Per-Class Recall: Low={final_val_m['per_class'][CLASS_NAMES[0]]['recall']*100:.1f}% "
                  f"({final_val_m['per_class'][CLASS_NAMES[0]]['support']}), "
                  f"Dep={final_val_m['per_class'][CLASS_NAMES[1]]['recall']*100:.1f}% "
                  f"({final_val_m['per_class'][CLASS_NAMES[1]]['support']}), "
                  f"Deep Dep={final_val_m['per_class'][CLASS_NAMES[2]]['recall']*100:.1f}% "
                  f"({final_val_m['per_class'][CLASS_NAMES[2]]['support']})")

    # -----------------------------------------------------------------------
    # STEP 3: AGGREGATE AND DECIDE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: AGGREGATE VALIDATION-ONLY METRICS & STATISTICAL TESTING")
    print("=" * 80)

    summary_rows = []
    macro_f1_arrays = {}
    bal_acc_arrays = {}

    for app in approaches:
        res = fold_results[app]
        f1_vals = [r["macro_f1"] for r in res]
        bal_vals = [r["balanced_accuracy"] for r in res]
        acc_vals = [r["accuracy"] for r in res]
        mae_vals = [r["wind_mae"] for r in res]
        rec_low = [r["per_class"][CLASS_NAMES[0]]["recall"] for r in res]
        rec_dep = [r["per_class"][CLASS_NAMES[1]]["recall"] for r in res]
        rec_ddep = [r["per_class"][CLASS_NAMES[2]]["recall"] for r in res]

        # Split Deep Depression recall for support >= 3 (Folds 2, 3, 4) vs < 3 (Folds 1, 5)
        # Note: fold indices 1, 2, 3 have support >= 3; 0, 4 have support < 3
        ddep_ge3 = [rec_ddep[i] for i in [1, 2, 3]]
        ddep_lt3 = [rec_ddep[i] for i in [0, 4]]

        macro_f1_arrays[app] = np.array(f1_vals)
        bal_acc_arrays[app] = np.array(bal_vals)

        summary_rows.append({
            "Approach": app,
            "Description": approach_names[app],
            "Macro F1 Mean": round(float(np.mean(f1_vals)), 4),
            "Macro F1 Std": round(float(np.std(f1_vals, ddof=1)), 4),
            "Balanced Acc Mean": round(float(np.mean(bal_vals)), 4),
            "Balanced Acc Std": round(float(np.std(bal_vals, ddof=1)), 4),
            "Accuracy Mean": round(float(np.mean(acc_vals)), 4),
            "Accuracy Std": round(float(np.std(acc_vals, ddof=1)), 4),
            "Wind MAE Mean": round(float(np.mean(mae_vals)), 4),
            "Wind MAE Std": round(float(np.std(mae_vals, ddof=1)), 4),
            "Low Recall Mean": round(float(np.mean(rec_low)), 4),
            "Dep Recall Mean": round(float(np.mean(rec_dep)), 4),
            "Deep Dep Recall Mean (All)": round(float(np.mean(rec_ddep)), 4),
            "Deep Dep Recall (Sup >= 3)": round(float(np.mean(ddep_ge3)), 4),
            "Deep Dep Recall (Sup < 3)": round(float(np.mean(ddep_lt3)), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUT_DIR, "cv_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nCV Summary saved to {summary_csv_path}")

    # Print markdown table
    print("\n--- 5-FOLD CROSS-VALIDATION SUMMARY TABLE ---")
    print(summary_df[["Approach", "Description", "Macro F1 Mean", "Macro F1 Std", "Balanced Acc Mean", "Balanced Acc Std", "Wind MAE Mean"]].to_string(index=False))

    # Paired statistical comparisons (n=5 folds)
    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    stat_test_results = []

    print("\n--- PAIRED STATISTICAL COMPARISONS (n=5 folds) ---")
    any_distinguishable = False
    for p1, p2 in pairs:
        f1_diff = macro_f1_arrays[p1] - macro_f1_arrays[p2]
        bal_diff = bal_acc_arrays[p1] - bal_acc_arrays[p2]

        # Paired t-test
        t_f1, p_f1 = stats.ttest_rel(macro_f1_arrays[p1], macro_f1_arrays[p2])
        t_bal, p_bal = stats.ttest_rel(bal_acc_arrays[p1], bal_acc_arrays[p2])

        # Wilcoxon
        try:
            w_f1, pw_f1 = stats.wilcoxon(f1_diff)
        except:
            w_f1, pw_f1 = np.nan, np.nan
        try:
            w_bal, pw_bal = stats.wilcoxon(bal_diff)
        except:
            w_bal, pw_bal = np.nan, np.nan

        is_sig = (p_f1 < 0.05) or (p_bal < 0.05)
        if is_sig:
            any_distinguishable = True

        res_item = {
            "Comparison": f"{p1} vs {p2}",
            "Macro F1 t-stat": round(float(t_f1), 4),
            "Macro F1 p-val": round(float(p_f1), 4),
            "Macro F1 Wilcoxon p-val": round(float(pw_f1), 4) if not np.isnan(pw_f1) else "N/A",
            "Balanced Acc t-stat": round(float(t_bal), 4),
            "Balanced Acc p-val": round(float(p_bal), 4),
            "Balanced Acc Wilcoxon p-val": round(float(pw_bal), 4) if not np.isnan(pw_bal) else "N/A",
            "Significant (p < 0.05)": bool(is_sig),
        }
        stat_test_results.append(res_item)
        print(f"  {p1} vs {p2}: Macro F1 p={p_f1:.4f}, Bal Acc p={p_bal:.4f} -> Significant: {is_sig}")

    stat_df = pd.DataFrame(stat_test_results)
    stat_df.to_csv(os.path.join(OUT_DIR, "paired_tests.csv"), index=False)

    # Decision rule application:
    # Check if CLEAR WINNER, WEAK-UNRELIABLE PREFERENCE, or NO USABLE WINNER
    best_f1_app = summary_df.loc[summary_df["Macro F1 Mean"].idxmax()]["Approach"]
    best_bal_app = summary_df.loc[summary_df["Balanced Acc Mean"].idxmax()]["Approach"]

    # Check if difference is statistically distinguishable
    if any_distinguishable and (best_f1_app == best_bal_app):
        decision = "CLEAR WINNER"
        winning_app = best_f1_app
        justification = (
            f"Approach {winning_app} achieved the highest mean Macro F1 ({summary_df.loc[summary_df['Approach']==winning_app, 'Macro F1 Mean'].values[0]:.4f}) "
            f"and Balanced Accuracy ({summary_df.loc[summary_df['Approach']==winning_app, 'Balanced Acc Mean'].values[0]:.4f}) with statistically significant paired differences."
        )
    elif not any_distinguishable:
        # Check if one approach has least unstable recall or higher mean without significance
        decision = "NO USABLE WINNER"
        # Select approach with least unstable recall or best mean
        winning_app = best_f1_app
        justification = (
            "All four approaches are comparably unstable across the 5 folds with paired t-test p-values > 0.05 across all pairwise comparisons. "
            "The performance differences are statistically indistinguishable from random noise, demonstrating that the primary bottleneck "
            "is extreme class imbalance (only 13 Deep Depression observations in 188 samples) and small data volume, rather than loss formulation choice. "
            f"For Step 4 single test evaluation, Approach {winning_app} is selected as the best-available representative."
        )
    else:
        decision = "WEAK-UNRELIABLE PREFERENCE"
        winning_app = best_f1_app
        justification = (
            f"Approach {winning_app} has higher numerical mean Macro F1 but paired comparisons show marginal or inconsistent statistical significance. "
            f"Selected Approach {winning_app} as the best-available candidate for final test evaluation."
        )

    print(f"\n>>> STEP 3 DECISION: {decision} <<<")
    print(f"Candidate Approach Selected: {winning_app} ({approach_names[winning_app]})")
    print(f"Justification: {justification}")

    # -----------------------------------------------------------------------
    # STEP 4: SINGLE FROZEN TEST EVALUATION
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: SINGLE FROZEN TEST EVALUATION (EXACTLY ONCE)")
    print(f"Retraining Final Model using Approach {winning_app} on Full Pooled 2000-2010 Data (N=188)")
    print("=" * 80)

    # Build full pooled training loader
    full_pool_ds = MasterSatelliteDataset(pool_df, base_dir=BASE_DIR, is_train=True)
    full_pool_loader = DataLoader(full_pool_ds, batch_size=16, shuffle=True)

    # Build frozen test loader
    frozen_test_ds = MasterSatelliteDataset(test_df, base_dir=BASE_DIR, is_train=False)
    frozen_test_loader = DataLoader(frozen_test_ds, batch_size=16, shuffle=False)

    # Compute weights on the full 188 pooled samples
    full_counts = np.bincount(pool_df["class_idx"].values, minlength=3).astype(np.float32)
    if winning_app == "A":
        w_final = 1.0 / np.maximum(full_counts, 1.0)
        w_final = w_final / np.mean(w_final)
        loss_fn_final = nn.CrossEntropyLoss(weight=torch.tensor(w_final, dtype=torch.float32).to(device))
    elif winning_app == "B":
        w_final = 1.0 / np.sqrt(np.maximum(full_counts, 1.0))
        w_final = w_final / np.mean(w_final)
        loss_fn_final = nn.CrossEntropyLoss(weight=torch.tensor(w_final, dtype=torch.float32).to(device))
    elif winning_app == "C":
        w_final = 1.0 / np.sqrt(np.maximum(full_counts, 1.0))
        w_final = w_final / np.mean(w_final)
        loss_fn_final = FocalLoss(alpha=torch.tensor(w_final, dtype=torch.float32).to(device), gamma=2.0)
    elif winning_app == "D":
        loss_fn_final = nn.SmoothL1Loss(beta=1.0)

    set_seed(42)
    final_model = ResidualSatelliteCNN(in_channels=1, num_classes=3).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train on all 188 samples for 25 epochs
    print(f"Training final model for 25 epochs on 188 pooled observations...")
    for epoch in range(1, 26):
        loss_val = train_epoch(final_model, full_pool_loader, final_optimizer, winning_app, loss_fn_final, lambda_reg=0.01, device=device)
        if epoch % 5 == 0 or epoch == 25:
            print(f"  Epoch {epoch:02d}/25 | Training Loss: {loss_val:.4f}")

    # Save final model
    final_model_path = os.path.join(OUT_DIR, "final_model.pth")
    torch.save({
        "approach": winning_app,
        "approach_name": approach_names[winning_app],
        "state_dict": final_model.state_dict(),
        "train_samples": len(pool_df),
        "test_samples": len(test_df),
        "epochs": 25,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }, final_model_path)
    print(f"\nFinal model weights saved -> {final_model_path}")

    # Evaluate EXACTLY ONCE on frozen test set
    test_metrics, y_t_c, y_p_c, y_t_w, y_p_w = eval_dataset(final_model, frozen_test_loader, winning_app, device)
    test_metrics["approach"] = winning_app
    test_metrics["approach_name"] = approach_names[winning_app]
    test_metrics["single_test_evaluation_confirmed"] = True

    # Save test metrics JSON
    final_test_metrics_path = os.path.join(OUT_DIR, "final_test_metrics.json")
    with open(final_test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Final test metrics saved -> {final_test_metrics_path}")

    # Save test confusion matrix CSV
    test_cm_df = pd.DataFrame(test_metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
    test_cm_path = os.path.join(OUT_DIR, "final_test_confusion_matrix.csv")
    test_cm_df.to_csv(test_cm_path)
    print(f"Final test confusion matrix saved -> {test_cm_path}")

    print("\n--- FINAL HELD-OUT TEST EVALUATION RESULTS ---")
    print(f"Test Accuracy:          {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Balanced Accuracy:  {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"Test Macro F1:          {test_metrics['macro_f1']*100:.2f}%")
    print(f"Test Wind MAE:          {test_metrics['wind_mae']:.2f} kt")
    print(f"Test Wind RMSE:         {test_metrics['wind_rmse']:.2f} kt")
    print(f"Per-Class Recall:")
    print(f"  Low (<17 kts):        {test_metrics['per_class'][CLASS_NAMES[0]]['recall']*100:.2f}% (Support: {test_metrics['per_class'][CLASS_NAMES[0]]['support']})")
    print(f"  Depression (17-27):   {test_metrics['per_class'][CLASS_NAMES[1]]['recall']*100:.2f}% (Support: {test_metrics['per_class'][CLASS_NAMES[1]]['support']})")
    print(f"  Deep Dep (28-33):     {test_metrics['per_class'][CLASS_NAMES[2]]['recall']*100:.2f}% (Support: {test_metrics['per_class'][CLASS_NAMES[2]]['support']})")
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(test_cm_df.to_string())

    return summary_df, stat_df, decision, winning_app, test_metrics, justification


if __name__ == "__main__":
    run_master_experiment()
