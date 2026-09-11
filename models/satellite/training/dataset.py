"""
Dataset and DataLoader implementation for Satellite-only Cyclone Intensity Estimation.
Works strictly on verified HURSAT-B1 128x128 satellite imagery.
"""

import os
import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SatelliteDataset(Dataset):
    """
    PyTorch Dataset for verified HURSAT satellite infrared arrays (128x128 float32).
    Applies data augmentations (flips and 90-deg rotations) only when is_train=True.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: str = r"C:\Users\hasin\Desktop\cycloneai",
        is_train: bool = False,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.is_train = is_train
        self.augment = augment and is_train

    def __len__(self) -> int:
        return len(self.df)

    def _apply_augmentation(self, arr: np.ndarray) -> np.ndarray:
        """Applies random horizontal flip, vertical flip, and 90-deg rotations."""
        if random.random() > 0.5:
            arr = np.fliplr(arr)
        if random.random() > 0.5:
            arr = np.flipud(arr)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            arr = np.rot90(arr, k=k)
        return arr.copy()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]

        # Resolve path
        if os.path.isabs(img_rel_path):
            img_path = img_rel_path
        else:
            img_path = os.path.join(self.base_dir, img_rel_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Satellite image not found: {img_path}")

        # Load array
        arr = np.load(img_path).astype(np.float32)

        if arr.ndim == 2:
            if self.augment:
                arr = self._apply_augmentation(arr)
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 128, 128)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            if self.augment:
                arr = self._apply_augmentation(arr[0])
                tensor = torch.from_numpy(arr).unsqueeze(0)
            else:
                tensor = torch.from_numpy(arr)
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape} in {img_path}")

        # Target wind speed in knots
        target = torch.tensor(row["intensity_knots"], dtype=torch.float32)

        meta = {
            "filename": row.get("filename", os.path.basename(img_path)),
            "cyclone_id": str(row.get("cyclone_id", "")),
            "cyclone_name": str(row.get("cyclone_name", "")),
            "year": int(row.get("year", 0)),
            "timestamp_utc": str(row.get("timestamp_utc", "")),
            "latitude": float(row.get("latitude", 0.0)),
            "longitude": float(row.get("longitude", 0.0)),
            "imd_category": str(row.get("imd_category", "")),
            "split": str(row.get("split", "")),
        }

        return tensor, target, meta


def get_dataloaders(
    manifest_path: str = r"models\satellite\dataset_audit\verified_satellite_manifest.csv",
    base_dir: str = r"C:\Users\hasin\Desktop\cycloneai",
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    Creates train, val, and test DataLoaders strictly adhering to the chronological split:
      - Train: 2000-2008 (158 obs, 75 cyclones)
      - Val:   2009-2010 (30 obs, 15 cyclones)
      - Test:  2011-2015 (46 obs, 23 cyclones)
    """
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(base_dir, manifest_path)

    df = pd.read_csv(manifest_path)

    # Split masks
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Integrity assertions
    assert len(train_df) == 158, f"Expected 158 train observations, got {len(train_df)}"
    assert len(val_df) == 30, f"Expected 30 val observations, got {len(val_df)}"
    assert len(test_df) == 46, f"Expected 46 test observations, got {len(test_df)}"
    assert len(df) == 234, f"Expected 234 total observations, got {len(df)}"

    train_cyclones = set(train_df["cyclone_id"].unique())
    val_cyclones = set(val_df["cyclone_id"].unique())
    test_cyclones = set(test_df["cyclone_id"].unique())

    assert len(train_cyclones) == 75, f"Expected 75 train cyclones, got {len(train_cyclones)}"
    assert len(val_cyclones) == 15, f"Expected 15 val cyclones, got {len(val_cyclones)}"
    assert len(test_cyclones) == 23, f"Expected 23 test cyclones, got {len(test_cyclones)}"

    assert len(train_cyclones.intersection(val_cyclones)) == 0, "Train-Val cyclone overlap detected!"
    assert len(train_cyclones.intersection(test_cyclones)) == 0, "Train-Test cyclone overlap detected!"
    assert len(val_cyclones.intersection(test_cyclones)) == 0, "Val-Test cyclone overlap detected!"

    # Datasets
    train_dataset = SatelliteDataset(train_df, base_dir=base_dir, is_train=True, augment=True)
    val_dataset = SatelliteDataset(val_df, base_dir=base_dir, is_train=False, augment=False)
    test_dataset = SatelliteDataset(test_df, base_dir=base_dir, is_train=False, augment=False)

    # Deterministic DataLoader generator
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, df
