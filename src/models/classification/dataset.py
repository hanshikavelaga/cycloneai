"""
CycloneAI Satellite Dataset & DataLoader Module
===============================================
Domain-specific PyTorch Dataset for NOAA HURSAT-B1 2000-2015 Infrared Satellite Imagery.
Enforces zero-leakage group-aware temporal splitting and physically valid data augmentations.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Official IMD Genesis Stage definitions
IMD_CLASSES = [
    'Low Pressure Area (< 17 kts)',
    'Depression (17-27 kts)',
    'Deep Depression (28-33 kts)'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(IMD_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(IMD_CLASSES)}


def get_imd_class(wind_speed_kts):
    """Maps continuous sustained wind speed (kts) to official IMD Genesis Category."""
    if wind_speed_kts < 17.0:
        return 'Low Pressure Area (< 17 kts)'
    elif wind_speed_kts <= 27.0:
        return 'Depression (17-27 kts)'
    else:
        return 'Deep Depression (28-33 kts)'


class CycloneSatelliteDataset(Dataset):
    """
    PyTorch Dataset loading pre-normalized (128, 128) HURSAT Infrared Window tensors.
    """
    def __init__(self, mapping_csv_path, split='train', repo_root=None, augment=False, in_channels=1):
        super().__init__()
        self.split = split.lower()
        self.augment = augment
        self.in_channels = in_channels
        
        if repo_root is None:
            # Assume 3 levels up from this script (src/models/classification -> root)
            self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        else:
            self.repo_root = repo_root
            
        df = pd.read_csv(mapping_csv_path)
        # Filter strictly for matched, label-verified observations
        matched_df = df[df['matched'] == True].copy()
        
        # Zero-leakage temporal group-aware splits (Strategy B - Approved)
        if self.split == 'train':
            # Seasons 2000 to 2008 (67.5% of data, 75 unique cyclones)
            self.data = matched_df[(matched_df['year'] >= 2000) & (matched_df['year'] <= 2008)].copy()
        elif self.split == 'val':
            # Seasons 2009 to 2010 (12.8% of data, 15 unique cyclones)
            self.data = matched_df[(matched_df['year'] >= 2009) & (matched_df['year'] <= 2010)].copy()
        elif self.split == 'test':
            # Seasons 2011 to 2015 (19.7% of data, 23 unique cyclones across 4 seasons)
            self.data = matched_df[(matched_df['year'] >= 2011) & (matched_df['year'] <= 2015)].copy()
        elif self.split == 'all':
            self.data = matched_df.copy()
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', 'test', or 'all'.")
            
        self.data.reset_index(drop=True, inplace=True)
        self.data['class_name'] = self.data['intensity_knots'].apply(get_imd_class)
        self.data['class_idx'] = self.data['class_name'].map(CLASS_TO_IDX)

    def __len__(self):
        return len(self.data)

    def _apply_augmentation(self, image_np):
        """
        Physically valid rotations for top-down satellite imagery.
        Preserves counter-clockwise cyclonic rotation chirality in the Northern Hemisphere.
        Flips (fliplr/flipud) are intentionally excluded as they invert storm chirality.
        """
        # Random 90-degree rotations (0, 90, 180, 270 deg)
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            image_np = np.rot90(image_np, k=rot_k)
        return image_np.copy()

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.repo_root, row['image_path'])
        
        # Load pre-normalized numpy tensor of shape (128, 128), float32 in [0, 1]
        image = np.load(image_path).astype(np.float32)
        
        if self.augment:
            image = self._apply_augmentation(image)
            
        # Reshape to (C, H, W)
        if self.in_channels == 1:
            tensor_img = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 128, 128)
        elif self.in_channels == 3:
            # Repeat IR band across 3 channels for composite compatibility
            tensor_img = torch.tensor(image, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)  # (3, 128, 128)
        else:
            raise ValueError(f"Unsupported in_channels: {self.in_channels}")
            
        # Target 1: Continuous wind speed in knots
        wind_speed = torch.tensor(row['intensity_knots'], dtype=torch.float32)
        
        # Target 2: Discrete IMD Genesis Stage
        class_idx = torch.tensor(row['class_idx'], dtype=torch.long)
        
        meta_info = {
            'filename': row['filename'],
            'cyclone_id': row['cyclone_id'],
            'cyclone_name': row['cyclone_name'],
            'year': int(row['year']),
            'intensity_knots': float(row['intensity_knots']),
            'class_name': row['class_name']
        }
        
        return tensor_img, wind_speed, class_idx, meta_info


def get_dataloaders(mapping_csv_path, batch_size=16, repo_root=None, in_channels=1, num_workers=0):
    """Utility to build reproducible train, val, and test DataLoaders."""
    train_dataset = CycloneSatelliteDataset(mapping_csv_path, split='train', repo_root=repo_root, augment=True, in_channels=in_channels)
    val_dataset = CycloneSatelliteDataset(mapping_csv_path, split='val', repo_root=repo_root, augment=False, in_channels=in_channels)
    test_dataset = CycloneSatelliteDataset(mapping_csv_path, split='test', repo_root=repo_root, augment=False, in_channels=in_channels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    import sys
    csv_path = r"C:\Users\hasin\Desktop\cycloneai\data\processed\image_metadata_mapping.csv"
    tr_loader, v_loader, te_loader, tr_ds, v_ds, te_ds = get_dataloaders(csv_path, batch_size=8)
    print(f"Dataset verification:")
    print(f"  Train: {len(tr_ds)} samples | Batches: {len(tr_loader)}")
    print(f"  Val:   {len(v_ds)} samples   | Batches: {len(v_loader)}")
    print(f"  Test:  {len(te_ds)} samples  | Batches: {len(te_loader)}")
    
    # Inspect a sample batch
    imgs, winds, cats, metas = next(iter(tr_loader))
    print(f"Batch shape: {imgs.shape}")
    print(f"Winds shape: {winds.shape} (Sample: {winds[:3].tolist()} kts)")
    print(f"Cats shape:  {cats.shape} (Sample: {cats[:3].tolist()})")
    print("Dataset module successfully verified!")
