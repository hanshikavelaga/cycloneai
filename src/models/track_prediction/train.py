import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    r"C:\Users\hansh\Downloads\CycloneAI_Data\Team1_IBTrACS\processed\ibtracs_ni_clean_v2.csv"
)

MODEL_DIR = Path("models/track_prediction")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

INPUT_STEPS = 4
# 4 observations × 3 hours = 12-hour history

TARGET_STEPS = [2, 4, 8, 16]
# +6h, +12h, +24h, +48h

FEATURES = [
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CycloneAI Track LSTM Training")
print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading IBTrACS dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original rows: {len(df):,}")


# ============================================================
# FILTER 2000–2015
# ============================================================

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

df = df[
    (df["ISO_TIME"].dt.year >= 2000)
    & (df["ISO_TIME"].dt.year <= 2015)
].copy()

print(f"Rows after 2000–2015 filter: {len(df):,}")
print(f"Storms: {df['SID'].nunique()}")


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["SID", "ISO_TIME"]
).reset_index(drop=True)


# ============================================================
# CHECK 3-HOUR INTERVAL
# ============================================================

df["dt_hours"] = (
    df.groupby("SID")["ISO_TIME"]
    .diff()
    .dt.total_seconds()
    / 3600
)

print("\nObservation interval:")
print(df["dt_hours"].value_counts().head())

# Remove helper column
df.drop(columns=["dt_hours"], inplace=True)


# ============================================================
# INTERPOLATE MISSING WIND/PRESSURE
# ============================================================

print("\nMissing values BEFORE interpolation:")

print(
    df[["WMO_WIND", "WMO_PRES"]]
    .isna()
    .sum()
)


# Interpolate only inside each cyclone.
# limit_area="inside" prevents filling values
# before the first or after the last valid observation.

for column in ["WMO_WIND", "WMO_PRES"]:

    df[column] = (
        df.groupby("SID")[column]
        .transform(
            lambda s: s.interpolate(
                method="linear",
                limit_area="inside"
            )
        )
    )


print("\nMissing values AFTER interpolation:")

print(
    df[["WMO_WIND", "WMO_PRES"]]
    .isna()
    .sum()
)


# ============================================================
# KEEP ONLY ROWS WITH COMPLETE FEATURES
# ============================================================

df = df.dropna(
    subset=FEATURES
).copy()

print(
    f"\nRows available after interpolation: {len(df):,}"
)


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(dataframe):

    X = []
    Y = []
    storm_ids = []

    for sid, storm in dataframe.groupby("SID"):

        storm = storm.sort_values("ISO_TIME").reset_index(
            drop=True
        )

        values = storm[FEATURES].values.astype(np.float32)

        # Need enough observations for:
        # input window + 48-hour target
        max_target = max(TARGET_STEPS)

        for i in range(
            INPUT_STEPS - 1,
            len(values) - max_target
        ):

            # Input:
            # last 4 observations = 12 hours
            x = values[
                i - INPUT_STEPS + 1 : i + 1
            ]

            # Targets:
            # +6, +12, +24, +48 hours
            y = np.stack(
                [
                    values[i + step]
                    for step in TARGET_STEPS
                ]
            )

            X.append(x)
            Y.append(y)
            storm_ids.append(sid)

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(storm_ids)
    )


X, Y, sequence_storms = create_sequences(df)

print("\nSequence dataset:")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Unique storms in sequences: {len(np.unique(sequence_storms))}")


# ============================================================
# STORM-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

unique_storms = np.unique(sequence_storms)

rng = np.random.default_rng(RANDOM_SEED)

rng.shuffle(unique_storms)

n_storms = len(unique_storms)

n_train = int(n_storms * 0.70)
n_val = int(n_storms * 0.15)

train_storms = unique_storms[:n_train]

val_storms = unique_storms[
    n_train : n_train + n_val
]

test_storms = unique_storms[
    n_train + n_val :
]


train_mask = np.isin(
    sequence_storms,
    train_storms
)

val_mask = np.isin(
    sequence_storms,
    val_storms
)

test_mask = np.isin(
    sequence_storms,
    test_storms
)


X_train = X[train_mask]
Y_train = Y[train_mask]

X_val = X[val_mask]
Y_val = Y[val_mask]

X_test = X[test_mask]
Y_test = Y[test_mask]


print("\nStorm split:")
print(f"Train storms: {len(train_storms)}")
print(f"Validation storms: {len(val_storms)}")
print(f"Test storms: {len(test_storms)}")

print("\nSequence split:")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")


# ============================================================
# NORMALIZATION
# ============================================================

# IMPORTANT:
# Fit scalers ONLY on training data.

feature_scaler = StandardScaler()

target_scaler = StandardScaler()


# Flatten training features
train_flat = X_train.reshape(
    -1,
    len(FEATURES)
)

feature_scaler.fit(train_flat)


# Targets are also normalized using training data
target_flat = Y_train.reshape(
    -1,
    len(FEATURES)
)

target_scaler.fit(target_flat)


def scale_X(X_data):

    shape = X_data.shape

    scaled = feature_scaler.transform(
        X_data.reshape(-1, len(FEATURES))
    )

    return scaled.reshape(shape)


def scale_Y(Y_data):

    shape = Y_data.shape

    scaled = target_scaler.transform(
        Y_data.reshape(-1, len(FEATURES))
    )

    return scaled.reshape(shape)


X_train = scale_X(X_train)
X_val = scale_X(X_val)
X_test = scale_X(X_test)

Y_train = scale_Y(Y_train)
Y_val = scale_Y(Y_val)
Y_test = scale_Y(Y_test)


# ============================================================
# PYTORCH DATASET
# ============================================================

class CycloneTrackDataset(Dataset):

    def __init__(self, X_data, Y_data):

        self.X = torch.tensor(
            X_data,
            dtype=torch.float32
        )

        self.Y = torch.tensor(
            Y_data,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        return self.X[index], self.Y[index]


train_dataset = CycloneTrackDataset(
    X_train,
    Y_train
)

val_dataset = CycloneTrackDataset(
    X_val,
    Y_val
)

test_dataset = CycloneTrackDataset(
    X_test,
    Y_test
)


# ============================================================
# DATALOADERS
# ============================================================

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LSTM MODEL
# ============================================================

class TrackLSTM(nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        num_layers=1,
        output_steps=4
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_steps * input_size
        )

        self.output_steps = output_steps
        self.input_size = input_size

    def forward(self, x):

        output, _ = self.lstm(x)

        last_hidden = output[:, -1, :]

        prediction = self.fc(last_hidden)

        prediction = prediction.view(
            -1,
            self.output_steps,
            self.input_size
        )

        return prediction


model = TrackLSTM().to(device)


print("\nModel:")
print(model)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 50

best_val_loss = float("inf")


print("\nStarting training...")
print("=" * 60)


for epoch in range(1, EPOCHS + 1):

    # -----------------------------
    # TRAIN
    # -----------------------------

    model.train()

    train_loss = 0.0

    for batch_X, batch_Y in train_loader:

        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = criterion(
            predictions,
            batch_Y
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() * batch_X.size(0)
        )

    train_loss /= len(train_dataset)


    # -----------------------------
    # VALIDATION
    # -----------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch_X, batch_Y in val_loader:

            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_Y
            )

            val_loss += (
                loss.item() * batch_X.size(0)
            )

    val_loss /= len(val_dataset)


    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| Train Loss: {train_loss:.6f} "
        f"| Val Loss: {val_loss:.6f}"
    )


    # -----------------------------
    # SAVE BEST MODEL
    # -----------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": 4,
                "hidden_size": 64,
                "input_steps": INPUT_STEPS,
                "target_steps": TARGET_STEPS,
                "features": FEATURES,
            },
            MODEL_DIR / "track_lstm_best.pth"
        )


print("\nTraining complete.")
print(f"Best validation loss: {best_val_loss:.6f}")


# ============================================================
# SAVE SCALERS
# ============================================================

import joblib

joblib.dump(
    feature_scaler,
    MODEL_DIR / "feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    MODEL_DIR / "target_scaler.pkl"
)

print("\nSaved:")
print(MODEL_DIR / "track_lstm_best.pth")
print(MODEL_DIR / "feature_scaler.pkl")
print(MODEL_DIR / "target_scaler.pkl")