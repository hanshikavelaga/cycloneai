import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    r"C:\Users\hansh\Downloads\CycloneAI_Data\Team1_IBTrACS\processed\ibtracs_ni_clean_v2.csv"
)

MODEL_PATH = Path(
    "models/track_prediction/track_lstm_best.pth"
)

FEATURE_SCALER_PATH = Path(
    "models/track_prediction/feature_scaler.pkl"
)

TARGET_SCALER_PATH = Path(
    "models/track_prediction/target_scaler.pkl"
)

INPUT_STEPS = 4

# 3-hourly observations:
# +6h  = 2 steps
# +12h = 4 steps
# +24h = 8 steps
# +48h = 16 steps

TARGET_STEPS = [2, 4, 8, 16]

FEATURES = [
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
]

HORIZONS = [
    "+6h",
    "+12h",
    "+24h",
    "+48h",
]

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 60)
print("CycloneAI Track LSTM Evaluation")
print("=" * 60)
print("Device: CPU")
print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

# Keep only 2000–2015
df = df[
    (df["ISO_TIME"].dt.year >= 2000)
    & (df["ISO_TIME"].dt.year <= 2015)
].copy()

df = df.sort_values(
    ["SID", "ISO_TIME"]
).reset_index(drop=True)

print(f"Rows: {len(df):,}")
print(f"Storms: {df['SID'].nunique()}")


# ============================================================
# INTERPOLATE MISSING VALUES
# ============================================================

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


# Remove rows where required features are still missing
df = df.dropna(
    subset=FEATURES
).copy()


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(dataframe):

    X = []
    Y = []
    storm_ids = []

    for sid, storm in dataframe.groupby("SID"):

        storm = storm.sort_values(
            "ISO_TIME"
        ).reset_index(drop=True)

        values = storm[
            FEATURES
        ].values.astype(np.float32)

        times = storm[
            "ISO_TIME"
        ].tolist()

        max_target = max(TARGET_STEPS)

        for i in range(
            INPUT_STEPS - 1,
            len(values) - max_target
        ):

            # ------------------------------------------------
            # Check that input observations are exactly 3 hours
            # apart
            # ------------------------------------------------

            input_times = times[
                i - INPUT_STEPS + 1 : i + 1
            ]

            input_diffs = np.array([
                (
                    input_times[j + 1]
                    - input_times[j]
                ).total_seconds() / 3600
                for j in range(
                    len(input_times) - 1
                )
            ])

            if not np.allclose(
                input_diffs,
                3.0
            ):
                continue

            # ------------------------------------------------
            # Check forecast targets
            # ------------------------------------------------

            valid_targets = True

            for step in TARGET_STEPS:

                target_diff = (
                    times[i + step]
                    - times[i]
                ).total_seconds() / 3600

                expected_hours = step * 3

                if target_diff != expected_hours:
                    valid_targets = False
                    break

            if not valid_targets:
                continue

            # ------------------------------------------------
            # Input sequence
            # ------------------------------------------------

            x = values[
                i - INPUT_STEPS + 1 : i + 1
            ]

            # ------------------------------------------------
            # Future targets
            # ------------------------------------------------

            y = np.stack([
                values[i + step]
                for step in TARGET_STEPS
            ])

            X.append(x)
            Y.append(y)
            storm_ids.append(sid)

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(storm_ids)
    )


X, Y, storm_ids = create_sequences(df)

print()
print("Sequence dataset:")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(
    f"Unique storms in sequences: "
    f"{len(np.unique(storm_ids))}"
)


# ============================================================
# RECREATE TRAIN / VALIDATION / TEST STORM SPLIT
# ============================================================

unique_storms = np.unique(
    storm_ids
)

rng = np.random.default_rng(
    RANDOM_SEED
)

rng.shuffle(
    unique_storms
)

n_storms = len(
    unique_storms
)

n_train = int(
    n_storms * 0.70
)

n_val = int(
    n_storms * 0.15
)

train_storms = unique_storms[
    :n_train
]

val_storms = unique_storms[
    n_train : n_train + n_val
]

test_storms = unique_storms[
    n_train + n_val :
]


# ============================================================
# TEST MASK
# ============================================================

test_mask = np.isin(
    storm_ids,
    test_storms
)

X_test = X[
    test_mask
]

Y_test = Y[
    test_mask
]

test_sequence_storms = storm_ids[
    test_mask
]


print()
print("Test set:")
print(
    f"Test storms: {len(test_storms)}"
)

print(
    f"Test sequences: {len(X_test)}"
)


# ============================================================
# LOAD SCALERS
# ============================================================

print()
print("Loading scalers...")

feature_scaler = joblib.load(
    FEATURE_SCALER_PATH
)

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)


# ============================================================
# SCALE TEST INPUT
# ============================================================

original_test_shape = X_test.shape

X_test_scaled = feature_scaler.transform(
    X_test.reshape(
        -1,
        len(FEATURES)
    )
).reshape(
    original_test_shape
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

        prediction = self.fc(
            last_hidden
        )

        prediction = prediction.view(
            -1,
            self.output_steps,
            self.input_size
        )

        return prediction


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = TrackLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=1,
    output_steps=len(TARGET_STEPS)
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("Generating predictions...")

X_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
).to(device)


with torch.no_grad():

    predictions_scaled = model(
        X_tensor
    ).cpu().numpy()


# ============================================================
# CONVERT PREDICTIONS BACK TO REAL UNITS
# ============================================================

prediction_shape = predictions_scaled.shape

predictions = target_scaler.inverse_transform(
    predictions_scaled.reshape(
        -1,
        len(FEATURES)
    )
).reshape(
    prediction_shape
)


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 80)
print("TRACK LSTM TEST RESULTS")
print("=" * 80)


results = []


for horizon_index, horizon in enumerate(HORIZONS):

    actual = Y_test[
        :,
        horizon_index,
        :
    ]

    predicted = predictions[
        :,
        horizon_index,
        :
    ]

    # --------------------------------------------------------
    # Latitude
    # --------------------------------------------------------

    lat_mae = mean_absolute_error(
        actual[:, 0],
        predicted[:, 0]
    )

    lat_rmse = np.sqrt(
        mean_squared_error(
            actual[:, 0],
            predicted[:, 0]
        )
    )

    # --------------------------------------------------------
    # Longitude
    # --------------------------------------------------------

    lon_mae = mean_absolute_error(
        actual[:, 1],
        predicted[:, 1]
    )

    lon_rmse = np.sqrt(
        mean_squared_error(
            actual[:, 1],
            predicted[:, 1]
        )
    )

    # --------------------------------------------------------
    # Wind
    # --------------------------------------------------------

    wind_mae = mean_absolute_error(
        actual[:, 2],
        predicted[:, 2]
    )

    wind_rmse = np.sqrt(
        mean_squared_error(
            actual[:, 2],
            predicted[:, 2]
        )
    )

    # --------------------------------------------------------
    # Pressure
    # --------------------------------------------------------

    pressure_mae = mean_absolute_error(
        actual[:, 3],
        predicted[:, 3]
    )

    pressure_rmse = np.sqrt(
        mean_squared_error(
            actual[:, 3],
            predicted[:, 3]
        )
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append(
        {
            "horizon": horizon,

            "latitude_mae_deg": lat_mae,
            "latitude_rmse_deg": lat_rmse,

            "longitude_mae_deg": lon_mae,
            "longitude_rmse_deg": lon_rmse,

            "wind_mae_knots": wind_mae,
            "wind_rmse_knots": wind_rmse,

            "pressure_mae_hpa": pressure_mae,
            "pressure_rmse_hpa": pressure_rmse,
        }
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(horizon)

    print(
        f"  Latitude  MAE : "
        f"{lat_mae:.4f}°"
    )

    print(
        f"  Latitude  RMSE: "
        f"{lat_rmse:.4f}°"
    )

    print(
        f"  Longitude MAE : "
        f"{lon_mae:.4f}°"
    )

    print(
        f"  Longitude RMSE: "
        f"{lon_rmse:.4f}°"
    )

    print(
        f"  Wind      MAE : "
        f"{wind_mae:.2f} knots"
    )

    print(
        f"  Wind      RMSE: "
        f"{wind_rmse:.2f} knots"
    )

    print(
        f"  Pressure  MAE : "
        f"{pressure_mae:.2f} hPa"
    )

    print(
        f"  Pressure  RMSE: "
        f"{pressure_rmse:.2f} hPa"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_dir = Path(
    "results"
)

results_dir.mkdir(
    exist_ok=True
)

results_df = pd.DataFrame(
    results
)

results_path = (
    results_dir
    / "track_lstm_metrics.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 80)
print(
    f"Metrics saved to: {results_path}"
)
print("=" * 80)