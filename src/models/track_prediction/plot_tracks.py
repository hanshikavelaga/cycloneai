import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


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
print("CycloneAI Track LSTM - Final Visualization")
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
# INTERPOLATE MISSING WIND AND PRESSURE
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


# Remove rows still missing required features
df = df.dropna(
    subset=FEATURES
).copy()


# ============================================================
# CREATE VALID SEQUENCES
# ============================================================

def create_sequences(dataframe):

    X = []
    Y = []
    storm_ids = []
    base_times = []

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
            # Check input interval = 3 hours
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
            # Check target intervals
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
            base_times.append(times[i])

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(storm_ids),
        np.array(base_times)
    )


X, Y, storm_ids, base_times = create_sequences(df)


print()
print("Sequence dataset:")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(
    f"Unique storms: "
    f"{len(np.unique(storm_ids))}"
)


# ============================================================
# RECREATE SAME STORM-LEVEL SPLIT
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
# TEST DATA
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

test_storm_ids = storm_ids[
    test_mask
]

test_base_times = base_times[
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
# SCALE INPUT
# ============================================================

X_test_scaled = feature_scaler.transform(
    X_test.reshape(
        -1,
        len(FEATURES)
    )
).reshape(
    X_test.shape
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
# LOAD TRAINED MODEL
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
# CONVERT TO REAL UNITS
# ============================================================

predictions = target_scaler.inverse_transform(
    predictions_scaled.reshape(
        -1,
        len(FEATURES)
    )
).reshape(
    predictions_scaled.shape
)


# ============================================================
# SELECT ONE TEST STORM
# ============================================================

selected_storm = test_storms[0]

storm_mask = (
    test_storm_ids
    == selected_storm
)

storm_indices = np.where(
    storm_mask
)[0]


if len(storm_indices) == 0:

    raise RuntimeError(
        "No sequences found for selected test storm."
    )


# Use first forecast window for this storm
index = storm_indices[0]


input_sequence = X_test[
    index
]

actual_targets = Y_test[
    index
]

predicted_targets = predictions[
    index
]

base_time = pd.Timestamp(
    test_base_times[index]
)


# ============================================================
# PRINT INFORMATION
# ============================================================

print()
print("=" * 70)
print("SELECTED TEST STORM")
print("=" * 70)

print(
    f"Storm ID: {selected_storm}"
)

print(
    f"Forecast initialization: {base_time} UTC"
)

print()


for i, horizon in enumerate(HORIZONS):

    actual = actual_targets[i]

    predicted = predicted_targets[i]

    print(horizon)

    print(
        f"  Actual    : "
        f"Lat={actual[0]:.2f}, "
        f"Lon={actual[1]:.2f}"
    )

    print(
        f"  Predicted : "
        f"Lat={predicted[0]:.2f}, "
        f"Lon={predicted[1]:.2f}"
    )

    print()


# ============================================================
# CURRENT POSITION
# ============================================================

current_lat = input_sequence[
    -1, 0
]

current_lon = input_sequence[
    -1, 1
]


# ============================================================
# HISTORICAL TRACK
# ============================================================

historical_lat = input_sequence[
    :, 0
]

historical_lon = input_sequence[
    :, 1
]


# ============================================================
# ACTUAL FUTURE TRACK
# ============================================================

actual_lat = actual_targets[
    :, 0
]

actual_lon = actual_targets[
    :, 1
]


# ============================================================
# PREDICTED FUTURE TRACK
# ============================================================

predicted_lat = predicted_targets[
    :, 0
]

predicted_lon = predicted_targets[
    :, 1
]


# ============================================================
# CREATE FINAL VISUALIZATION
# ============================================================

plt.figure(
    figsize=(11, 8)
)


# ------------------------------------------------------------
# Historical trajectory
# ------------------------------------------------------------

plt.plot(
    historical_lon,
    historical_lat,
    marker="o",
    linewidth=2,
    label="Historical track"
)


# ------------------------------------------------------------
# Current position
# ------------------------------------------------------------

plt.scatter(
    current_lon,
    current_lat,
    marker="*",
    s=220,
    label="Current position",
    zorder=5
)


# ------------------------------------------------------------
# Actual future trajectory
# ------------------------------------------------------------

actual_lon_plot = np.concatenate(
    ([current_lon], actual_lon)
)

actual_lat_plot = np.concatenate(
    ([current_lat], actual_lat)
)

plt.plot(
    actual_lon_plot,
    actual_lat_plot,
    marker="o",
    linewidth=2.5,
    label="Actual future track"
)


# ------------------------------------------------------------
# Predicted future trajectory
# ------------------------------------------------------------

predicted_lon_plot = np.concatenate(
    ([current_lon], predicted_lon)
)

predicted_lat_plot = np.concatenate(
    ([current_lat], predicted_lat)
)

plt.plot(
    predicted_lon_plot,
    predicted_lat_plot,
    marker="x",
    linestyle="--",
    linewidth=2.5,
    markersize=8,
    label="LSTM predicted track"
)


# ============================================================
# LABEL ACTUAL POINTS
# ============================================================

for i, horizon in enumerate(HORIZONS):

    plt.annotate(
        f"Actual {horizon}",
        (
            actual_lon[i],
            actual_lat[i]
        ),
        xytext=(7, -12),
        textcoords="offset points",
        fontsize=9
    )


# ============================================================
# LABEL PREDICTED POINTS
# ============================================================

for i, horizon in enumerate(HORIZONS):

    plt.annotate(
        f"Pred {horizon}",
        (
            predicted_lon[i],
            predicted_lat[i]
        ),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=9
    )


# ============================================================
# AXES
# ============================================================

plt.xlabel(
    "Longitude (°)"
)

plt.ylabel(
    "Latitude (°)"
)

plt.title(
    "CycloneAI Track LSTM\n"
    f"Actual vs Predicted — Test Storm {selected_storm}"
)


# ============================================================
# GRID / LEGEND
# ============================================================

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    loc="best"
)


# ============================================================
# SAVE
# ============================================================

results_dir = Path(
    "results"
)

results_dir.mkdir(
    exist_ok=True
)

plot_path = (
    results_dir
    / "track_lstm_final_track_visualization.png"
)

plt.tight_layout()

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

print("=" * 70)

print(
    f"Final visualization saved to:\n"
    f"{plot_path}"
)

print("=" * 70)


plt.show()