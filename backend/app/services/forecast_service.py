import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATHS
# ============================================================

# forecast_service.py
#   CycloneAI/
#     backend/
#       app/
#         services/
#
# parents[3] = CycloneAI project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "models" / "track_prediction"

MODEL_PATH = MODEL_DIR / "track_lstm_best.pth"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.pkl"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 1
INPUT_STEPS = 4
OUTPUT_STEPS = 4

FEATURES = [
    "latitude",
    "longitude",
    "wind_speed",
    "pressure",
]

FORECAST_OFFSETS = [6, 12, 24, 48]


# ============================================================
# EXACT MODEL ARCHITECTURE USED DURING EVALUATION
# ============================================================

class TrackLSTM(nn.Module):
    """
    Exact TrackLSTM architecture used by evaluate.py.

    Input:
        (batch, 4 observations, 4 features)

    Features:
        [latitude, longitude, wind_speed, pressure]

    Output:
        (batch, 4 forecast steps, 4 features)
    """

    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        num_layers=1,
        output_steps=4,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_size,
            output_steps * input_size,
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
            self.input_size,
        )

        return prediction


# ============================================================
# FORECAST SERVICE
# ============================================================

class ForecastService:

    def __init__(self):

        self.device = torch.device("cpu")

        self.model = None
        self.feature_scaler = None
        self.target_scaler = None

        self.model_loaded = False
        self.scalers_loaded = False

        # ------------------------------------------------------
        # LOAD FEATURE SCALER
        # ------------------------------------------------------

        if FEATURE_SCALER_PATH.exists():

            try:

                self.feature_scaler = joblib.load(
                    FEATURE_SCALER_PATH
                )

                print(
                    "ForecastService: Loaded feature scaler from "
                    f"{FEATURE_SCALER_PATH}"
                )

            except Exception as e:

                print(
                    "ForecastService: Error loading feature scaler: "
                    f"{e}"
                )

        else:

            print(
                "ForecastService: Warning: Feature scaler not found at "
                f"{FEATURE_SCALER_PATH}"
            )

        # ------------------------------------------------------
        # LOAD TARGET SCALER
        # ------------------------------------------------------

        if TARGET_SCALER_PATH.exists():

            try:

                self.target_scaler = joblib.load(
                    TARGET_SCALER_PATH
                )

                print(
                    "ForecastService: Loaded target scaler from "
                    f"{TARGET_SCALER_PATH}"
                )

            except Exception as e:

                print(
                    "ForecastService: Error loading target scaler: "
                    f"{e}"
                )

        else:

            print(
                "ForecastService: Warning: Target scaler not found at "
                f"{TARGET_SCALER_PATH}"
            )

        # ------------------------------------------------------
        # LOAD BEST MODEL CHECKPOINT
        # ------------------------------------------------------

        if MODEL_PATH.exists():

            try:

                checkpoint = torch.load(
                    MODEL_PATH,
                    map_location=self.device,
                    weights_only=False,
                )

                # The training/evaluation code stores the trained
                # weights under "model_state_dict".
                if (
                    isinstance(checkpoint, dict)
                    and "model_state_dict" in checkpoint
                ):

                    state_dict = checkpoint["model_state_dict"]

                    input_size = checkpoint.get(
                        "input_size",
                        INPUT_SIZE,
                    )

                    hidden_size = checkpoint.get(
                        "hidden_size",
                        HIDDEN_SIZE,
                    )

                else:

                    # Fallback for a raw state_dict checkpoint.
                    state_dict = checkpoint

                    input_size = INPUT_SIZE
                    hidden_size = HIDDEN_SIZE

                # Recreate EXACT evaluation architecture.
                self.model = TrackLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=NUM_LAYERS,
                    output_steps=OUTPUT_STEPS,
                ).to(self.device)

                self.model.load_state_dict(
                    state_dict
                )

                self.model.eval()

                self.model_loaded = True

                print(
                    "ForecastService: Loaded best Track LSTM model from "
                    f"{MODEL_PATH}"
                )

            except Exception as e:

                print(
                    "ForecastService: Error loading model weights: "
                    f"{e}"
                )

        else:

            print(
                "ForecastService: Warning: Model file not found at "
                f"{MODEL_PATH}"
            )

        # ------------------------------------------------------
        # FINAL READINESS CHECK
        # ------------------------------------------------------

        self.scalers_loaded = (
            self.feature_scaler is not None
            and self.target_scaler is not None
        )

        if self.model_loaded and self.scalers_loaded:

            print(
                "ForecastService: Track LSTM inference pipeline READY."
            )

        else:

            print(
                "ForecastService: WARNING - Track LSTM inference "
                "pipeline is not fully ready."
            )

    # ==========================================================
    # FORECAST TRACK
    # ==========================================================

    def forecast_track(
        self,
        historical_observations: list,
    ):
        """
        Generate +6h, +12h, +24h and +48h forecasts.

        The preprocessing exactly follows the evaluation pipeline:

            raw input
                ↓
            feature StandardScaler
                ↓
            TrackLSTM
                ↓
            target StandardScaler inverse transform
                ↓
            forecast values
        """

        if not historical_observations:

            return []

        if not self.model_loaded:

            print(
                "ForecastService: Model is not loaded."
            )

            return []

        if not self.scalers_loaded:

            print(
                "ForecastService: Scalers are not loaded."
            )

            return []

        try:

            # --------------------------------------------------
            # SORT OBSERVATIONS
            # --------------------------------------------------

            sorted_obs = sorted(
                historical_observations,
                key=lambda o: o["timestamp"],
            )

            if not sorted_obs:

                return []

            # --------------------------------------------------
            # TAKE LAST 4 OBSERVATIONS
            # --------------------------------------------------

            features = []

            for obs in sorted_obs[-INPUT_STEPS:]:

                features.append(
                    [
                        float(obs["latitude"]),
                        float(obs["longitude"]),
                        float(obs["wind_speed"]),
                        float(obs["pressure"]),
                    ]
                )

            # --------------------------------------------------
            # PAD IF FEWER THAN 4 OBSERVATIONS
            # --------------------------------------------------

            while len(features) < INPUT_STEPS:

                features.insert(
                    0,
                    features[0],
                )

            # --------------------------------------------------
            # CONVERT TO NUMPY
            # --------------------------------------------------

            input_array = np.asarray(
                features,
                dtype=np.float32,
            )

            # Shape:
            # (4, 4)

            # --------------------------------------------------
            # APPLY SAME FEATURE SCALER AS TRAINING
            # --------------------------------------------------

            input_scaled = self.feature_scaler.transform(
                input_array
            )

            # Shape remains:
            # (4, 4)

            # --------------------------------------------------
            # CREATE MODEL INPUT
            # --------------------------------------------------

            input_tensor = torch.tensor(
                input_scaled,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            # Shape:
            # (1, 4, 4)

            # --------------------------------------------------
            # MODEL INFERENCE
            # --------------------------------------------------

            with torch.no_grad():

                prediction_tensor = self.model(
                    input_tensor
                )

            # --------------------------------------------------
            # CONVERT MODEL OUTPUT TO NUMPY
            # --------------------------------------------------

            predictions_scaled = (
                prediction_tensor
                .detach()
                .cpu()
                .numpy()
            )

            # Shape:
            # (1, 4, 4)

            # --------------------------------------------------
            # INVERSE TARGET SCALING
            # --------------------------------------------------

            prediction_shape = predictions_scaled.shape

            predictions = self.target_scaler.inverse_transform(
                predictions_scaled.reshape(
                    -1,
                    INPUT_SIZE,
                )
            ).reshape(
                prediction_shape
            )

            # Remove batch dimension.
            predictions = predictions[0]

            # Shape:
            # (4, 4)

            # --------------------------------------------------
            # BUILD FORECAST RESPONSE
            # --------------------------------------------------

            latest_obs = sorted_obs[-1]

            base_time = latest_obs["timestamp"]

            forecast_results = []

            for idx, hours in enumerate(
                FORECAST_OFFSETS
            ):

                # Native Python floats.
                pred_lat = float(
                    predictions[idx, 0]
                )

                pred_lon = float(
                    predictions[idx, 1]
                )

                pred_wind = float(
                    predictions[idx, 2]
                )

                pred_pressure = float(
                    predictions[idx, 3]
                )

                # ------------------------------------------------
                # Operational NIO display bounds.
                # ------------------------------------------------

                pred_lat = max(
                    -5.0,
                    min(35.0, pred_lat),
                )

                pred_lon = max(
                    40.0,
                    min(110.0, pred_lon),
                )

                # Keep intensity values physically reasonable
                # for dashboard display.
                pred_wind = max(
                    0.0,
                    min(200.0, pred_wind),
                )

                pred_pressure = max(
                    850.0,
                    min(1050.0, pred_pressure),
                )

                forecast_time = (
                    base_time
                    + datetime.timedelta(
                        hours=hours
                    )
                )

                # Horizon-based confidence indicator.
                # This is NOT a calibrated probability from the
                # LSTM model itself.
                confidence = max(
                    0.40,
                    0.95 - (hours * 0.01),
                )

                forecast_results.append(
                    {
                        "forecast_time": forecast_time,
                        "latitude": float(
                            round(
                                pred_lat,
                                2,
                            )
                        ),
                        "longitude": float(
                            round(
                                pred_lon,
                                2,
                            )
                        ),
                        "wind_speed": float(
                            round(
                                pred_wind,
                                1,
                            )
                        ),
                        "pressure": float(
                            round(
                                pred_pressure,
                                1,
                            )
                        ),
                        "confidence": float(
                            round(
                                confidence,
                                2,
                            )
                        ),
                    }
                )

            return forecast_results

        except Exception as e:

            print(
                "ForecastService: Error generating forecast: "
                f"{e}"
            )

            return []


# ============================================================
# SINGLETON
# ============================================================

forecast_service = ForecastService()