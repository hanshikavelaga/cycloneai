from pathlib import Path

import torch

from src.models.genesis.model import CycloneGenesisLSTM


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "genesis"
    / "genesis_lstm.pth"
)


# ---------------------------------------------------------
# Genesis Service
# ---------------------------------------------------------

class GenesisService:

    def __init__(self):

        # Create the exact architecture used by the
        # trained genesis_lstm.pth checkpoint.
        self.model = CycloneGenesisLSTM(
            input_size=7,
            hidden_size=32,
            num_layers=1
        )

        # -------------------------------------------------
        # Load trained model
        # -------------------------------------------------

        if MODEL_PATH.exists():

            try:

                checkpoint = torch.load(
                    MODEL_PATH,
                    map_location=torch.device("cpu")
                )

                # genesis_lstm.pth is a raw state_dict
                # (OrderedDict), NOT:
                # {"model_state_dict": ...}
                self.model.load_state_dict(checkpoint)

                self.model.eval()

                print(
                    f"GenesisService: Loaded model weights from "
                    f"{MODEL_PATH}"
                )

            except Exception as e:

                print(
                    f"GenesisService: Error loading weights: {e}"
                )

        else:

            print(
                f"GenesisService: Warning: Weight file not found "
                f"at {MODEL_PATH}. Running uninitialized."
            )

        self.model.eval()


    # ---------------------------------------------------------
    # Genesis Prediction
    # ---------------------------------------------------------

    def predict_genesis(self, historical_observations: list):

        """
        Takes historical cyclone observations and runs the
        Genesis LSTM model.

        Current model expects 7 features:

        1. latitude
        2. longitude
        3. wind_speed
        4. pressure
        5. SST
        6. humidity
        7. convection

        IMPORTANT:
        The current IBTrACS data provides the first four
        features. SST, humidity and convection require
        environmental data such as ERA5/satellite inputs.
        """

        try:

            # -------------------------------------------------
            # Prepare feature sequence
            # -------------------------------------------------

            features_list = []

            # Use the latest five observations.
            for obs in historical_observations[-5:]:

                lat = obs.get("latitude")
                lon = obs.get("longitude")
                wind = obs.get("wind_speed")
                press = obs.get("pressure")

                # Environmental features are intentionally
                # NOT invented here.
                #
                # They must come from real environmental data
                # such as ERA5 / satellite integration.
                sst = obs.get("sst")
                hum = obs.get("humidity")
                conv = obs.get("convection")

                # We cannot perform a scientifically valid
                # 7-feature prediction without these values.
                if any(
                    value is None
                    for value in [
                        lat,
                        lon,
                        wind,
                        press,
                        sst,
                        hum,
                        conv
                    ]
                ):
                    return {
                        "status": "Insufficient environmental data",
                        "reason": (
                            "Genesis LSTM requires real "
                            "latitude, longitude, wind, pressure, "
                            "SST, humidity and convection values."
                        ),
                        "genesis_probability": None,
                        "confidence": None,
                        "expected_formation_hours": None
                    }

                # -------------------------------------------------
                # Normalize features
                # -------------------------------------------------

                normalized_features = [

                    (lat - 15.0) / 10.0,

                    (lon - 80.0) / 15.0,

                    (wind - 25.0) / 30.0,

                    (press - 1000.0) / 20.0,

                    (sst - 28.0) / 2.0,

                    (hum - 80.0) / 15.0,

                    (conv - 0.5) / 0.5
                ]

                features_list.append(normalized_features)


            # -------------------------------------------------
            # Require enough observations
            # -------------------------------------------------

            if len(features_list) < 5:

                return {
                    "status": "Insufficient historical observations",
                    "reason": (
                        "Genesis LSTM requires at least "
                        "5 valid observations."
                    ),
                    "genesis_probability": None,
                    "confidence": None,
                    "expected_formation_hours": None
                }


            # -------------------------------------------------
            # Convert to tensor
            # -------------------------------------------------

            input_tensor = torch.tensor(
                features_list,
                dtype=torch.float32
            ).unsqueeze(0)

            # Shape:
            # (batch_size, sequence_length, features)
            #
            # = (1, 5, 7)


            # -------------------------------------------------
            # Model inference
            # -------------------------------------------------

            with torch.no_grad():

                prob_tensor = self.model(input_tensor)


            probability = (
                float(prob_tensor.squeeze().item()) * 100.0
            )

            probability = min(
                99.0,
                max(0.0, probability)
            )


            # -------------------------------------------------
            # Presentation confidence
            # -------------------------------------------------

            confidence = (
                0.75 + probability / 400.0
                if probability > 50
                else 0.65
            )

            confidence = min(
                0.99,
                max(0.0, confidence)
            )


            # -------------------------------------------------
            # Status
            # -------------------------------------------------

            if probability > 75.0:

                status = "High probability"
                expected_hours = 24

            elif probability > 40.0:

                status = "Watch"
                expected_hours = 48

            else:

                status = "Developing System"
                expected_hours = 72


            # -------------------------------------------------
            # Return result
            # -------------------------------------------------

            return {

                "genesis_probability": round(
                    probability,
                    1
                ),

                "confidence": round(
                    confidence,
                    2
                ),

                "status": status,

                "expected_formation_hours": (
                    expected_hours
                )
            }


        except Exception as e:

            print(
                f"GenesisService: Error running prediction: {e}"
            )

            return {
                "status": "Prediction error",
                "reason": str(e),
                "genesis_probability": None,
                "confidence": None,
                "expected_formation_hours": None
            }


# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------

genesis_service = GenesisService()