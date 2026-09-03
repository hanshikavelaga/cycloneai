import os
import sys
import math
import random
import datetime
import numpy as np
from PIL import Image, ImageDraw

# Add parent directory to path to enable absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.database.connection import engine, SessionLocal, Base
from backend.app.database.models import Cyclone, Observation, Prediction, GenesisPrediction, Alert

# Create database tables if they do not exist
Base.metadata.create_all(bind=engine)

# Paths for storing mock satellite files
STORAGE_PATH = "./storage/satellite"
os.makedirs(STORAGE_PATH, exist_ok=True)


def generate_spiral_matrix(size, pattern_type, wind_speed, seed=42):
    """
    Generates a 3-channel (IR, WV, VIS) synthetic cyclone matrix of size (size, size).
    Uses mathematical functions to simulate cyclone shapes:
    - Eye: Circular warm center.
    - Curved Band: Rotating spiral arms.
    - CDO: Dense circular blob of clouds covering center.
    - Shear: Clouds displaced to one side of the center.
    """
    np.random.seed(seed)
    # Coordinate grid
    x = np.linspace(-2.0, 2.0, size)
    y = np.linspace(-2.0, 2.0, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    # Core parameters based on wind speed
    spiral_arms = 2
    tightness = 4.0 + (wind_speed / 40.0)  # Tighter spirals for higher winds
    cdo_radius = 0.4 + (wind_speed / 200.0)

    # 1. Base cloud templates
    # Spiral formula: Cosine of (tightness * R - spiral_arms * Theta)
    spirals = np.cos(tightness * R - spiral_arms * Theta)
    
    # CDO central dense overcast
    cdo = np.exp(-(R**2) / (2 * (cdo_radius**2)))

    # Apply patterns
    if pattern_type == "Eye":
        # Eye pattern: CDO + tight spirals, but center (R < 0.15) is clear (low cloud top, i.e., warm)
        cloud_base = 0.7 * cdo + 0.3 * (spirals + 1) / 2
        eye_mask = 1.0 - np.exp(-(R**2) / (2 * (0.15**2)))
        cloud_base = cloud_base * eye_mask
    elif pattern_type == "CDO":
        # Central Dense Overcast: High dense clouds in center, minor spirals outside
        cloud_base = 0.85 * cdo + 0.15 * (spirals + 1) / 2
    elif pattern_type == "Curved Band":
        # Curved Band: High spiral presence, moderate central cloudiness
        cloud_base = 0.4 * np.exp(-R**2) + 0.6 * (spirals + 1) / 2
    elif pattern_type == "Shear":
        # Shear pattern: Center displaced (shift X and Y for cloud structure)
        shift_X = X - 0.4
        shift_Y = Y - 0.4
        R_shifted = np.sqrt(shift_X**2 + shift_Y**2)
        cdo_shifted = np.exp(-(R_shifted**2) / (2 * (0.6**2)))
        cloud_base = 0.6 * cdo_shifted + 0.4 * (np.cos(3.0 * R_shifted - 2 * Theta) + 1) / 2
    else:  # No Cyclone / Disorganized
        # Random noise + very weak Gaussian blob
        noise = np.random.rand(size, size) * 0.4
        background_cloud = 0.2 * np.exp(-R**2)
        cloud_base = background_cloud + noise

    # Clip to 0-1 range
    cloud_base = np.clip(cloud_base, 0, 1)

    # 2. Multi-channel mapping (Multi-Source Satellite Data)
    # VIS (Visible) channel: High contrast, detailed cloud structures
    vis = cloud_base * 255
    
    # IR (Infrared) channel: Measures cloud-top temperature. Cold tops (strong convection) are bright.
    # Warm eye stands out. We simulate temperature profile.
    ir = cloud_base * 220
    if pattern_type == "Eye":
        # Make the eye warmer (darker in convective terms, i.e., lower IR pixel values)
        ir[R < 0.18] = 50 + (1.0 - R[R < 0.18]/0.18) * 100
        
    # WV (Water Vapor) channel: Upper-level moisture. Usually more diffused.
    wv = (0.5 * cloud_base + 0.5 * np.exp(-(R**2)/4.0)) * 255

    # Assemble channels into image array
    img_data = np.zeros((size, size, 3), dtype=np.uint8)
    img_data[..., 0] = ir.astype(np.uint8)   # Red: IR
    img_data[..., 1] = wv.astype(np.uint8)   # Green: WV
    img_data[..., 2] = vis.astype(np.uint8)  # Blue: VIS

    return Image.fromarray(img_data)


def create_mock_database_records():
    """
    Populates the database with historical and active cyclones, track observations,
    forecast tracks, alerts, and cyclogenesis probabilities.
    """
    db = SessionLocal()
    try:
        # Clear existing data to make it clean
        db.query(Alert).delete()
        db.query(Prediction).delete()
        db.query(Observation).delete()
        db.query(Cyclone).delete()
        db.query(GenesisPrediction).delete()
        db.commit()

        print("Cleared existing database tables. Inserting mock records...")

        # ----------------------------------------------------
        # 1. Historical Cyclone: Cyclone Fani (BOB 2019)
        # ----------------------------------------------------
        fani = Cyclone(
            id="2019_BOB_01",
            name="Fani",
            basin="Bay of Bengal",
            status="Dissipated",
            start_time=datetime.datetime(2019, 4, 27, 6, 0),
            end_time=datetime.datetime(2019, 5, 4, 18, 0)
        )
        db.add(fani)

        # Fani Path coordinates (historical)
        # Moving North-West then North-East towards Odisha coast
        fani_coords = [
            (datetime.datetime(2019, 4, 27, 6, 0), 5.2, 88.5, 30, 1000, "Shear"),
            (datetime.datetime(2019, 4, 28, 6, 0), 7.4, 87.2, 45, 992, "Curved Band"),
            (datetime.datetime(2019, 4, 29, 6, 0), 9.6, 86.5, 60, 982, "Curved Band"),
            (datetime.datetime(2019, 4, 30, 6, 0), 11.8, 86.2, 80, 965, "CDO"),
            (datetime.datetime(2019, 5, 1, 6, 0), 13.9, 85.4, 115, 935, "Eye"),
            (datetime.datetime(2019, 5, 2, 6, 0), 16.5, 84.8, 130, 915, "Eye"),
            (datetime.datetime(2019, 5, 3, 6, 0), 19.6, 85.7, 110, 940, "CDO"),  # Landfall near Puri
            (datetime.datetime(2019, 5, 4, 6, 0), 23.1, 89.2, 40, 990, "Shear")
        ]

        for i, (ts, lat, lon, wind, press, pat) in enumerate(fani_coords):
            # Generate and save image
            filename = f"2019_BOB_01_{ts.strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(STORAGE_PATH, filename)
            img = generate_spiral_matrix(128, pat, wind, seed=i)
            img.save(filepath)

            # Insert observation
            t_number = 1.0 + (wind / 20.0)
            obs = Observation(
                cyclone_id=fani.id,
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                wind_speed=wind,
                pressure=press,
                dvorak_t_number=round(t_number, 1),
                pattern_type=pat,
                image_path=filepath,
                is_synthetic=True
            )
            db.add(obs)

        # ----------------------------------------------------
        # 2. Active Cyclone: Cyclone Aila-2 (Active Mock Storm for demonstration)
        # ----------------------------------------------------
        active_time = datetime.datetime.utcnow() - datetime.timedelta(days=2)
        aila = Cyclone(
            id="2026_BOB_02",
            name="Aila-II",
            basin="Bay of Bengal",
            status="Active",
            start_time=active_time,
            end_time=None
        )
        db.add(aila)

        # Path history for active cyclone up to "current" time
        aila_history = [
            (active_time, 10.2, 85.0, 35, 995, "Curved Band"),
            (active_time + datetime.timedelta(hours=12), 11.5, 84.8, 50, 988, "Curved Band"),
            (active_time + datetime.timedelta(hours=24), 12.8, 84.3, 70, 975, "CDO"),
            (active_time + datetime.timedelta(hours=36), 14.1, 83.9, 90, 960, "Eye"),
            (active_time + datetime.timedelta(hours=48), 15.4, 83.5, 105, 948, "Eye")  # Latest position
        ]

        for i, (ts, lat, lon, wind, press, pat) in enumerate(aila_history):
            filename = f"2026_BOB_02_{ts.strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(STORAGE_PATH, filename)
            img = generate_spiral_matrix(128, pat, wind, seed=100+i)
            img.save(filepath)

            obs = Observation(
                cyclone_id=aila.id,
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                wind_speed=wind,
                pressure=press,
                dvorak_t_number=round(1.0 + (wind / 20.0), 1),
                pattern_type=pat,
                image_path=filepath,
                is_synthetic=True
            )
            db.add(obs)

        # Forecast predictions for active storm Aila-2 (+6h, +12h, +24h, +48h)
        latest_time = active_time + datetime.timedelta(hours=48)
        forecasts = [
            (latest_time + datetime.timedelta(hours=6), 16.2, 83.1, 115, 940, 0.95),
            (latest_time + datetime.timedelta(hours=12), 17.0, 82.9, 120, 932, 0.88),
            (latest_time + datetime.timedelta(hours=24), 18.5, 82.8, 125, 928, 0.75),
            (latest_time + datetime.timedelta(hours=48), 20.9, 83.4, 100, 950, 0.58)  # Nearing landfall
        ]

        for ts, lat, lon, wind, press, conf in forecasts:
            pred = Prediction(
                cyclone_id=aila.id,
                timestamp=latest_time,
                forecast_time=ts,
                latitude=lat,
                longitude=lon,
                wind_speed=wind,
                pressure=press,
                confidence=conf
            )
            db.add(pred)

        # ----------------------------------------------------
        # 3. Genesis Predictions (Monitoring Oceanic Basins)
        # ----------------------------------------------------
        # Monitored regions showing formation probabilities
        gp_bob = GenesisPrediction(
            region="Bay of Bengal (Central)",
            timestamp=datetime.datetime.utcnow(),
            probability=78.0,
            confidence=0.85,
            expected_formation_time=datetime.datetime.utcnow() + datetime.timedelta(hours=36)
        )
        gp_as = GenesisPrediction(
            region="Arabian Sea (South)",
            timestamp=datetime.datetime.utcnow(),
            probability=24.0,
            confidence=0.62,
            expected_formation_time=None
        )
        db.add(gp_bob)
        db.add(gp_as)

        # ----------------------------------------------------
        # 4. Alerts
        # ----------------------------------------------------
        alert_1 = Alert(
            cyclone_id=aila.id,
            type="Intensification",
            severity="Red",
            message="Cyclone Aila-II has intensified to an Extremely Severe Cyclonic Storm. Wind speeds reached 105 knots.",
            timestamp=datetime.datetime.utcnow()
        )
        alert_2 = Alert(
            cyclone_id=None,
            type="Genesis",
            severity="Orange",
            message="A high-probability (78%) cyclogenesis event is expected in central Bay of Bengal within 36-48 hours. Monitoring disturbance 92B.",
            timestamp=datetime.datetime.utcnow()
        )
        db.add(alert_1)
        db.add(alert_2)

        db.commit()
        print("Successfully seeded mock cyclone data and generated synthetic satellite images.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    create_mock_database_records()
