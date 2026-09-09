import os
import sys
import datetime
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from io import BytesIO

# Add parent directory to path to enable absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.database.connection import get_db, engine, Base
from backend.app.database import models, schemas
from backend.app.services.detection_service import detection_service
from backend.app.services.genesis_service import genesis_service
from backend.app.services.forecast_service import forecast_service
from src.features.fusion import perform_spectral_fusion
from backend.app.scheduler.live_scheduler import start_scheduler, stop_scheduler
from backend.app.services.status_cache import status_manager
from backend.app.services.live_sync import execute_sync

app = FastAPI(
    title="CycloneAI API",
    description="Backend API for cyclone genesis, classification, and forecast tracking.",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite default port 5173 and others)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    # Start the background sync scheduler task
    start_scheduler()

@app.on_event("shutdown")
def shutdown_event():
    # Stop the background sync scheduler task
    stop_scheduler()

STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "satellite"))


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow()}


@app.get("/api/system/status")
def get_system_status():
    return status_manager.get_status()


@app.post("/api/system/sync")
def trigger_manual_sync():
    try:
        execute_sync()
        return {"status": "success", "message": "Manual synchronization completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Cyclone endpoints ---

@app.get("/api/cyclones/active", response_model=list[schemas.CycloneResponse])
def get_active_cyclones(db: Session = Depends(get_db)):
    return db.query(models.Cyclone).filter(models.Cyclone.status == "Active").all()


@app.get("/api/cyclones/historical", response_model=list[schemas.CycloneResponse])
def get_historical_cyclones(db: Session = Depends(get_db)):
    return db.query(models.Cyclone).filter(models.Cyclone.status == "Dissipated").all()


@app.get("/api/cyclones/{cyclone_id}", response_model=schemas.CycloneDetailResponse)
def get_cyclone_detail(cyclone_id: str, db: Session = Depends(get_db)):
    cyclone = db.query(models.Cyclone).filter(models.Cyclone.id == cyclone_id).first()
    if not cyclone:
        raise HTTPException(status_code=404, detail="Cyclone not found")
    return cyclone


# --- Genesis prediction ---

@app.get("/api/genesis", response_model=list[schemas.GenesisPredictionResponse])
def get_genesis_predictions(db: Session = Depends(get_db)):
    return db.query(models.GenesisPrediction).order_by(models.GenesisPrediction.timestamp.desc()).all()


# --- Alerts ---

@app.get("/api/alerts", response_model=list[schemas.AlertResponse])
def get_alerts(db: Session = Depends(get_db)):
    return db.query(models.Alert).filter(models.Alert.status == "Active").order_by(models.Alert.timestamp.desc()).all()


# --- Satellite image access & fusion ---

@app.get("/api/satellite/image/{filename}")
def get_satellite_image(filename: str):
    filepath = os.path.join(STORAGE_PATH, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(filepath)


@app.get("/api/satellite/fuse/{filename}")
def fuse_satellite_image(filename: str, alpha: float = Query(0.6, ge=0.0, le=1.0)):
    filepath = os.path.join(STORAGE_PATH, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        fused_img = perform_spectral_fusion(filepath, alpha)
        
        # Save to memory buffer
        buf = BytesIO()
        fused_img.save(buf, format="PNG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image fusion error: {e}")


# --- Inference & Prediction triggers ---

@app.post("/api/predict/image")
async def predict_uploaded_image(file: UploadFile = File(...)):
    """
    Accepts uploaded satellite image, saves it temporarily, runs PyTorch CNN,
    and returns Dvorak pattern and intensity metrics.
    """
    temp_path = f"./storage/processed/temp_{file.filename}"
    os.makedirs("./storage/processed", exist_ok=True)
    
    try:
        # Save temp file
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
            
        # Run inference
        results = detection_service.run_inference(temp_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/forecast/{cyclone_id}")
def run_forecast(cyclone_id: str, db: Session = Depends(get_db)):
    """
    Loads historical observations for specified cyclone, feeds them to LSTM,
    calculates forecast coordinates, saves them to database, and returns forecasts.
    """
    cyclone = db.query(models.Cyclone).filter(models.Cyclone.id == cyclone_id).first()
    if not cyclone:
        raise HTTPException(status_code=404, detail="Cyclone not found")

    # Fetch observations
    obs_list = db.query(models.Observation).filter(models.Observation.cyclone_id == cyclone_id).order_by(models.Observation.timestamp.asc()).all()
    if not obs_list:
        raise HTTPException(status_code=400, detail="No observations found to base forecast upon")

    # Format observations for service
    obs_data = []
    for obs in obs_list:
        obs_data.append({
            "timestamp": obs.timestamp,
            "latitude": obs.latitude,
            "longitude": obs.longitude,
            "wind_speed": obs.wind_speed,
            "pressure": obs.pressure
        })

    # Run LSTM forecasting
    forecasts = forecast_service.forecast_track(obs_data)

    # Clear old predictions
    db.query(models.Prediction).filter(models.Prediction.cyclone_id == cyclone_id).delete()

    # Save new predictions
    for f in forecasts:
        pred = models.Prediction(
            cyclone_id=cyclone_id,
            timestamp=datetime.datetime.utcnow(),
            forecast_time=f["forecast_time"],
            latitude=f["latitude"],
            longitude=f["longitude"],
            wind_speed=f["wind_speed"],
            pressure=f["pressure"],
            confidence=f["confidence"]
        )
        db.add(pred)
    db.commit()

    return forecasts


# --- Historical Replay Mode (Winning SIH Demo Feature) ---

@app.get("/api/replay/{cyclone_id}/steps")
def get_replay_steps(cyclone_id: str, db: Session = Depends(get_db)):
    """
    Returns the list of observations (timestamps) representing the steps of a cyclone's progression.
    """
    obs_list = db.query(models.Observation).filter(models.Observation.cyclone_id == cyclone_id).order_by(models.Observation.timestamp.asc()).all()
    return [{"index": idx, "timestamp": obs.timestamp} for idx, obs in enumerate(obs_list)]


@app.get("/api/replay/{cyclone_id}/step/{step_idx}")
def get_replay_step_state(cyclone_id: str, step_idx: int, db: Session = Depends(get_db)):
    """
    Simulates a point-in-time state of the dashboard:
    Loads only observations up to step_idx.
    Runs forecasting based only on this historical slice, and returns:
    1. Cyclone details at that moment.
    2. Observations list up to step_idx.
    3. Forecasts calculated at that moment.
    """
    # 1. Fetch cyclone
    cyclone = db.query(models.Cyclone).filter(models.Cyclone.id == cyclone_id).first()
    if not cyclone:
        raise HTTPException(status_code=404, detail="Cyclone not found")

    # 2. Fetch observations up to step_idx
    all_obs = db.query(models.Observation).filter(models.Observation.cyclone_id == cyclone_id).order_by(models.Observation.timestamp.asc()).all()
    if step_idx < 0 or step_idx >= len(all_obs):
        raise HTTPException(status_code=400, detail="Invalid step index")

    current_obs = all_obs[:step_idx + 1]
    latest_obs = current_obs[-1]

    # Format history slice for forecast input
    obs_data = [{
        "timestamp": o.timestamp,
        "latitude": o.latitude,
        "longitude": o.longitude,
        "wind_speed": o.wind_speed,
        "pressure": o.pressure
    } for o in current_obs]

    # 3. Calculate forecast based ONLY on history slice
    forecasts = forecast_service.forecast_track(obs_data)

    # 4. Generate Explainable AI triggers (Grad-CAM confidence indicators)
    # Simulated explanation for why AI predicted intensity/track
    explanation = {
        "reasons": [
            f"Sustained wind speed is increasing ({latest_obs.wind_speed} knots)",
            f"Central pressure is falling ({latest_obs.pressure} hPa)",
            "Cloud spiral structure is organizing in VIS/IR bands"
        ]
    }
    if latest_obs.pattern_type == "Eye":
        explanation["reasons"].append("Clear central eye visible; indicates highly matured vortex")
    elif latest_obs.pattern_type == "CDO":
        explanation["reasons"].append("Dense cloud shielding center; indicating central convective core formation")
    
    # Check if there is land nearby to explain path or decay
    # Odisha coast region bounding box
    if 18.0 < latest_obs.latitude < 22.0 and 84.0 < latest_obs.longitude < 88.0:
        explanation["reasons"].append("Friction of land interaction will weaken convective structures soon")

    return {
        "cyclone": {
            "id": cyclone.id,
            "name": cyclone.name,
            "basin": cyclone.basin,
            "status": "Active" if step_idx < len(all_obs)-1 else "Dissipated",
            "current_step": step_idx,
            "total_steps": len(all_obs)
        },
        "latest_observation": latest_obs,
        "history": current_obs,
        "forecasts": forecasts,
        "explanation": explanation
    }
