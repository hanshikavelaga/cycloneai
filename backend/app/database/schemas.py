from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime

# Configure configurations for Pydantic v2 compatible with SQLAlchemy models
class BaseConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- Alerts Schemas ---
class AlertBase(BaseConfig):
    type: str
    severity: str
    message: str
    cyclone_id: Optional[str] = None
    status: str = "Active"

class AlertCreate(AlertBase):
    pass

class AlertResponse(AlertBase):
    id: int
    timestamp: datetime


# --- Observation Schemas ---
class ObservationBase(BaseConfig):
    timestamp: datetime
    latitude: float
    longitude: float
    wind_speed: float
    pressure: float
    dvorak_t_number: Optional[float] = None
    pattern_type: str
    image_path: Optional[str] = None
    is_synthetic: bool = True

class ObservationCreate(ObservationBase):
    cyclone_id: Optional[str] = None

class ObservationResponse(ObservationBase):
    id: int
    cyclone_id: Optional[str] = None


# --- Prediction Schemas ---
class PredictionBase(BaseConfig):
    forecast_time: datetime
    latitude: float
    longitude: float
    wind_speed: float
    pressure: float
    confidence: float

class PredictionCreate(PredictionBase):
    cyclone_id: str

class PredictionResponse(PredictionBase):
    id: int
    cyclone_id: str
    timestamp: datetime


# --- Cyclone Schemas ---
class CycloneBase(BaseConfig):
    id: str
    name: str
    basin: str
    status: str

class CycloneCreate(CycloneBase):
    pass

class CycloneResponse(CycloneBase):
    start_time: datetime
    end_time: Optional[datetime] = None

class CycloneDetailResponse(CycloneResponse):
    observations: List[ObservationResponse] = []
    predictions: List[PredictionResponse] = []
    alerts: List[AlertResponse] = []


# --- Genesis Prediction Schemas ---
class GenesisPredictionBase(BaseConfig):
    region: str
    probability: float
    confidence: float
    expected_formation_time: Optional[datetime] = None

class GenesisPredictionCreate(GenesisPredictionBase):
    pass

class GenesisPredictionResponse(GenesisPredictionBase):
    id: int
    timestamp: datetime
