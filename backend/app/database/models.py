from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from .connection import Base
import datetime

class Cyclone(Base):
    __tablename__ = "cyclones"

    id = Column(String, primary_key=True, index=True)  # Format: YYYY_BASIN_ID (e.g., 2026_BOB_01)
    name = Column(String, nullable=False)
    basin = Column(String, nullable=False)              # "Bay of Bengal", "Arabian Sea"
    status = Column(String, nullable=False)             # "Developing", "Active", "Dissipated"
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)

    # Relationships
    observations = relationship("Observation", back_populates="cyclone", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="cyclone", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="cyclone", cascade="all, delete-orphan")


class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cyclone_id = Column(String, ForeignKey("cyclones.id"), nullable=True)  # Nullable for genesis/disturbance events
    timestamp = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=False)          # In knots
    pressure = Column(Float, nullable=False)            # In hPa
    dvorak_t_number = Column(Float, nullable=True)      # Dvorak intensity index (1.0 to 8.0)
    pattern_type = Column(String, nullable=False)       # Curved Band, CDO, Eye, Shear, No Cyclone
    image_path = Column(String, nullable=True)          # Local filepath to stacked spectral image (IR/WV/VIS)
    is_synthetic = Column(Boolean, default=True)

    cyclone = relationship("Cyclone", back_populates="observations")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cyclone_id = Column(String, ForeignKey("cyclones.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)         # Time when prediction was computed
    forecast_time = Column(DateTime, nullable=False)                      # Target time for the forecast (+6h, +12h, etc.)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=False)
    pressure = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)                            # Range: 0.0 to 1.0

    cyclone = relationship("Cyclone", back_populates="predictions")


class GenesisPrediction(Base):
    __tablename__ = "genesis_predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    region = Column(String, nullable=False)                              # "Bay of Bengal", "Arabian Sea"
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    probability = Column(Float, nullable=False)                          # Formation probability percentage (0 to 100)
    confidence = Column(Float, nullable=False)                            # Range: 0.0 to 1.0
    expected_formation_time = Column(DateTime, nullable=True)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cyclone_id = Column(String, ForeignKey("cyclones.id"), nullable=True)
    type = Column(String, nullable=False)                                # "Genesis", "Intensification", "Track", "Landfall"
    severity = Column(String, nullable=False)                            # "Green", "Yellow", "Orange", "Red"
    message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="Active")                            # "Active", "Dismissed"

    cyclone = relationship("Cyclone", back_populates="alerts")
