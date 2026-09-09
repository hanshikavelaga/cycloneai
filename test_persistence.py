from datetime import datetime

from backend.app.database.connection import SessionLocal
from backend.app.database.models import Cyclone, Observation
from backend.app.services.region_filter import classify_region


TEST_RECORD = {
    "source": "TEST",
    "source_storm_id": "PERSISTENCE_TEST_01",
    "storm_name": "PERSISTENCE TEST",
    "timestamp_utc": "2026-09-06 14:00:00 UTC",
    "latitude": 15.0,
    "longitude": 85.0,
    "wind_kts": 45.0,
    "pressure_hpa": 1000.0,
    "region": classify_region(15.0, 85.0),
}


db = SessionLocal()

try:
    print("Region:", TEST_RECORD["region"])

    if TEST_RECORD["region"] != "NORTH_INDIAN_OCEAN":
        raise RuntimeError("Test record is not inside NIO")

    cyclone_id = f'{TEST_RECORD["source"]}_{TEST_RECORD["source_storm_id"]}'

    cyclone = db.query(Cyclone).filter(
        Cyclone.id == cyclone_id
    ).first()

    if not cyclone:
        cyclone = Cyclone(
            id=cyclone_id,
            name=TEST_RECORD["storm_name"],
            basin="Bay of Bengal",
            status="Active",
            start_time=datetime.utcnow(),
        )

        db.add(cyclone)
        db.commit()

        print("Cyclone created:", cyclone_id)
    else:
        print("Cyclone already exists:", cyclone_id)

    timestamp = datetime.strptime(
        TEST_RECORD["timestamp_utc"].replace(" UTC", ""),
        "%Y-%m-%d %H:%M:%S"
    )

    existing = db.query(Observation).filter(
        Observation.cyclone_id == cyclone_id,
        Observation.timestamp == timestamp
    ).first()

    if existing:
        print("DUPLICATE DETECTED - not inserting")
    else:
        observation = Observation(
            cyclone_id=cyclone_id,
            timestamp=timestamp,
            latitude=TEST_RECORD["latitude"],
            longitude=TEST_RECORD["longitude"],
            wind_speed=TEST_RECORD["wind_kts"],
            pressure=TEST_RECORD["pressure_hpa"],
            pattern_type="Shear",
            is_synthetic=False,
        )

        db.add(observation)
        db.commit()

        print("Observation inserted!")
        print("Observation ID:", observation.id)

finally:
    db.close()