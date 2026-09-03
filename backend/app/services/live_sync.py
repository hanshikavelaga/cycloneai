from datetime import datetime
import logging
from backend.app.database.connection import SessionLocal
from backend.app.database.models import Cyclone, Observation
from backend.app.services.gdacs_client import fetch_active_storms_gdacs
from backend.app.services.ibtracs_live_client import fetch_active_storms_noaa
from backend.app.services.status_cache import status_manager

logger = logging.getLogger("live_sync")

def execute_sync():
    """
    Main orchestrator that polls GDACS and NOAA active feeds, normalizes records,
    classifies coordinates, and writes unique North Indian Ocean observations to Supabase.
    """
    db = SessionLocal()
    last_sync_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    status_manager.add_log_message("Starting automated data synchronization cycle...")
    
    # ----------------------------------------------------
    # 1. Fetch and Process GDACS Feed
    # ----------------------------------------------------
    gdacs_received = 0
    gdacs_stored = 0
    gdacs_status = "CONNECTED"
    gdacs_error = None
    
    try:
        status_manager.add_log_message("Polling GDACS active tropical cyclones feed...")
        gdacs_records = fetch_active_storms_gdacs()
        gdacs_received = len(gdacs_records)
        
        for record in gdacs_records:
            # Check region
            if record["region"] != "NORTH_INDIAN_OCEAN":
                logger.info(f"Skipping GDACS storm {record['storm_name']} - location {record['region']}")
                continue
                
            db_cyclone_id = f"{record['source']}_{record['source_storm_id']}"
            
            # Map longitude to specific basin
            # Lon < 80.0 is Arabian Sea; Lon >= 80.0 is Bay of Bengal
            basin = "Bay of Bengal" if record["longitude"] >= 80.0 else "Arabian Sea"
            
            # Check/Insert Cyclone
            cyclone = db.query(Cyclone).filter(Cyclone.id == db_cyclone_id).first()
            if not cyclone:
                cyclone = Cyclone(
                    id=db_cyclone_id,
                    name=record["storm_name"],
                    basin=basin,
                    status="Active",
                    start_time=datetime.utcnow()
                )
                db.add(cyclone)
                db.commit()
                status_manager.add_log_message(f"Registered new active cyclone: {record['storm_name']} ({db_cyclone_id})")
            
            # Check/Insert unique Observation
            # Unique Constraint = source + storm_id + timestamp (implicitly enforced via db_cyclone_id + timestamp)
            dt_timestamp = datetime.strptime(record["timestamp_utc"].replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
            
            obs_exists = db.query(Observation).filter(
                Observation.cyclone_id == db_cyclone_id,
                Observation.timestamp == dt_timestamp
            ).first()
            
            if not obs_exists:
                new_obs = Observation(
                    cyclone_id=db_cyclone_id,
                    timestamp=dt_timestamp,
                    latitude=record["latitude"],
                    longitude=record["longitude"],
                    wind_speed=record["wind_kts"],
                    pressure=record["pressure_hpa"],
                    pattern_type="Shear",  # Default classification category
                    is_synthetic=False
                )
                db.add(new_obs)
                db.commit()
                gdacs_stored += 1
                status_manager.add_log_message(f"Stored observation for {record['storm_name']} at {record['timestamp_utc']}")
                
        status_manager.update_gdacs(
            status=gdacs_status,
            last_success=last_sync_time,
            received=gdacs_received,
            stored=gdacs_stored
        )
    except Exception as e:
        gdacs_status = "ERROR"
        gdacs_error = str(e)
        logger.error(f"GDACS synchronization failed: {e}")
        status_manager.update_gdacs(status="ERROR", last_error=gdacs_error)
        status_manager.add_log_message(f"GDACS Ingestion Failed: {gdacs_error}")

    # ----------------------------------------------------
    # 2. Fetch and Process NOAA Active CSV Feed
    # ----------------------------------------------------
    noaa_received = 0
    noaa_stored = 0
    noaa_status = "CONNECTED"
    noaa_error = None
    
    try:
        status_manager.add_log_message("Polling NOAA active 7-day IBTrACS CSV database...")
        noaa_records = fetch_active_storms_noaa()
        noaa_received = len(noaa_records)
        
        for record in noaa_records:
            # Check region
            if record["region"] != "NORTH_INDIAN_OCEAN":
                logger.info(f"Skipping NOAA storm {record['storm_name']} - location {record['region']}")
                continue
                
            db_cyclone_id = f"{record['source']}_{record['source_storm_id']}"
            basin = "Bay of Bengal" if record["longitude"] >= 80.0 else "Arabian Sea"
            
            # Check/Insert Cyclone
            cyclone = db.query(Cyclone).filter(Cyclone.id == db_cyclone_id).first()
            if not cyclone:
                cyclone = Cyclone(
                    id=db_cyclone_id,
                    name=record["storm_name"],
                    basin=basin,
                    status="Active",
                    start_time=datetime.utcnow()
                )
                db.add(cyclone)
                db.commit()
                status_manager.add_log_message(f"Registered new active cyclone: {record['storm_name']} ({db_cyclone_id})")
            
            # Check/Insert unique Observation
            dt_timestamp = datetime.strptime(record["timestamp_utc"].replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
            
            obs_exists = db.query(Observation).filter(
                Observation.cyclone_id == db_cyclone_id,
                Observation.timestamp == dt_timestamp
            ).first()
            
            if not obs_exists:
                new_obs = Observation(
                    cyclone_id=db_cyclone_id,
                    timestamp=dt_timestamp,
                    latitude=record["latitude"],
                    longitude=record["longitude"],
                    wind_speed=record["wind_kts"],
                    pressure=record["pressure_hpa"],
                    pattern_type="Shear",
                    is_synthetic=False
                )
                db.add(new_obs)
                db.commit()
                noaa_stored += 1
                status_manager.add_log_message(f"Stored observation for {record['storm_name']} at {record['timestamp_utc']}")
                
        status_manager.update_noaa(
            status=noaa_status,
            last_success=last_sync_time,
            received=noaa_received,
            stored=noaa_stored
        )
    except Exception as e:
        noaa_status = "ERROR"
        noaa_error = str(e)
        logger.error(f"NOAA synchronization failed: {e}")
        status_manager.update_noaa(status="ERROR", last_error=noaa_error)
        status_manager.add_log_message(f"NOAA Ingestion Failed: {noaa_error}")
        
    db.close()
    status_manager.add_log_message("Automated data synchronization cycle completed.")
