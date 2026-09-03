import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from backend.app.services.region_filter import classify_region

NOAA_ACTIVE_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ACTIVE.list.v04r01.csv"

def fetch_active_storms_noaa():
    """
    Downloads the active 7-day IBTrACS CSV database from NOAA,
    filters for the North Indian Ocean (NI) basin, and normalizes them.
    """
    try:
        response = requests.get(NOAA_ACTIVE_URL, timeout=20)
        response.raise_for_status()
        
        # Read CSV directly (no skiprows, since active list has headers at line 0)
        data_io = StringIO(response.text)
        df = pd.read_csv(data_io, low_memory=False)
        
        if df.empty:
            return []
            
        # The first row of the parsed DataFrame contains column data types (e.g. 'str', 'deg', 'kt')
        # We need to drop this row before processing the storm observations
        df = df.drop(df.index[0])
        
        # Filter for the North Indian Ocean (NI) basin
        # Note: Columns in IBTrACS are case-insensitive or usually uppercase.
        # Let's ensure we map the column names correctly.
        df.columns = [col.upper() for col in df.columns]
        
        # Filter rows by BASIN == "NI"
        ni_df = df[df["BASIN"] == "NI"].copy()
        
        normalized_records = []
        
        for _, row in ni_df.iterrows():
            sid = row.get("SID")
            name = row.get("NAME")
            iso_time = row.get("ISO_TIME")
            lat_str = row.get("LAT")
            lon_str = row.get("LON")
            wind_str = row.get("WMO_WIND")
            pres_str = row.get("WMO_PRES")
            
            if not sid or pd.isna(lat_str) or pd.isna(lon_str):
                continue
                
            try:
                lat = float(lat_str)
                lon = float(lon_str)
            except ValueError:
                continue
                
            # Parse Wind Speed (WMO_WIND is reported in knots)
            wind_kts = 0.0
            if not pd.isna(wind_str):
                try:
                    val = float(wind_str)
                    if val >= 0:
                        wind_kts = val
                except ValueError:
                    pass
            
            # Parse Pressure (WMO_PRES is reported in hPa)
            pressure_hpa = 1010.0
            if not pd.isna(pres_str):
                try:
                    val = float(pres_str)
                    if val >= 0:
                        pressure_hpa = val
                except ValueError:
                    pass
            
            # Format Timestamp: IBTrACS has ISO_TIME like '2026-08-27 12:00:00'
            timestamp_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            if not pd.isna(iso_time):
                try:
                    # Append UTC suffix if missing
                    time_str = str(iso_time).strip()
                    if not time_str.endswith("UTC"):
                        timestamp_utc = f"{time_str} UTC"
                    else:
                        timestamp_utc = time_str
                except Exception:
                    pass
            
            # Run Region Filter
            region = classify_region(lat, lon)
            
            record = {
                "source": "NOAA_IBTRACS",
                "source_storm_id": str(sid).strip(),
                "storm_name": str(name).strip().upper() if not pd.isna(name) else "UNNAMED",
                "timestamp_utc": timestamp_utc,
                "latitude": lat,
                "longitude": lon,
                "wind_kts": round(wind_kts, 1),
                "pressure_hpa": round(pressure_hpa, 1),
                "region": region
            }
            normalized_records.append(record)
            
        return normalized_records
    except Exception as e:
        raise RuntimeError(f"Error fetching/parsing NOAA Active CSV: {e}")

if __name__ == "__main__":
    # Quick test run
    try:
        data = fetch_active_storms_noaa()
        print(f"Success! Fetched {len(data)} storms from NOAA Active Feed.")
        for idx, storm in enumerate(data[:3]):
            print(f"Storm {idx+1}: {storm['storm_name']} at ({storm['latitude']}, {storm['longitude']}) - Wind: {storm['wind_kts']} kts")
    except Exception as ex:
        print(f"Test run failed: {ex}")
