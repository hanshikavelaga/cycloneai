import xml.etree.ElementTree as ET
import requests
from datetime import datetime
import email.utils
from backend.app.services.region_filter import classify_region

GDACS_RSS_URL = "https://www.gdacs.org/xml/rss.xml"

def strip_namespaces(tree_root):
    """
    Strips namespaces from all tags in the XML tree to make parsing 
    independent of the exact namespace URL schema used by GDACS.
    """
    for elem in tree_root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return tree_root

def fetch_active_storms_gdacs():
    """
    Fetches the active storm list from the GDACS RSS feed, filters 
    for Tropical Cyclones (TC), and normalizes them to the common schema.
    """
    try:
        response = requests.get(GDACS_RSS_URL, timeout=15)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        root = strip_namespaces(root)
        
        items = root.findall(".//item")
        normalized_records = []
        
        for item in items:
            event_type = item.findtext("eventtype")
            if event_type != "TC":
                continue  # Only process Tropical Cyclones
                
            event_id = item.findtext("eventid")
            storm_name = item.findtext("evname")
            lat_str = item.findtext("latitude")
            lon_str = item.findtext("longitude")
            wind_kmh_str = item.findtext("windspeed")
            pressure_str = item.findtext("pressure")
            pub_date_str = item.findtext("pubDate")
            
            if not event_id or not storm_name or not lat_str or not lon_str:
                continue
                
            # Parse Coordinates
            lat = float(lat_str.strip())
            lon = float(lon_str.strip())
            
            # Convert Wind Speed: GDACS reports in km/h. Convert to Knots.
            # 1 Knot = 1.852 km/h
            wind_kts = 0.0
            if wind_kmh_str:
                try:
                    wind_kts = float(wind_kmh_str.strip()) / 1.852
                except ValueError:
                    pass
                    
            # Parse Pressure
            pressure_hpa = 1010.0
            if pressure_str:
                try:
                    pressure_hpa = float(pressure_str.strip())
                except ValueError:
                    pass
            
            # Format Timestamp to ISO 8601 UTC
            timestamp_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            if pub_date_str:
                try:
                    dt = email.utils.parsedate_to_datetime(pub_date_str)
                    timestamp_utc = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                except Exception:
                    pass
            
            # Run Region Filter
            region = classify_region(lat, lon)
            
            record = {
                "source": "GDACS",
                "source_storm_id": str(event_id).strip(),
                "storm_name": str(storm_name).strip().upper(),
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
        raise RuntimeError(f"Error fetching/parsing GDACS RSS feed: {e}")

if __name__ == "__main__":
    # Quick test run
    try:
        data = fetch_active_storms_gdacs()
        print(f"Success! Fetched {len(data)} storms from GDACS.")
        for idx, storm in enumerate(data[:3]):
            print(f"Storm {idx+1}: {storm['storm_name']} at ({storm['latitude']}, {storm['longitude']}) - Wind: {storm['wind_kts']} kts")
    except Exception as ex:
        print(f"Test run failed: {ex}")
