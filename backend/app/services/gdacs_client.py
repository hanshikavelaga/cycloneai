import xml.etree.ElementTree as ET
import requests
import re

from datetime import datetime, timezone
import email.utils

from backend.app.services.region_filter import classify_region


GDACS_RSS_URL = "https://www.gdacs.org/xml/rss.xml"


def strip_namespaces(tree_root):
    """
    Strip XML namespaces from all tags.

    GDACS currently uses namespaced tags such as:

        <gdacs:eventtype>TC</gdacs:eventtype>

    After stripping the namespace, the tag becomes:

        <eventtype>TC</eventtype>

    This makes the rest of the parser independent of the
    GDACS namespace URL.
    """

    for elem in tree_root.iter():

        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    return tree_root


def get_text(item, *field_names):
    """
    Return the first non-empty text value among the requested
    XML field names.

    This allows the parser to support multiple possible
    GDACS field names.
    """

    for field_name in field_names:

        value = item.findtext(field_name)

        if value is not None and value.strip():
            return value.strip()

    return None


def parse_point(point_str):
    """
    Parse GDACS point format.

    Example:

        "28.5 133.8"

    Returns:

        latitude, longitude

    If parsing fails:

        None, None
    """

    if not point_str:
        return None, None

    try:

        # Handle both:
        # "28.5 133.8"
        # "28.5,133.8"

        parts = point_str.replace(",", " ").split()

        if len(parts) < 2:
            return None, None

        latitude = float(parts[0])
        longitude = float(parts[1])

        return latitude, longitude

    except (ValueError, TypeError):

        return None, None


def parse_wind_speed(item):
    """
    Extract maximum wind speed from the current GDACS RSS schema.

    Current GDACS TC records do not provide a dedicated
    <windspeed> field.

    Instead, the wind speed appears inside fields such as:

        description:
        "maximum wind speed of 74 km/h"

    and/or:

        severity:
        "maximum wind speed of 46 km/h"

    Returns wind speed in knots.

    Returns None if wind speed cannot be extracted.
    """

    description = get_text(
        item,
        "description"
    ) or ""

    severity = get_text(
        item,
        "severity"
    ) or ""

    # Search description first because it represents
    # the maximum wind speed in the current GDACS item.

    search_texts = [
        description,
        severity
    ]

    for text in search_texts:

        match = re.search(
            r"maximum\s+wind\s+speed\s+of\s+"
            r"(\d+(?:\.\d+)?)\s*km/h",
            text,
            flags=re.IGNORECASE
        )

        if match:

            try:

                wind_kmh = float(
                    match.group(1)
                )

                # km/h -> knots
                wind_kts = wind_kmh / 1.852

                return round(
                    wind_kts,
                    1
                )

            except (ValueError, TypeError):

                continue

    return None


def parse_pressure(item):
    """
    Extract central pressure if GDACS provides it.

    The currently observed GDACS RSS TC item does not expose
    a dedicated pressure field.

    Therefore we return None instead of inventing a value
    such as 1010 hPa.

    This prevents false pressure information from entering
    the CycloneAI database.
    """

    pressure_str = get_text(
        item,
        "pressure",
        "centralpressure",
        "central_pressure",
        "minpressure",
        "minimumpressure"
    )

    if not pressure_str:
        return None

    try:

        return round(
            float(pressure_str),
            1
        )

    except (ValueError, TypeError):

        return None


def parse_timestamp(pub_date_str):
    """
    Convert GDACS pubDate into a UTC timestamp.

    Example input:

        Fri, 04 Sep 2026 15:42:34 GMT

    Output:

        2026-09-04 15:42:34 UTC
    """

    if pub_date_str:

        try:

            dt = email.utils.parsedate_to_datetime(
                pub_date_str
            )

            if dt.tzinfo is None:

                dt = dt.replace(
                    tzinfo=timezone.utc
                )

            dt = dt.astimezone(
                timezone.utc
            )

            return dt.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

        except Exception:
            pass

    # Fallback to current UTC time

    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def fetch_active_storms_gdacs():
    """
    Fetch and normalize active Tropical Cyclone records
    from the GDACS RSS feed.

    Current GDACS TC schema includes:

        eventtype
        eventid
        eventname
        point
        description
        severity
        pubDate

    The function converts these into CycloneAI's
    common normalized schema.
    """

    try:

        # ====================================================
        # FETCH GDACS RSS
        # ====================================================

        response = requests.get(
            GDACS_RSS_URL,
            timeout=15
        )

        response.raise_for_status()

        # ====================================================
        # PARSE XML
        # ====================================================

        root = ET.fromstring(
            response.content
        )

        # Remove gdacs: namespace
        root = strip_namespaces(
            root
        )

        items = root.findall(
            ".//item"
        )

        normalized_records = []

        # ====================================================
        # PROCESS EACH RSS ITEM
        # ====================================================

        for item in items:

            # ------------------------------------------------
            # EVENT TYPE
            # ------------------------------------------------

            event_type = get_text(
                item,
                "eventtype"
            )

            if event_type:

                event_type = (
                    event_type
                    .strip()
                    .upper()
                )

            # Only process Tropical Cyclones

            if event_type != "TC":

                continue

            # ------------------------------------------------
            # EVENT ID
            # ------------------------------------------------

            event_id = get_text(
                item,
                "eventid",
                "event_id"
            )

            # ------------------------------------------------
            # STORM NAME
            # ------------------------------------------------

            storm_name = get_text(
                item,
                "eventname",
                "evname",
                "stormname",
                "storm_name"
            )

            # ------------------------------------------------
            # COORDINATES
            # ------------------------------------------------

            lat_str = get_text(
                item,
                "latitude",
                "lat"
            )

            lon_str = get_text(
                item,
                "longitude",
                "lon"
            )

            # Current GDACS format stores coordinates as:

            # point = "latitude longitude"

            if not lat_str or not lon_str:

                point_str = get_text(
                    item,
                    "point"
                )

                point_lat, point_lon = parse_point(
                    point_str
                )

                if point_lat is not None:

                    lat_str = str(
                        point_lat
                    )

                if point_lon is not None:

                    lon_str = str(
                        point_lon
                    )

            # ------------------------------------------------
            # REQUIRED FIELD VALIDATION
            # ------------------------------------------------

            if (
                not event_id
                or not storm_name
                or not lat_str
                or not lon_str
            ):

                continue

            # ------------------------------------------------
            # CONVERT COORDINATES
            # ------------------------------------------------

            try:

                latitude = float(
                    lat_str
                )

                longitude = float(
                    lon_str
                )

            except (ValueError, TypeError):

                continue

            # ------------------------------------------------
            # WIND SPEED
            # ------------------------------------------------

            wind_kts = parse_wind_speed(
                item
            )

            # ------------------------------------------------
            # PRESSURE
            # ------------------------------------------------

            pressure_hpa = parse_pressure(
                item
            )

            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            pub_date_str = get_text(
                item,
                "pubDate",
                "pubdate"
            )

            timestamp_utc = parse_timestamp(
                pub_date_str
            )

            # ------------------------------------------------
            # REGION CLASSIFICATION
            # ------------------------------------------------

            region = classify_region(
                latitude,
                longitude
            )

            # ------------------------------------------------
            # NORMALIZED RECORD
            # ------------------------------------------------

            record = {

                "source": "GDACS",

                "source_storm_id": str(
                    event_id
                ).strip(),

                "storm_name": str(
                    storm_name
                ).strip().upper(),

                "timestamp_utc": timestamp_utc,

                "latitude": latitude,

                "longitude": longitude,

                "wind_kts": wind_kts,

                "pressure_hpa": pressure_hpa,

                "region": region
            }

            normalized_records.append(
                record
            )

        return normalized_records

    except Exception as e:

        raise RuntimeError(
            f"Error fetching/parsing GDACS RSS feed: {e}"
        )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    try:

        data = fetch_active_storms_gdacs()

        print()
        print("=" * 70)
        print(
            f"Success! Fetched {len(data)} "
            f"Tropical Cyclone records from GDACS."
        )
        print("=" * 70)

        for idx, storm in enumerate(
            data[:10]
        ):

            print(
                f"{idx + 1}. "
                f"{storm['storm_name']} "
                f"({storm['source_storm_id']})"
            )

            print(
                f"   Position: "
                f"({storm['latitude']}, "
                f"{storm['longitude']})"
            )

            print(
                f"   Wind: "
                f"{storm['wind_kts']} kt"
            )

            print(
                f"   Pressure: "
                f"{storm['pressure_hpa']} hPa"
            )

            print(
                f"   Region: "
                f"{storm['region']}"
            )

            print(
                f"   Timestamp: "
                f"{storm['timestamp_utc']}"
            )

            print()

        print("=" * 70)

    except Exception as ex:

        print()
        print("=" * 70)
        print(
            f"Test run failed: {ex}"
        )
        print("=" * 70)