import pandas as pd
import datetime
from sqlalchemy.exc import SQLAlchemyError

from app.database.connection import SessionLocal
from app.database.models import Cyclone, Observation


CSV_PATH = r"C:\Users\hansh\OneDrive\Desktop\CycloneAI_Data\Team1_IBTrACS\processed\ibtracs_ni_clean_v2.csv"

MIN_LAT = 0
MAX_LAT = 35
MIN_LON = 40
MAX_LON = 110

# Initial integration batch
NUMBER_OF_STORMS = None


def clean_number(series):
    return pd.to_numeric(series, errors="coerce")


def get_basin(subbasins):
    values = {
        str(x).strip().upper()
        for x in subbasins
        if pd.notna(x)
    }

    if "BB" in values:
        return "Bay of Bengal"

    if "AS" in values:
        return "Arabian Sea"

    return "North Indian Ocean"


def get_storm_name(group):
    names = (
        group["NAME"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    named = [
        name for name in names
        if name and name.upper() != "UNNAMED"
    ]

    if named:
        return named[0]

    return "UNNAMED"


print("=" * 70)
print("IBTRACS → CYCLONEAI REAL DATA IMPORT")
print("=" * 70)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\nLoading IBTrACS dataset...")

df = pd.read_csv(
    CSV_PATH,
    low_memory=False
)

print(f"CSV rows: {len(df):,}")

# ---------------------------------------------------------
# NORMALIZE
# ---------------------------------------------------------

df["LAT"] = clean_number(df["LAT"])
df["LON"] = clean_number(df["LON"])

df["WMO_WIND"] = clean_number(df["WMO_WIND"])
df["WMO_PRES"] = clean_number(df["WMO_PRES"])

df["wind_speed"] = clean_number(df["wind_speed"])
df["pressure"] = clean_number(df["pressure"])

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce",
    utc=True
)

# ---------------------------------------------------------
# NIO FILTER
# ---------------------------------------------------------

df = df[
    (df["LAT"] >= MIN_LAT)
    & (df["LAT"] <= MAX_LAT)
    & (df["LON"] >= MIN_LON)
    & (df["LON"] <= MAX_LON)
].copy()

print(f"NIO observations: {len(df):,}")

# ---------------------------------------------------------
# WIND FALLBACK
# ---------------------------------------------------------

df["resolved_wind"] = df["WMO_WIND"]

df.loc[
    df["resolved_wind"].isna(),
    "resolved_wind"
] = df.loc[
    df["resolved_wind"].isna(),
    "wind_speed"
]

# ---------------------------------------------------------
# PRESSURE FALLBACK
# ---------------------------------------------------------

df["resolved_pressure"] = df["WMO_PRES"]

df.loc[
    df["resolved_pressure"].isna(),
    "resolved_pressure"
] = df.loc[
    df["resolved_pressure"].isna(),
    "pressure"
]

# ---------------------------------------------------------
# VALID OBSERVATIONS
# ---------------------------------------------------------

df = df[
    df["SID"].notna()
    & df["ISO_TIME"].notna()
    & df["LAT"].notna()
    & df["LON"].notna()
    & df["resolved_wind"].notna()
    & df["resolved_pressure"].notna()
].copy()

print(f"Valid observations: {len(df):,}")

# ---------------------------------------------------------
# GROUP STORMS
# ---------------------------------------------------------

storm_groups = []

for sid, group in df.groupby("SID"):

    group = group.sort_values("ISO_TIME")

    name = get_storm_name(group)

    # We want named storms for the first integration batch
    if name == "UNNAMED":
        continue

    basin = get_basin(group["SUBBASIN"].unique())

    storm_groups.append({
        "sid": str(sid),
        "name": name,
        "basin": basin,
        "season": int(group["SEASON"].iloc[0]),
        "observations": len(group),
        "start": group["ISO_TIME"].min(),
        "end": group["ISO_TIME"].max(),
        "data": group,
    })

# Highest-observation storms first
storm_groups.sort(
    key=lambda x: x["observations"],
    reverse=True
)

selected = storm_groups if NUMBER_OF_STORMS is None else storm_groups[:NUMBER_OF_STORMS]

print("\n" + "=" * 70)
print(f"SELECTED {len(selected)} NAMED STORMS")
print("=" * 70)

for i, storm in enumerate(selected, 1):

    print(
        f"{i:2}. "
        f"{storm['name']:<15} | "
        f"{storm['basin']:<15} | "
        f"{storm['observations']:>4} obs | "
        f"{storm['start']} → {storm['end']} | "
        f"SID={storm['sid']}"
    )

# ---------------------------------------------------------
# DATABASE IMPORT
# ---------------------------------------------------------

db = SessionLocal()

inserted_cyclones = 0
existing_cyclones = 0
inserted_observations = 0
duplicate_observations = 0

try:

    print("\n" + "=" * 70)
    print("IMPORTING INTO DATABASE")
    print("=" * 70)

    for storm in selected:

        cyclone_id = f"IB_{storm['sid']}"

        # -------------------------------------------------
        # CYCLONE
        # -------------------------------------------------

        cyclone = (
            db.query(Cyclone)
            .filter(Cyclone.id == cyclone_id)
            .first()
        )

        if cyclone:

            existing_cyclones += 1

            print(
                f"\nEXISTS: {storm['name']} "
                f"({cyclone_id})"
            )

        else:

            cyclone = Cyclone(
                id=cyclone_id,
                name=storm["name"],
                basin=storm["basin"],
                status="Dissipated",
                start_time=storm["start"].to_pydatetime().replace(
                    tzinfo=None
                ),
                end_time=storm["end"].to_pydatetime().replace(
                    tzinfo=None
                ),
            )

            db.add(cyclone)
            db.flush()

            inserted_cyclones += 1

            print(
                f"\nINSERTED CYCLONE: "
                f"{storm['name']} ({cyclone_id})"
            )

        # -------------------------------------------------
        # OBSERVATIONS
        # -------------------------------------------------

        for _, row in storm["data"].iterrows():

            timestamp = row["ISO_TIME"].to_pydatetime().replace(
                tzinfo=None
            )

            # Application-level duplicate prevention
            existing_obs = (
                db.query(Observation)
                .filter(
                    Observation.cyclone_id == cyclone_id,
                    Observation.timestamp == timestamp
                )
                .first()
            )

            if existing_obs:

                duplicate_observations += 1
                continue

            observation = Observation(
                cyclone_id=cyclone_id,
                timestamp=timestamp,
                latitude=float(row["LAT"]),
                longitude=float(row["LON"]),
                wind_speed=float(row["resolved_wind"]),
                pressure=float(row["resolved_pressure"]),
                dvorak_t_number=None,
                pattern_type="Historical IBTrACS",
                image_path=None,
                is_synthetic=False,
            )

            db.add(observation)

            inserted_observations += 1

    db.commit()

    print("\n" + "=" * 70)
    print("IMPORT SUCCESSFUL")
    print("=" * 70)

    print(f"""
Cyclones inserted:          {inserted_cyclones}
Cyclones already existing:  {existing_cyclones}

Observations inserted:      {inserted_observations}
Duplicate observations:     {duplicate_observations}

Database changes committed.
""")

except SQLAlchemyError as e:

    db.rollback()

    print("\n" + "=" * 70)
    print("IMPORT FAILED")
    print("=" * 70)

    print("Database transaction rolled back.")
    print(f"\nError: {e}")

    raise

except Exception as e:

    db.rollback()

    print("\n" + "=" * 70)
    print("IMPORT FAILED")
    print("=" * 70)

    print("Database transaction rolled back.")
    print(f"\nError: {e}")

    raise

finally:

    db.close()

print("=" * 70)
print("DONE")
print("=" * 70)