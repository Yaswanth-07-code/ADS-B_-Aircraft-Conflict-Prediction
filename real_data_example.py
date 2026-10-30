"""
real_data_example.py
---------------------
Loads one real scene file from TrajAir's processed_data folder and
runs the CPA baseline on two real aircraft from it - but ONLY if they
were actually flying at the same moment. Comparing two aircraft from
different times would give a meaningless answer, so this specifically
looks for a Frame (a moment in time) where two different aircraft IDs
both have a reading.

Confirmed real file format (checked directly against TrajAir's own
processing code):
  - Plain text, space-separated, NO header row
  - Columns, always in this order: Frame ID x y z Headwind Crosswind
  - Frame  : seconds elapsed (1 reading per second)
  - ID     : which aircraft this row belongs to
  - x, y   : position in KILOMETERS already (no conversion needed)
  - z      : altitude in KILOMETERS (needs converting to feet)
  - Headwind, Crosswind : wind info, not used here

Only line you need to change is FILE_PATH below. Then run:
    python3 real_data_example.py
"""

import pandas as pd
from cpa import check_conflict

FILE_PATH = r"C:\Users\HP\Downloads\14866251\7days1\processed_data\test"

COLUMN_NAMES = ["Frame", "ID", "x", "y", "z", "Headwind", "Crosswind"]
SECONDS_PER_FRAME = 1.0
KM_TO_FEET = 1000.0 / 0.3048


def estimate_velocity_kmh(row_early, row_late, dt_seconds):
    """Speed and direction from two position readings, dt_seconds apart."""
    dx_km = row_late["x"] - row_early["x"]
    dy_km = row_late["y"] - row_early["y"]
    vx_kmh = (dx_km / dt_seconds) * 3600
    vy_kmh = (dy_km / dt_seconds) * 3600
    return vx_kmh, vy_kmh


def get_pos_vel_alt(aircraft_df, frame):
    """
    Pulls out position, velocity, and altitude for one aircraft at a
    specific Frame (moment in time). Velocity is worked out by
    comparing this Frame to the very next one for the same aircraft.
    """
    row_now = aircraft_df[aircraft_df["Frame"] == frame].iloc[0]
    row_next = aircraft_df[aircraft_df["Frame"] == frame + 1]

    if row_next.empty:
        # No "next second" reading available - fall back to comparing
        # against the previous second instead.
        row_prev = aircraft_df[aircraft_df["Frame"] == frame - 1].iloc[0]
        vel = estimate_velocity_kmh(row_prev, row_now, SECONDS_PER_FRAME)
    else:
        vel = estimate_velocity_kmh(row_now, row_next.iloc[0], SECONDS_PER_FRAME)

    pos = (row_now["x"], row_now["y"])
    alt_ft = row_now["z"] * KM_TO_FEET
    return pos, vel, alt_ft


df = pd.read_csv(FILE_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)

aircraft_ids = df["ID"].unique()
print(f"Found {len(aircraft_ids)} different aircraft in this file: {list(aircraft_ids)}")

if len(aircraft_ids) < 2:
    raise SystemExit(
        "This file only has ONE aircraft in it - open the file yourself "
        "and confirm, or try a different scene file from processed_data."
    )

# Find a Frame (moment in time) where at least two different aircraft
# both have a reading, rather than just grabbing the first two IDs.
frame_counts = df.groupby("Frame")["ID"].nunique()
shared_frames = frame_counts[frame_counts >= 2].index

if len(shared_frames) == 0:
    raise SystemExit(
        "No moment in this file has two aircraft airborne at the same "
        "time - try a different scene file from processed_data."
    )

shared_frame = shared_frames[0]
two_ids = df[df["Frame"] == shared_frame]["ID"].unique()[:2]
id_a, id_b = two_ids[0], two_ids[1]

print(f"\nUsing Frame {shared_frame} - aircraft {id_a} and {id_b} were both airborne then.")

data_a = df[df["ID"] == id_a].sort_values("Frame").reset_index(drop=True)
data_b = df[df["ID"] == id_b].sort_values("Frame").reset_index(drop=True)

pos_a, vel_a, alt_a_ft = get_pos_vel_alt(data_a, shared_frame)
pos_b, vel_b, alt_b_ft = get_pos_vel_alt(data_b, shared_frame)

print(f"\nAircraft A ({id_a}) - position (km): {pos_a}, velocity (km/h): {vel_a}, altitude (ft): {alt_a_ft:.0f}")
print(f"Aircraft B ({id_b}) - position (km): {pos_b}, velocity (km/h): {vel_b}, altitude (ft): {alt_b_ft:.0f}")

result = check_conflict(pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft)

print("\nResult:")
for key, value in result.items():
    print(f"  {key}: {value}")
