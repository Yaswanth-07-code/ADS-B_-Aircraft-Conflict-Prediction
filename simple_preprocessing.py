"""
data_utils.py
--------------
Shared helpers for reading real TrajAir scene files. Used by both
real_data_example.py (one pair) and multi_aircraft_demo.py (several
aircraft at once), so there's only one trusted place this logic lives.
"""

import pandas as pd

COLUMN_NAMES = ["Frame", "ID", "x", "y", "z", "Headwind", "Crosswind"]
SECONDS_PER_FRAME = 1.0
KM_TO_FEET = 1000.0 / 0.3048


def load_scene_file(file_path):
    r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data\test\177.txt"
    return pd.read_csv(file_path, sep=r"\s+", header=None, names=COLUMN_NAMES)


def estimate_velocity_kmh(row_early, row_late, dt_seconds=SECONDS_PER_FRAME):
    """Speed and direction from two position readings, dt_seconds apart."""
    dx_km = row_late["x"] - row_early["x"]
    dy_km = row_late["y"] - row_early["y"]
    vx_kmh = (dx_km / dt_seconds) * 3600
    vy_kmh = (dy_km / dt_seconds) * 3600
    return vx_kmh, vy_kmh


def get_pos_vel_alt(aircraft_df, frame):
    """
    Position, velocity, and altitude for one aircraft at a specific
    Frame (a moment in time). Velocity comes from comparing this
    Frame to the next one for the same aircraft (or the previous one,
    if there's no "next" available).
    """
    row_now = aircraft_df[aircraft_df["Frame"] == frame].iloc[0]
    row_next = aircraft_df[aircraft_df["Frame"] == frame + 1]

    if row_next.empty:
        row_prev = aircraft_df[aircraft_df["Frame"] == frame - 1]
        if row_prev.empty:
            return None  # can't work out a velocity at all - skip this one
        vel = estimate_velocity_kmh(row_prev.iloc[0], row_now)
    else:
        vel = estimate_velocity_kmh(row_now, row_next.iloc[0])

    pos = (row_now["x"], row_now["y"])
    alt_ft = row_now["z"] * KM_TO_FEET
    return pos, vel, alt_ft