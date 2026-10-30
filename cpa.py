"""
cpa.py
------
Deterministic Closest Point of Approach (CPA) baseline.

This is pure geometry: no training, no ML, no external data needed.
Given two aircraft's current position and velocity, it answers:
  - When will they be closest to each other? (time_to_closest_approach)
  - How close will they actually get? (minimum_distance_km)
  - Does that violate separation minima? (conflict)

Build and trust this file BEFORE plugging in any real trajectory data.
"""

import numpy as np

# Standard enroute separation minima
HORIZONTAL_MINIMA_NM = 5.0
HORIZONTAL_MINIMA_KM = HORIZONTAL_MINIMA_NM * 1.852  # 1 nm = 1.852 km
VERTICAL_MINIMA_FT = 1000.0


def closest_point_of_approach(pos_a, vel_a, pos_b, vel_b):
    """
    pos_a, pos_b : (x, y) position in km
    vel_a, vel_b : (vx, vy) velocity in km/h

    Returns (time_to_closest_approach_hours, minimum_distance_km):
      time_to_closest_approach_hours - time until the two aircraft are
                                        nearest each other
      minimum_distance_km            - the horizontal distance between
                                        them at that moment
    """
    pos_a = np.array(pos_a, dtype=float)
    pos_b = np.array(pos_b, dtype=float)
    vel_a = np.array(vel_a, dtype=float)
    vel_b = np.array(vel_b, dtype=float)

    relative_position = pos_b - pos_a   # how far apart they are right now
    relative_velocity = vel_b - vel_a   # how differently they're moving (closing vector)

    relative_speed_sq = np.dot(relative_velocity, relative_velocity)

    if relative_speed_sq < 1e-9:
        # Aircraft are moving at (near) identical velocity.
        # Their separation never changes, so "closest approach" is now.
        time_to_closest_approach = 0.0
    else:
        # This is the one equation from the theory - just distance/speed
        # dressed up as a dot product so it works at any angle, not just
        # head-on.
        time_to_closest_approach = -np.dot(relative_position, relative_velocity) / relative_speed_sq

    if time_to_closest_approach < 0:
        # The closest point was in the past - use the current distance.
        time_to_closest_approach = 0.0

    future_relative_position = relative_position + relative_velocity * time_to_closest_approach
    minimum_distance_km = np.linalg.norm(future_relative_position)

    return time_to_closest_approach, minimum_distance_km


def check_conflict(pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft):
    """
    Runs the full baseline check for one aircraft pair and returns a
    plain dictionary of results - easy to print, log, or feed into a
    visualization.
    """
    time_to_closest_approach, minimum_distance_km = closest_point_of_approach(
        pos_a, vel_a, pos_b, vel_b
    )
    vertical_gap_ft = abs(alt_a_ft - alt_b_ft)

    horizontal_violation = minimum_distance_km < HORIZONTAL_MINIMA_KM
    vertical_violation = vertical_gap_ft < VERTICAL_MINIMA_FT

    # Separation standards: a conflict requires BOTH horizontal AND
    # vertical minima to be violated at the same time. Either one being
    # respected keeps the aircraft safely separated.
    conflict = horizontal_violation and vertical_violation

    return {
        "time_to_closest_approach_minutes": round(time_to_closest_approach * 60, 2),
        "minimum_horizontal_distance_nm": round(minimum_distance_km / 1.852, 2),
        "vertical_gap_ft": round(vertical_gap_ft, 1),
        "horizontal_violation": horizontal_violation,
        "vertical_violation": vertical_violation,
        "conflict": conflict,
    }


# ---------------------------------------------------------------------
# This part ONLY runs when you execute cpa.py directly (not when
# demo.py or anything else imports it). Change the numbers below to
# whatever you want and re-run this file to see the result instantly.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Aircraft A
    pos_a = (0, 0)        # (east-west km, north-south km)
    vel_a = (300, 0)      # (east-west km/h, north-south km/h)
    alt_a_ft = 35000       # altitude in feet

    # Aircraft B
    pos_b = (60, 8)
    vel_b = (-300, 0)
    alt_b_ft = 35000

    result = check_conflict(pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft)

    print("Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")