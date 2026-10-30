"""
h3_spatial.py
--------------
H3 Hexagonal Spatial Indexing for ADS-B Conflict Prediction.

WHY: Instead of checking every possible aircraft pair (O(N^2) which is slow
when tracking dozens of aircraft), we divide the airspace into hexagonal cells.
We only evaluate pairs of aircraft that are in the same cell or
neighboring cells. This drastically reduces computation and mirrors
how real Air Traffic Control (ATC) spatial sectors operate.

WHAT H3 IS:
  H3 is Uber's discrete global hexagonal grid system. It partitions the
  earth into hierarchical hexagonal cells.
  - Resolution 6: edge length ~3.7 km, cell diameter ~7.4 km, area ~36.4 km2.
  - With a 1-ring neighborhood (immediate adjacent hexagons), the covered
    diameter is ~15-20 km, which safely encloses our 5 nm (9.26 km) conflict
    separation boundary without missing any potential conflicts.

HOW IT CONNECTS:
  Dataset -> H3 (group aircraft into cells & filter nearby candidate pairs)
          -> Uncertainty -> Monte Carlo -> Conflict Probability -> Dashboard

Run standalone:
    python h3_spatial.py                     # runs cluster demo
    python h3_spatial.py --file test/78.txt  # inspects hexagons in real flight file
"""

import os
import sys
import argparse
import math
from collections import defaultdict
import h3

# --- Configuration ---
H3_RESOLUTION = 6
K_RING_NEIGHBORS = 1

# Reference origin: Pittsburgh-Butler Regional Airport (approximate coordinates for TrajAir)
AIRPORT_LAT = 40.7767
AIRPORT_LON = -79.9497

# Conversion factor: 1 degree latitude ~ 111 km
KM_PER_DEG_LAT = 111.0
DATA_ROOT = r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data"


def km_to_latlon(x_km, y_km, ref_lat=AIRPORT_LAT, ref_lon=AIRPORT_LON):
    """
    Converts local Cartesian coordinates (km from airport) to lat/lon.
    """
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(ref_lat))
    lat = ref_lat + (y_km / KM_PER_DEG_LAT)
    lon = ref_lon + (x_km / km_per_deg_lon)
    return lat, lon


def get_h3_cell(x_km, y_km, resolution=H3_RESOLUTION):
    """
    Returns the H3 hexagonal cell ID for a position given in local km.
    """
    lat, lon = km_to_latlon(x_km, y_km)
    return h3.latlng_to_cell(lat, lon, resolution)


def get_h3_neighbors(cell_id, k=K_RING_NEIGHBORS):
    """
    Returns the cell itself + all neighboring cells within k rings.
    """
    return h3.grid_disk(cell_id, k)


def group_aircraft_by_cell(aircraft_positions, resolution=H3_RESOLUTION):
    """
    Groups aircraft into H3 cells based on their current positions.

    Input:
      aircraft_positions: dict of {aircraft_id: (x_km, y_km)}

    Output:
      cell_to_aircraft: dict of {cell_id: [aircraft_id, ...]}
      aircraft_to_cell: dict of {aircraft_id: cell_id}
    """
    cell_to_aircraft = defaultdict(list)
    aircraft_to_cell = {}

    for ac_id, (x_km, y_km) in aircraft_positions.items():
        cell = get_h3_cell(x_km, y_km, resolution=resolution)
        cell_to_aircraft[cell].append(ac_id)
        aircraft_to_cell[ac_id] = cell

    return dict(cell_to_aircraft), aircraft_to_cell


def get_candidate_pairs(aircraft_positions, resolution=H3_RESOLUTION, k_ring=K_RING_NEIGHBORS):
    """
    Filters aircraft pairs to only those within same or neighboring H3 cells.
    Eliminates distant aircraft pairs that cannot physically conflict.

    Input:
      aircraft_positions: dict of {aircraft_id: (x_km, y_km)}

    Output:
      List of (id_a, id_b) pairs that require conflict analysis.
    """
    cell_to_aircraft, aircraft_to_cell = group_aircraft_by_cell(aircraft_positions, resolution)
    candidate_pairs = set()

    for cell_id, ac_ids in cell_to_aircraft.items():
        # Aircraft in the same cell can always conflict
        for i in range(len(ac_ids)):
            for j in range(i + 1, len(ac_ids)):
                pair = tuple(sorted([ac_ids[i], ac_ids[j]]))
                candidate_pairs.add(pair)

        # Aircraft in neighboring cells
        neighbor_cells = get_h3_neighbors(cell_id, k=k_ring)
        for n_cell in neighbor_cells:
            if n_cell == cell_id or n_cell not in cell_to_aircraft:
                continue
            for a in ac_ids:
                for b in cell_to_aircraft[n_cell]:
                    pair = tuple(sorted([a, b]))
                    candidate_pairs.add(pair)

    return sorted(list(candidate_pairs))


def get_cell_info(cell_id):
    """
    Returns geographical metadata for an H3 cell.
    """
    boundary = h3.cell_to_boundary(cell_id)
    center = h3.cell_to_latlng(cell_id)
    area_km2 = h3.cell_area(cell_id, unit='km^2')

    return {
        "cell_id": cell_id,
        "center_lat": round(center[0], 5),
        "center_lon": round(center[1], 5),
        "area_km2": round(area_km2, 2),
        "boundary": boundary,
    }


def inspect_file(file_path):
    """
    Inspects which hexagons contain which aircraft in a real flight file.
    """
    from data_utils import load_scene_file, get_pos_vel_alt

    if not os.path.isabs(file_path):
        file_path = os.path.join(DATA_ROOT, file_path)

    print("=" * 70)
    print(f"H3 SPATIAL INSPECTION: {os.path.basename(file_path)}")
    print(f"Hexagonal Resolution: {H3_RESOLUTION} (Cell diameter ~7.4 km, Area ~36.4 km2)")
    print("=" * 70)

    df = load_scene_file(file_path)
    all_ids = df["ID"].unique()
    positions = {}
    flight_data = {}

    for ac_id in all_ids:
        ac_df = df[df["ID"] == ac_id].sort_values("Frame").reset_index(drop=True)
        mid_frame = ac_df["Frame"].iloc[len(ac_df) // 2]
        info = get_pos_vel_alt(ac_df, mid_frame)
        if info:
            positions[ac_id] = info[0]
            flight_data[ac_id] = (info[1], info[2], mid_frame)

    cell_to_aircraft, _ = group_aircraft_by_cell(positions)
    candidate_pairs = get_candidate_pairs(positions)
    total_pairs = len(positions) * (len(positions) - 1) // 2

    print(f"\nTotal Aircraft Found: {len(positions)}")
    print(f"Total H3 Hexagons Active: {len(cell_to_aircraft)}\n")

    for idx, (cell_id, ac_ids) in enumerate(cell_to_aircraft.items(), 1):
        info = get_cell_info(cell_id)
        print(f"----------------------------------------------------------------------")
        print(f"HEXAGON #{idx}: Cell ID = {cell_id}")
        print(f"  Geographic Center: ({info['center_lat']}, {info['center_lon']}) | Cell Area: {info['area_km2']} km2")
        print(f"  Aircraft Count Inside This Hexagon: {len(ac_ids)}")
        print(f"  Flight Details:")
        for ac in ac_ids:
            p = positions[ac]
            v, alt, frame = flight_data[ac]
            speed = math.sqrt(v[0]**2 + v[1]**2)
            print(f"    * Aircraft {ac:<10} | Pos=({p[0]:>5.2f}, {p[1]:>5.2f}) km | Alt={alt:>5.0f} ft | Speed={speed:>5.1f} km/h")

    print("\n" + "=" * 70)
    print("PAIR FILTERING ANALYSIS FOR THIS SCENARIO:")
    print(f"  - Total possible pairs without H3: {total_pairs}")
    print(f"  - Candidate pairs to check:        {len(candidate_pairs)}")
    pruned = total_pairs - len(candidate_pairs)
    print(f"  - Pruned (distant pairs):          {pruned} ({(pruned/total_pairs)*100:.0f}% saved)" if total_pairs > 0 else "")
    print("=" * 70)


# ---- Self-test ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H3 Spatial Indexing Inspector")
    parser.add_argument("--file", type=str, default="test/78.txt", help="Path to real scene file (default: test/78.txt)")
    parser.add_argument("--demo", action="store_true", help="Run synthetic 2-cluster demonstration")
    args = parser.parse_args()

    if args.demo or args.file.lower() == "demo":
        print("=" * 60)
        print("H3 Spatial Indexing - Self Test (2 Distant Clusters)")
        print("=" * 60)

        positions = {
            "AC_1": (0.5, 0.3),    # cluster A (near airport)
            "AC_2": (0.8, 0.5),    # cluster A
            "AC_3": (15.0, 12.0),  # cluster B (19 km away)
            "AC_4": (0.2, -0.1),   # cluster A
            "AC_5": (14.5, 11.8),  # cluster B
        }

        print("\nAircraft positions (km from airport):")
        for ac_id, pos in positions.items():
            cell = get_h3_cell(pos[0], pos[1])
            lat, lon = km_to_latlon(pos[0], pos[1])
            print(f"  {ac_id}: ({pos[0]:>5.1f}, {pos[1]:>5.1f}) km -> ({lat:.4f}, {lon:.4f}) -> H3 {cell}")

        print("\nH3 cell grouping (Resolution 6):")
        cell_groups, _ = group_aircraft_by_cell(positions)
        for cell_id, ac_ids in cell_groups.items():
            info = get_cell_info(cell_id)
            print(f"  Cell {cell_id} (Center: {info['center_lat']}, {info['center_lon']}): {ac_ids}")

        print("\nCandidate pairs (after H3 spatial filtering):")
        pairs = get_candidate_pairs(positions)
        total_possible = len(positions) * (len(positions) - 1) // 2
        for a, b in pairs:
            print(f"  [CHECK] {a} <-> {b}")

        print(f"\nFiltering summary: checked {len(pairs)} / {total_possible} pairs "
              f"({(1 - len(pairs)/total_possible)*100:.0f}% pairs pruned by H3)")
    else:
        inspect_file(args.file)
