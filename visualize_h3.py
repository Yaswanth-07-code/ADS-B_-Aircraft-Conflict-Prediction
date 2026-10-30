"""
visualize_h3.py
----------------
Generates a presentation-ready diagram of H3 Hexagonal Spatial Indexing.
Plots aircraft positions on top of the discrete H3 hexagonal grid cells,
showing how H3 filters candidate conflict pairs and prunes distant aircraft.

Run with:
    python visualize_h3.py
    python visualize_h3.py --file test/78.txt
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import h3
from data_utils import load_scene_file, get_pos_vel_alt
from h3_spatial import (
    km_to_latlon, get_h3_cell, group_aircraft_by_cell,
    get_candidate_pairs, H3_RESOLUTION, AIRPORT_LAT, AIRPORT_LON, KM_PER_DEG_LAT
)

DEFAULT_FILE = r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data\test\78.txt"
DATA_ROOT = r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data"


def latlon_to_km(lat, lon, ref_lat=AIRPORT_LAT, ref_lon=AIRPORT_LON):
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(ref_lat))
    y_km = (lat - ref_lat) * KM_PER_DEG_LAT
    x_km = (lon - ref_lon) * km_per_deg_lon
    return x_km, y_km


def plot_h3_grid(file_path=None, output_png="h3_grid_visualization.png", max_aircraft=8):
    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)

    # 1. Obtain aircraft positions
    aircraft_positions = {}
    aircraft_info = {}

    if file_path and os.path.exists(file_path):
        print(f"Loading real flight data from: {os.path.basename(file_path)}")
        df = load_scene_file(file_path)
        all_ids = list(df["ID"].unique())[:max_aircraft]
        for ac_id in all_ids:
            ac_df = df[df["ID"] == ac_id].sort_values("Frame").reset_index(drop=True)
            mid_frame = ac_df["Frame"].iloc[len(ac_df) // 2]
            info = get_pos_vel_alt(ac_df, mid_frame)
            if info:
                aircraft_positions[str(ac_id)] = info[0]
                aircraft_info[str(ac_id)] = (info[1], info[2])  # vel, alt
    else:
        # Example demonstration scenario with 2 spatial clusters
        print("Using demonstration multi-aircraft scenario...")
        aircraft_positions = {
            "AC_101": (0.8, 1.2),
            "AC_102": (-1.5, 0.4),
            "AC_103": (1.1, -0.9),
            "AC_104": (2.3, 1.8),
            "AC_201": (14.2, 11.5),
            "AC_202": (15.8, 10.2),
            "AC_203": (13.5, 13.1),
        }
        for k in aircraft_positions:
            aircraft_info[k] = ((150.0, 0.0), 2500.0)

    # 2. H3 Grouping & Cells
    cell_to_aircraft, aircraft_to_cell = group_aircraft_by_cell(aircraft_positions, resolution=H3_RESOLUTION)
    candidate_pairs = get_candidate_pairs(aircraft_positions, resolution=H3_RESOLUTION)
    total_pairs = len(aircraft_positions) * (len(aircraft_positions) - 1) // 2

    # Get set of occupied cells plus their 1-ring neighbors to draw the surrounding grid
    cells_to_draw = set()
    for cell in cell_to_aircraft.keys():
        neighbors = h3.grid_disk(cell, 1)
        cells_to_draw.update(neighbors)

    # 3. Draw Hexagons
    drawn_legend_cells = set()
    for cell in cells_to_draw:
        # Get boundary vertices in (lat, lon)
        boundary_latlon = h3.cell_to_boundary(cell)
        # Convert boundary to local km coordinates
        poly_km = [latlon_to_km(lat, lon) for lat, lon in boundary_latlon]

        is_occupied = cell in cell_to_aircraft
        if is_occupied:
            facecolor = "#D0E1FD"  # soft pastel blue
            edgecolor = "#185ABC"  # deep blue
            linewidth = 2.0
            alpha = 0.55
            label = "Active H3 Cell (Aircraft Present)" if "active" not in drawn_legend_cells else None
            drawn_legend_cells.add("active")
        else:
            facecolor = "#F8F9FA"  # very light gray
            edgecolor = "#9AA0A6"  # medium gray
            linewidth = 1.0
            alpha = 0.3
            label = "Neighboring H3 Cell (Empty Airspace)" if "empty" not in drawn_legend_cells else None
            drawn_legend_cells.add("empty")

        polygon = Polygon(poly_km, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=linewidth, alpha=alpha, label=label)
        ax.add_patch(polygon)

        # Label cell ID at center
        center_lat, center_lon = h3.cell_to_latlng(cell)
        cx, cy = latlon_to_km(center_lat, center_lon)
        cell_short = cell[-6:]  # short hex suffix
        ax.text(cx, cy, f"Cell\n..{cell_short}", fontsize=8, color="#5F6368",
                ha="center", va="center", alpha=0.7)

    # 4. Draw Connecting Lines for Candidate Pairs (Green)
    for id_a, id_b in candidate_pairs:
        pos_a = aircraft_positions[id_a]
        pos_b = aircraft_positions[id_b]
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], color="#1E8E3E",
                linestyle="-", linewidth=1.8, alpha=0.75,
                label="Candidate Pair (Check CPA / MLP)" if "candidate" not in drawn_legend_cells else None)
        drawn_legend_cells.add("candidate")

    # Draw Pruned / Distant Lines (Dashed Gray/Red) for filtered pairs
    all_ids = list(aircraft_positions.keys())
    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            pair = tuple(sorted([all_ids[i], all_ids[j]]))
            if pair not in candidate_pairs:
                pos_a = aircraft_positions[pair[0]]
                pos_b = aircraft_positions[pair[1]]
                ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], color="#D93025",
                        linestyle=":", linewidth=1.0, alpha=0.35,
                        label="Pruned Pair (Too Far Apart)" if "pruned" not in drawn_legend_cells else None)
                drawn_legend_cells.add("pruned")

    # 5. Plot Aircraft Positions
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(aircraft_positions), 10)))
    for idx, (ac_id, pos) in enumerate(aircraft_positions.items()):
        color = colors[idx % len(colors)]
        ax.scatter(pos[0], pos[1], s=130, color="#1A73E8", edgecolor="#0D47A1",
                   linewidth=1.5, zorder=5)

        # Label aircraft
        alt_str = ""
        if ac_id in aircraft_info:
            alt_str = f"\n{aircraft_info[ac_id][1]:.0f}ft"
        ax.annotate(f"  {ac_id}{alt_str}", (pos[0], pos[1]),
                    fontsize=8.5, fontweight="bold", color="#202124",
                    va="center", zorder=6)

    # 6. Formatting & Summary Box
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("Local X Position (km East-West)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Local Y Position (km North-South)", fontsize=11, fontweight="bold")
    ax.set_title("H3 Hexagonal Spatial Indexing for ADS-B Conflict Detection\n"
                 f"Airspace Partitioning (Resolution {H3_RESOLUTION})",
                 fontsize=13, fontweight="bold", pad=12)

    # Information Box
    pruned_count = total_pairs - len(candidate_pairs)
    prune_pct = (pruned_count / total_pairs * 100) if total_pairs > 0 else 0
    info_text = (
        f"Spatial Indexing Summary:\n"
        f"------------------------\n"
        f"- Aircraft Count:     {len(aircraft_positions)}\n"
        f"- H3 Cells Occupied:  {len(cell_to_aircraft)}\n"
        f"- Total Pairs:        {total_pairs}\n"
        f"- Candidate Pairs:    {len(candidate_pairs)}\n"
        f"- Pruned by H3:       {pruned_count} ({prune_pct:.0f}% saved)"
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="#FFFFFF", edgecolor="#185ABC", alpha=0.9, linewidth=1.2)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props, fontfamily="monospace")

    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    print(f"H3 visualization successfully generated: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate H3 Spatial Indexing plot")
    parser.add_argument("--file", type=str, default=DEFAULT_FILE, help="Path to scene .txt file (or omit for demo)")
    parser.add_argument("--output", type=str, default="h3_grid_visualization.png", help="Output PNG path")
    args = parser.parse_args()

    target_file = args.file
    if target_file and not os.path.isabs(target_file):
        target_file = os.path.join(DATA_ROOT, target_file)

    plot_h3_grid(target_file, args.output)
