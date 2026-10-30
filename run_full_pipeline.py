"""
run_full_pipeline.py
---------------------
Full Phase 2 ADS-B Conflict Prediction Pipeline.

Executes the complete flow:
  TrajAir Dataset -> H3 Spatial Indexing -> Position Uncertainty
                  -> Monte Carlo Simulation -> Conflict Probability
                  -> Side-by-Side Comparison & Risk Scoring

Run with:
    python run_full_pipeline.py
    python run_full_pipeline.py --file test/78.txt
"""

import os
import sys
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_scene_file, get_pos_vel_alt
from h3_spatial import group_aircraft_by_cell, get_candidate_pairs, get_cell_info, H3_RESOLUTION
from uncertainty import add_uncertainty, HORIZONTAL_STD_KM, VERTICAL_STD_FT
from monte_carlo import monte_carlo_conflict
from conflict_probability import compute_conflict_probability, load_lstm
from cpa import closest_point_of_approach, HORIZONTAL_MINIMA_KM

DEFAULT_FILE = r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data\test\177.txt"
DATA_ROOT = r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data"


def find_shared_frame(df, id_a, id_b):
    frames_a = set(df[df["ID"] == id_a]["Frame"])
    frames_b = set(df[df["ID"] == id_b]["Frame"])
    shared = frames_a & frames_b
    return min(shared) if shared else None


def run_pipeline(file_path, n_simulations=500, max_aircraft=10):
    print("=" * 70)
    print("PHASE 2: ADS-B CONFLICT PREDICTION PIPELINE")
    print("Flow: Dataset -> H3 -> Uncertainty -> Monte Carlo -> Conflict Probability")
    print("=" * 70)

    # 1. Dataset Loading
    print(f"\n[STEP 1] Loading Dataset: {os.path.basename(file_path)}")
    df = load_scene_file(file_path)
    all_ids = df["ID"].unique()
    aircraft_ids = list(all_ids[:max_aircraft])
    print(f"  Found {len(all_ids)} aircraft total in file. Analyzing {len(aircraft_ids)}: {aircraft_ids}")

    # Extract positions at median frame for each aircraft for H3 spatial indexing
    aircraft_positions = {}
    aircraft_data = {}
    for ac_id in aircraft_ids:
        ac_df = df[df["ID"] == ac_id].sort_values("Frame").reset_index(drop=True)
        mid_frame = ac_df["Frame"].iloc[len(ac_df) // 2]
        info = get_pos_vel_alt(ac_df, mid_frame)
        if info:
            pos, vel, alt_ft = info
            aircraft_positions[ac_id] = pos
            aircraft_data[ac_id] = ac_df

    # 2. H3 Spatial Indexing
    print(f"\n[STEP 2] H3 Hexagonal Spatial Indexing (Resolution {H3_RESOLUTION})")
    h3_cells, aircraft_to_cell = group_aircraft_by_cell(aircraft_positions)
    candidate_pairs = get_candidate_pairs(aircraft_positions)
    total_possible_pairs = len(aircraft_ids) * (len(aircraft_ids) - 1) // 2

    print(f"  Airspace partitioned into {len(h3_cells)} active H3 hex cell(s):")
    for cell_id, ac_list in h3_cells.items():
        info = get_cell_info(cell_id)
        print(f"    - Cell {cell_id} ({info['area_km2']} km2): Aircraft {ac_list}")

    pruned = total_possible_pairs - len(candidate_pairs)
    print(f"  H3 Candidate Pair Filtering:")
    print(f"    Total possible pairs: {total_possible_pairs}")
    print(f"    Candidate pairs to check: {len(candidate_pairs)}")
    if total_possible_pairs > 0:
        print(f"    Pairs pruned (too far apart): {pruned} ({(pruned/total_possible_pairs)*100:.0f}% savings)")

    # 3. Model Loading
    print("\n[STEP 3] Loading LSTM Model, Scaler & Config")
    lstm_model, lstm_scaler, lstm_config = load_lstm()
    if lstm_model:
        print("  LSTM neural network (10 timesteps x 14 features, 98.3% accuracy) loaded successfully")
    else:
        print("  WARNING: lstm_model.pth not found. Running with CPA + Monte Carlo.")

    # 4. Uncertainty & Monte Carlo & Conflict Probability
    print(f"\n[STEP 4] Uncertainty Modeling & Monte Carlo Simulation ({n_simulations} iterations/pair)")
    print(f"  Sensor noise model: GPS Horiz std = {HORIZONTAL_STD_KM*1000:.0f}m, Vert std = {VERTICAL_STD_FT:.0f}ft")

    results = []
    for id_a, id_b in candidate_pairs:
        if id_a not in aircraft_data or id_b not in aircraft_data:
            continue

        shared_frame = find_shared_frame(df, id_a, id_b)
        if shared_frame is None:
            continue

        info_a = get_pos_vel_alt(aircraft_data[id_a], shared_frame)
        info_b = get_pos_vel_alt(aircraft_data[id_b], shared_frame)
        if info_a is None or info_b is None:
            continue

        pos_a, vel_a, alt_a_ft = info_a
        pos_b, vel_b, alt_b_ft = info_b

        prob_info = compute_conflict_probability(
            pos_a, vel_a, alt_a_ft,
            pos_b, vel_b, alt_b_ft,
            lstm_model=lstm_model,
            lstm_scaler=lstm_scaler,
            lstm_config=lstm_config,
            n_simulations=n_simulations
        )

        results.append({
            "id_a": id_a, "id_b": id_b, "frame": shared_frame,
            "pos_a": pos_a, "vel_a": vel_a, "alt_a_ft": alt_a_ft,
            "pos_b": pos_b, "vel_b": vel_b, "alt_b_ft": alt_b_ft,
            **prob_info,
        })

    # Sort results by final probability
    results.sort(key=lambda x: x["final_probability"], reverse=True)

    # 5. Output Summary Table
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS: COMPREHENSIVE CONFLICT RISK SUMMARY")
    print("=" * 70)
    print(f"{'Pair':<20} | {'CPA Baseline':<12} | {'LSTM AI':<8} | {'Monte Carlo':<12} | {'Final Prob':<10} | {'Risk Level'}")
    print("-" * 70)

    for r in results:
        cpa_str = "CONFLICT" if r["cpa_conflict"] else "SAFE"
        lstm_str = f"{r['lstm_probability']:.0%}" if r["lstm_probability"] is not None else "N/A"
        mc_str = f"{r['mc_probability']:.0%}"
        final_str = f"{r['final_probability']:.0%}"
        pair_str = f"{r['id_a']} vs {r['id_b']}"
        print(f"{pair_str:<20} | {cpa_str:<12} | {lstm_str:<8} | {mc_str:<12} | {final_str:<10} | {r['risk_level']}")

    # 6. Detailed High Risk Pairs
    high_risks = [r for r in results if r["risk_level"] == "HIGH"]
    print("\n" + "-" * 70)
    print(f"Risk Breakdown: {len(high_risks)} HIGH | "
          f"{sum(1 for r in results if r['risk_level'] == 'MEDIUM')} MEDIUM | "
          f"{sum(1 for r in results if r['risk_level'] == 'LOW')} LOW")

    if high_risks:
        print("\nTop Conflict Warning:")
        top = high_risks[0]
        mc_det = top["mc_details"]
        print(f"  CRITICAL: Aircraft {top['id_a']} vs {top['id_b']} at Frame {top['frame']}")
        print(f"    - Closest approach horizontal dist: {top['cpa_min_distance_nm']} nm (Separation rule: 5.0 nm)")
        print(f"    - Vertical separation:              {top['cpa_vertical_gap_ft']} ft (Separation rule: 1000 ft)")
        print(f"    - Monte Carlo conflicts:            {mc_det['n_conflicts']}/{mc_det['n_simulations']} runs ({top['mc_probability']:.1%})")
        print(f"    - 95th Percentile min distance:     {mc_det['p95_min_distance_nm']} nm")
        print(f"    - Final Conflict Probability:       {top['final_probability']:.1%}")

    # 7. Visualization
    save_pipeline_plot(results, "pipeline_conflict_result.png")
    return results


def save_pipeline_plot(results, filename="pipeline_conflict_result.png"):
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    plotted = set()

    for r in results:
        pos_a, vel_a = np.array(r["pos_a"]), np.array(r["vel_a"])
        pos_b, vel_b = np.array(r["pos_b"]), np.array(r["vel_b"])
        time_to_cpa, _ = closest_point_of_approach(pos_a, vel_a, pos_b, vel_b)
        t_vals = np.linspace(0, max(time_to_cpa * 1.5, 0.03), 30)

        path_a = np.array([pos_a + vel_a * t for t in t_vals])
        path_b = np.array([pos_b + vel_b * t for t in t_vals])

        prob = r["final_probability"]
        if prob >= 0.7:
            color = "#E24B4A"  # Red
        elif prob >= 0.3:
            color = "#E8A838"  # Orange/Yellow
        else:
            color = "#3FA66B"  # Green

        label_a = f"Aircraft {r['id_a']}" if r["id_a"] not in plotted else None
        label_b = f"Aircraft {r['id_b']}" if r["id_b"] not in plotted else None
        plotted.add(r["id_a"])
        plotted.add(r["id_b"])

        ax.plot(path_a[:, 0], path_a[:, 1], "--", color="#065A82", alpha=0.6, label=label_a)
        ax.plot(path_b[:, 0], path_b[:, 1], "--", color="#065A82", alpha=0.6, label=label_b)
        ax.plot(*pos_a, "o", color="#065A82", markersize=6)
        ax.plot(*pos_b, "o", color="#065A82", markersize=6)

        if prob >= 0.3:
            closest_a = pos_a + vel_a * time_to_cpa
            circle = plt.Circle(closest_a, HORIZONTAL_MINIMA_KM, fill=True, alpha=0.18, color=color)
            ax.add_patch(circle)
            ax.annotate(f"Risk: {prob:.0%}\n({r['risk_level']})",
                        xy=closest_a, fontsize=8, ha="center",
                        color=color, fontweight="bold")

    ax.set_title("ADS-B Conflict Prediction (H3 + Uncertainty + Monte Carlo + LSTM)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Local X (East-West km)")
    ax.set_ylabel("Local Y (North-South km)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n[STEP 5] Visualization plot saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADS-B Conflict Prediction Pipeline")
    parser.add_argument("--file", type=str, default=DEFAULT_FILE, help="Path to scene .txt file")
    parser.add_argument("--sims", type=int, default=500, help="Number of Monte Carlo simulations per pair")
    args = parser.parse_args()

    target_file = args.file
    if not os.path.isabs(target_file):
        target_file = os.path.join(DATA_ROOT, target_file)

    run_pipeline(target_file, n_simulations=args.sims)
