"""
dashboard.py
--------------
Flask Web Dashboard for ADS-B Conflict Prediction.

This is the FINAL piece of the pipeline:
  Dataset -> H3 -> Uncertainty -> Monte Carlo -> Conflict Probability (LSTM) -> DASHBOARD

Run with:
    python dashboard.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import glob
import itertools
from flask import Flask, render_template, request

from data_utils import load_scene_file, get_pos_vel_alt
from h3_spatial import group_aircraft_by_cell, get_candidate_pairs, H3_RESOLUTION
from conflict_probability import compute_conflict_probability, load_lstm

app = Flask(__name__)

# --- Configuration ---
DATA_FOLDER = r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data"
MAX_AIRCRAFT = 10
N_SIMULATIONS = 500


def get_available_files():
    """Lists all .txt files with 2+ aircraft from test/ and train/ folders."""
    files = []
    for subfolder in ["test", "train"]:
        folder = os.path.join(DATA_FOLDER, subfolder)
        if os.path.exists(folder):
            for f in sorted(glob.glob(os.path.join(folder, "*.txt"))):
                basename = os.path.basename(f)
                try:
                    df = load_scene_file(f)
                    n_ids = len(df["ID"].unique())
                    if n_ids >= 2:
                        files.append(f"{subfolder}/{basename}")
                except Exception:
                    pass
    return files


def analyze_file(file_rel_path):
    """
    Runs the full pipeline on one scene file:
      1. Load data
      2. H3 spatial indexing
      3. Find candidate pairs
      4. For each pair: CPA + LSTM + Monte Carlo -> conflict probability
    """
    file_path = os.path.join(DATA_FOLDER, file_rel_path)
    df = load_scene_file(file_path)

    all_ids = df["ID"].unique()
    aircraft_ids = list(all_ids[:MAX_AIRCRAFT])

    # Representative position per aircraft for H3 grouping
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

    # H3 spatial indexing
    h3_cells, aircraft_to_cell = group_aircraft_by_cell(aircraft_positions)
    candidate_pairs = get_candidate_pairs(aircraft_positions)
    n_all_pairs = len(aircraft_ids) * (len(aircraft_ids) - 1) // 2

    # Load LSTM model
    lstm_model, lstm_scaler, lstm_config = load_lstm()

    # Analyze each candidate pair
    results = []
    for id_a, id_b in candidate_pairs:
        if id_a not in aircraft_data or id_b not in aircraft_data:
            continue

        frames_a = set(df[df["ID"] == id_a]["Frame"])
        frames_b = set(df[df["ID"] == id_b]["Frame"])
        shared = frames_a & frames_b
        if not shared:
            continue

        frame = min(shared)
        info_a = get_pos_vel_alt(aircraft_data[id_a], frame)
        info_b = get_pos_vel_alt(aircraft_data[id_b], frame)
        if info_a is None or info_b is None:
            continue

        pos_a, vel_a, alt_a_ft = info_a
        pos_b, vel_b, alt_b_ft = info_b

        prob_result = compute_conflict_probability(
            pos_a, vel_a, alt_a_ft,
            pos_b, vel_b, alt_b_ft,
            lstm_model=lstm_model,
            lstm_scaler=lstm_scaler,
            lstm_config=lstm_config,
            n_simulations=N_SIMULATIONS
        )

        results.append({
            "id_a": int(id_a),
            "id_b": int(id_b),
            "frame": int(frame),
            **prob_result,
        })

    # Sort by probability (highest risk first)
    results.sort(key=lambda x: x["final_probability"], reverse=True)

    n_high = sum(1 for r in results if r["risk_level"] == "HIGH")
    n_medium = sum(1 for r in results if r["risk_level"] == "MEDIUM")
    n_low = sum(1 for r in results if r["risk_level"] == "LOW")

    return {
        "results": results,
        "n_aircraft": len(aircraft_ids),
        "n_pairs": len(results),
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
        "n_h3_cells": len(h3_cells),
        "h3_cells": {k: [int(v) for v in vals] for k, vals in h3_cells.items()},
        "h3_resolution": H3_RESOLUTION,
        "n_candidate_pairs": len(candidate_pairs),
        "n_all_pairs": n_all_pairs,
    }


@app.route("/")
def dashboard():
    files = get_available_files()
    current_file = request.args.get("file", files[0] if files else None)

    if current_file:
        data = analyze_file(current_file)
    else:
        data = {
            "results": [], "n_aircraft": 0, "n_pairs": 0,
            "n_high": 0, "n_medium": 0, "n_low": 0,
            "n_h3_cells": 0, "h3_cells": {},
            "h3_resolution": H3_RESOLUTION,
            "n_candidate_pairs": 0, "n_all_pairs": 0,
        }

    return render_template("dashboard.html",
                           files=files,
                           current_file=current_file,
                           **data)


if __name__ == "__main__":
    print("=" * 60)
    print("ADS-B Conflict Prediction Dashboard (LSTM Engine)")
    print("=" * 60)
    print("Starting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=5000)
