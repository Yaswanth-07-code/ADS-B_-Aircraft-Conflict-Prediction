"""
demo_all.py
------------
ONE-CLICK PRESENTATION DEMO FOR ADS-B CONFLICT PREDICTION.

Runs the entire Phase 2 workflow in a single terminal in clear,
step-by-step order:
  Step 1: H3 Spatial Indexing (generates h3_grid_visualization.png)
  Step 2: Position Uncertainty Modeling (GPS/altimeter noise)
  Step 3: Monte Carlo Simulation (500 stochastic iterations)
  Step 4: Conflict Probability Engine (CPA + LSTM + Monte Carlo)
  Step 5: Full Real Dataset Demonstration (test/177.txt with 5 aircraft)

Run with:
    python demo_all.py
"""

import os
import sys
import math
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h3_spatial import (
    H3_RESOLUTION, get_h3_cell, km_to_latlon,
    group_aircraft_by_cell, get_candidate_pairs, get_cell_info
)
from visualize_h3 import plot_h3_grid
from uncertainty import add_uncertainty, generate_samples, HORIZONTAL_STD_KM, VERTICAL_STD_FT
from monte_carlo import monte_carlo_conflict
from conflict_probability import compute_conflict_probability, load_lstm
from run_full_pipeline import run_pipeline, DEFAULT_FILE


def banner(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "   ADS-B NEAR-MISS / CONFLICT PREDICTION: PHASE 2 MASTER DEMO    " + "*")
    print("*" + "   Architecture: H3 -> Uncertainty -> Monte Carlo -> LSTM AI     " + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    # -------------------------------------------------------------
    # STAGE 1: H3 HEXAGONAL SPATIAL INDEXING
    # -------------------------------------------------------------
    banner("STAGE 1: H3 HEXAGONAL SPATIAL INDEXING (AIRSPACE PARTITIONING)")
    print("EXPLANATION FOR REVIEW:")
    print("  'Instead of checking every pair of aircraft across the entire sky (O(N^2)),")
    print("   we divide the airspace into hexagonal cells (Uber H3, Resolution 6).")
    print("   Planes in the same or adjacent hexagons are checked; distant planes are pruned.'\n")

    positions_demo = {
        "AC_1": (0.5, 0.3),
        "AC_2": (0.8, 0.5),
        "AC_3": (15.0, 12.0),
        "AC_4": (0.2, -0.1),
        "AC_5": (14.5, 11.8),
    }

    cell_groups, _ = group_aircraft_by_cell(positions_demo)
    candidate_pairs = get_candidate_pairs(positions_demo)
    total_pairs = len(positions_demo) * (len(positions_demo) - 1) // 2

    print(f"Aircraft Grouping in Airspace (5 Aircraft across 2 distant clusters):")
    for cell_id, ac_ids in cell_groups.items():
        info = get_cell_info(cell_id)
        print(f"  - Hex Cell {cell_id[:12]}... (Area: {info['area_km2']} km2): Aircraft {ac_ids}")

    print(f"\nSpatial Pruning Results:")
    print(f"  - Total possible pairs without H3: {total_pairs}")
    print(f"  - Candidate pairs to check with H3: {len(candidate_pairs)} {candidate_pairs}")
    pruned = total_pairs - len(candidate_pairs)
    print(f"  - Distant pairs pruned:             {pruned} ({(pruned/total_pairs)*100:.0f}% compute savings)")

    print("\nGenerating H3 Visualization Images...")
    plot_h3_grid(file_path=DEFAULT_FILE, output_png="h3_grid_visualization.png")
    plot_h3_grid(file_path=None, output_png="h3_cluster_pruning.png")
    print("  [SAVED] h3_grid_visualization.png (Real flight scene 78.txt)")
    print("  [SAVED] h3_cluster_pruning.png (Pruning demonstration diagram)")

    # -------------------------------------------------------------
    # STAGE 2: POSITION UNCERTAINTY MODELING
    # -------------------------------------------------------------
    banner("STAGE 2: SENSOR UNCERTAINTY MODELING (GPS & ALTIMETER JITTER)")
    print("EXPLANATION FOR REVIEW:")
    print("  'Real ADS-B broadcasts are not 100% exact. Satellite signals and barometric")
    print("   sensors experience small errors. We model this uncertainty using aviation")
    print("   standards (DO-260B): 100m horizontal GPS noise and 100ft altitude noise.'\n")

    reported_pos = (1.500, 2.300)
    reported_vel = (150.0, -50.0)
    reported_alt = 2500.0

    print(f"Reported Aircraft State:")
    print(f"  Position: ({reported_pos[0]:.3f}, {reported_pos[1]:.3f}) km | Alt: {reported_alt:.0f} ft | Speed: 158 km/h\n")

    samples = generate_samples(reported_pos, reported_vel, reported_alt, n_samples=5, seed=42)
    print("Generated 5 Noisy Sensor Realizations (What the true position could be):")
    for i, (spos, svel, salt) in enumerate(samples, 1):
        pos_err_m = math.sqrt((spos[0] - reported_pos[0])**2 + (spos[1] - reported_pos[1])**2) * 1000
        alt_err_ft = salt - reported_alt
        print(f"  Sample {i}: pos=({spos[0]:.3f}, {spos[1]:.3f}) km  alt={salt:.0f} ft | "
              f"Noise drift: {pos_err_m:.0f}m horiz, {alt_err_ft:+.0f}ft vert")

    # -------------------------------------------------------------
    # STAGE 3: MONTE CARLO SIMULATION
    # -------------------------------------------------------------
    banner("STAGE 3: MONTE CARLO SIMULATION (500 ITERATIONS PER PAIR)")
    print("EXPLANATION FOR REVIEW:")
    print("  'Because sensor data is uncertain, checking conflict just ONCE is risky.")
    print("   Monte Carlo repeats the conflict check 500 times with random sensor noise.")
    print("   The fraction of runs that result in separation loss IS the conflict probability.'\n")

    print("Test Scenario 1: Converging Head-on (Same Altitude 2000 ft)")
    mc1 = monte_carlo_conflict((0, 0), (150, 0), 2000, (5, 0.2), (-150, 0), 2100, n_simulations=500)
    print(f"  -> Conflicts: {mc1['n_conflicts']}/500 | Probability: {mc1['conflict_probability']:.1%} | Risk: {mc1['risk_level']}")

    print("\nTest Scenario 2: Vertically Separated (2000 ft Vertical Gap)")
    mc2 = monte_carlo_conflict((0, 0), (150, 0), 2000, (5, 0.2), (-150, 0), 4000, n_simulations=500)
    print(f"  -> Conflicts: {mc2['n_conflicts']}/500 | Probability: {mc2['conflict_probability']:.1%} | Risk: {mc2['risk_level']}")

    print("\nTest Scenario 3: BORDERLINE CASE (Aircraft near 5nm / 1000ft boundary)")
    mc3 = monte_carlo_conflict((0, 0), (150, 0), 2000, (8.5, 0.5), (-150, 0), 2950, n_simulations=500)
    print(f"  -> Conflicts: {mc3['n_conflicts']}/500 | Probability: {mc3['conflict_probability']:.1%} | Risk: {mc3['risk_level']}")
    print("  * NOTICE: A single deterministic check only gives SAFE or CONFLICT.")
    print(f"    Monte Carlo reveals a {mc3['conflict_probability']:.1%} probability of conflict under real GPS drift!")

    # -------------------------------------------------------------
    # STAGE 4: COMBINED CONFLICT PROBABILITY (CPA + LSTM + MONTE CARLO)
    # -------------------------------------------------------------
    banner("STAGE 4: CONFLICT PROBABILITY ENGINE (FUSING CPA + LSTM AI + MONTE CARLO)")
    print("EXPLANATION FOR REVIEW:")
    print("  'We fuse three complementary approaches into a single score:")
    print("   1. CPA: Strict geometric separation check")
    print("   2. LSTM Neural Network: Sequence AI learning trajectory history (10 timesteps, 98.3% accuracy)")
    print("   3. Monte Carlo: Stochastic simulation under sensor noise'")
    print("   Formula: Final Risk = (0.4 * LSTM_Prob) + (0.6 * MonteCarlo_Prob)\n")

    lstm_m, lstm_s, lstm_c = load_lstm()
    if lstm_m:
        print("  [OK] Trained LSTM model, Scaler, and Config loaded from disk.")

    cp_res = compute_conflict_probability(
        pos_a=(0, 0), vel_a=(150, 0), alt_a_ft=2000,
        pos_b=(8.5, 0.5), vel_b=(-150, 0), alt_b_ft=2950,
        lstm_model=lstm_m, lstm_scaler=lstm_s, lstm_config=lstm_c, n_simulations=500
    )
    print(f"\nBorderline Pair Evaluation:")
    print(f"  - CPA Baseline:     {'CONFLICT' if cp_res['cpa_conflict'] else 'SAFE'}")
    print(f"  - LSTM Neural Net:  {cp_res['lstm_probability']:.1%}")
    print(f"  - Monte Carlo:      {cp_res['mc_probability']:.1%}")
    print(f"  -------------------------------------------")
    print(f"  - FINAL COMBINED:   {cp_res['final_probability']:.1%} -> [{cp_res['risk_level']} RISK]")

    # -------------------------------------------------------------
    # STAGE 5: REAL DATASET EXECUTION
    # -------------------------------------------------------------
    banner("STAGE 5: FULL PIPELINE EXECUTION ON REAL DATASET (test/177.txt)")
    print("Executing all 5 stages on 5 simultaneous real aircraft from TrajAir...\n")
    results = run_pipeline(DEFAULT_FILE, n_simulations=500)

    # -------------------------------------------------------------
    # SUMMARY OF GENERATED FILES
    # -------------------------------------------------------------
    banner("ALL OUTPUT ARTIFACTS READY FOR PRESENTATION")
    print("All image files have been generated and saved to your project folder:")
    print("  1. h3_grid_visualization.png    -> Real flight scene inside H3 hex grid")
    print("  2. h3_cluster_pruning.png       -> Demonstrates candidate pairs vs pruned pairs")
    print("  3. pipeline_conflict_result.png -> Trajectory paths, separation zones, risk levels")
    print("  4. case1_conflict.png           -> CPA baseline direct conflict example")
    print("  5. case2_safe_altitude.png      -> CPA baseline vertical clearance example")
    print("\nWeb Dashboard:")
    print("  Run 'python dashboard.py' to explore scenarios interactively in browser at http://127.0.0.1:5000")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
