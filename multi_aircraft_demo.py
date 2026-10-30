"""
multi_aircraft_demo.py
------------------------
The full multi-aircraft conflict detection pipeline:

    Real Data -> Process Data -> Track Multiple Aircraft
              -> Straight-Line Prediction (CPA)
              -> LSTM Neural Network Prediction (Sequence AI)
              -> Visualize Results

Loads several real aircraft from one TrajAir file (up to MAX_AIRCRAFT
of them), checks EVERY pair that shares a real moment in time using
BOTH the geometric CPA baseline AND the trained LSTM neural network,
prints the results side by side, and plots everyone at once.

Before running this, train the model first:
    python train_conflict_predictor.py

Then run:
    python multi_aircraft_demo.py
"""

import itertools
import os
import matplotlib.pyplot as plt
import numpy as np

from cpa import check_conflict, closest_point_of_approach, HORIZONTAL_MINIMA_KM
from data_utils import load_scene_file, get_pos_vel_alt
from conflict_probability import load_lstm, lstm_predict_single_frame

FILE_PATH = r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data\test\177.txt"
MAX_AIRCRAFT = 5  # this many aircraft make up ONE case/scenario


def lstm_predict(model, scaler, config, pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft):
    """
    Runs the LSTM on one aircraft pair.
    Returns (predicted_conflict, confidence_percent).
    """
    conflict_prob = lstm_predict_single_frame(
        model, scaler, config,
        pos_a, vel_a, alt_a_ft,
        pos_b, vel_b, alt_b_ft
    )
    predicted_conflict = conflict_prob >= 0.5
    return predicted_conflict, conflict_prob


def find_shared_frame(df, id_a, id_b):
    """Finds a moment in time where both aircraft have a reading."""
    frames_a = set(df[df["ID"] == id_a]["Frame"])
    frames_b = set(df[df["ID"] == id_b]["Frame"])
    shared = frames_a & frames_b
    return min(shared) if shared else None


def run_pipeline(file_path, lstm_model=None, lstm_scaler=None, lstm_config=None):
    print("Step 1: Real Data -> loading file")
    df = load_scene_file(file_path)

    all_ids = df["ID"].unique()
    aircraft_ids = all_ids[:MAX_AIRCRAFT]
    print(f"Step 2: Process Data -> {len(all_ids)} aircraft found, using {len(aircraft_ids)}: {list(aircraft_ids)}")

    print("Step 3: Track Multiple Aircraft -> checking every possible pair")
    results = []
    for id_a, id_b in itertools.combinations(aircraft_ids, 2):
        shared_frame = find_shared_frame(df, id_a, id_b)
        if shared_frame is None:
            continue

        data_a = df[df["ID"] == id_a].sort_values("Frame").reset_index(drop=True)
        data_b = df[df["ID"] == id_b].sort_values("Frame").reset_index(drop=True)

        info_a = get_pos_vel_alt(data_a, shared_frame)
        info_b = get_pos_vel_alt(data_b, shared_frame)
        if info_a is None or info_b is None:
            continue

        pos_a, vel_a, alt_a_ft = info_a
        pos_b, vel_b, alt_b_ft = info_b

        # --- CPA Baseline ---
        result = check_conflict(pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft)

        print(f"\n  Pair: {id_a} vs {id_b}  (Frame {shared_frame})")
        print(f"    [CPA Baseline]   Conflict: {result['conflict']}  |  "
              f"Min dist: {result['minimum_horizontal_distance_nm']} nm  |  "
              f"Vert gap: {result['vertical_gap_ft']} ft  |  "
              f"Time to CPA: {result['time_to_closest_approach_minutes']} min")

        # --- LSTM Neural Network ---
        lstm_conflict = None
        lstm_confidence = None
        if lstm_model is not None and lstm_scaler is not None and lstm_config is not None:
            lstm_conflict, lstm_confidence = lstm_predict(
                lstm_model, lstm_scaler, lstm_config,
                pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft
            )
            label = "CONFLICT" if lstm_conflict else "SAFE"
            print(f"    [LSTM Neural Net] Conflict: {lstm_conflict}  |  "
                  f"Confidence: {lstm_confidence:.0%}  |  Prediction: {label}")

            if lstm_conflict != result["conflict"]:
                print(f"    ** DISAGREEMENT: CPA says {'CONFLICT' if result['conflict'] else 'SAFE'}, "
                      f"LSTM says {label} **")

        results.append({
            "id_a": id_a, "id_b": id_b, "frame": shared_frame,
            "pos_a": pos_a, "vel_a": vel_a, "alt_a_ft": alt_a_ft,
            "pos_b": pos_b, "vel_b": vel_b, "alt_b_ft": alt_b_ft,
            "lstm_conflict": lstm_conflict,
            "lstm_confidence": lstm_confidence,
            **result,
        })

    return results


def visualize_all(results, filename="multi_aircraft_result.png"):
    print("\nStep 5: Visualize Results")
    if not results:
        print("  Nothing to plot - no overlapping pairs were found.")
        return

    has_lstm = any(r["lstm_conflict"] is not None for r in results)

    fig, ax = plt.subplots(figsize=(10, 8))
    plotted_ids = set()

    for r in results:
        time_to_cpa, _ = closest_point_of_approach(r["pos_a"], r["vel_a"], r["pos_b"], r["vel_b"])
        t_vals = np.linspace(0, max(time_to_cpa * 1.5, 0.02), 30)

        pos_a = np.array(r["pos_a"]); vel_a = np.array(r["vel_a"])
        pos_b = np.array(r["pos_b"]); vel_b = np.array(r["vel_b"])
        path_a = np.array([pos_a + vel_a * t for t in t_vals])
        path_b = np.array([pos_b + vel_b * t for t in t_vals])

        if r["lstm_confidence"] is not None:
            risk = r["lstm_confidence"]
            if risk > 0.7:
                color = "#E24B4A"  # red
            elif risk > 0.3:
                color = "#E8A838"  # orange/yellow
            else:
                color = "#3FA66B"  # green
        else:
            color = "#E24B4A" if r["conflict"] else "#3FA66B"

        label_a = f"Aircraft {r['id_a']}" if r["id_a"] not in plotted_ids else None
        label_b = f"Aircraft {r['id_b']}" if r["id_b"] not in plotted_ids else None
        plotted_ids.add(r["id_a"]); plotted_ids.add(r["id_b"])

        ax.plot(path_a[:, 0], path_a[:, 1], "--", color="#065A82", alpha=0.7, label=label_a)
        ax.plot(path_b[:, 0], path_b[:, 1], "--", color="#065A82", alpha=0.7, label=label_b)
        ax.plot(*pos_a, "o", color="#065A82", markersize=6)
        ax.plot(*pos_b, "o", color="#065A82", markersize=6)

        if r["conflict"] or (r["lstm_conflict"] is True):
            closest_a = pos_a + vel_a * time_to_cpa
            circle = plt.Circle(closest_a, HORIZONTAL_MINIMA_KM, fill=True, alpha=0.15, color=color)
            ax.add_patch(circle)

            if r["lstm_confidence"] is not None:
                ax.annotate(f"LSTM: {r['lstm_confidence']:.0%}",
                            xy=closest_a, fontsize=7, ha="center",
                            color=color, fontweight="bold")

    title = "Multi-Aircraft Conflict Check"
    if has_lstm:
        title += " (CPA Baseline + LSTM Neural Network)"
    ax.set_title(title)
    ax.set_xlabel("km (east-west)")
    ax.set_ylabel("km (north-south)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


if __name__ == "__main__":
    print("=" * 60)
    print("Loading LSTM Neural Network model...")
    print("=" * 60)
    lstm_model, lstm_scaler, lstm_config = load_lstm()

    results = run_pipeline(FILE_PATH, lstm_model, lstm_scaler, lstm_config)

    print("\n" + "=" * 60)
    print(f"CASE SUMMARY - {len(results)} pairs checked among the same {MAX_AIRCRAFT}-aircraft scenario")
    print("=" * 60)

    cpa_conflicts = [r for r in results if r["conflict"]]
    print(f"\nPairs checked: {len(results)}")
    print(f"CPA Baseline conflicts: {len(cpa_conflicts)}")

    if lstm_model is not None:
        lstm_conflicts = [r for r in results if r["lstm_conflict"]]
        print(f"LSTM Neural Net conflicts: {len(lstm_conflicts)}")

        agree = sum(1 for r in results if r["conflict"] == r["lstm_conflict"])
        print(f"Agreement: {agree}/{len(results)} pairs ({agree/len(results):.0%})")

    print("\n" + "-" * 60)
    print("DETAILED RESULTS")
    print("-" * 60)
    for r in results:
        cpa_label = "CONFLICT" if r["conflict"] else "SAFE"
        line = (f"  {r['id_a']} vs {r['id_b']}: "
                f"CPA={cpa_label} ({r['minimum_horizontal_distance_nm']} nm, "
                f"{r['vertical_gap_ft']} ft)")
        if r["lstm_confidence"] is not None:
            lstm_label = "CONFLICT" if r["lstm_conflict"] else "SAFE"
            line += f"  |  LSTM={lstm_label} ({r['lstm_confidence']:.0%})"
        print(line)

    visualize_all(results)
