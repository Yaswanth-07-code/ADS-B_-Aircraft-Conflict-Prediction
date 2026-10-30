"""
visualize.py
------------
Step 5 of the build order: a simple plot of two aircraft's projected
paths, marking the closest point of approach and whether it's a
conflict. This is what you'd screenshot for the mid-review slides.

Run this with:  python3 visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
from cpa import check_conflict, closest_point_of_approach, HORIZONTAL_MINIMA_KM


def plot_pair(pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft, title, filename):
    time_to_closest_approach, minimum_distance_km = closest_point_of_approach(
        pos_a, vel_a, pos_b, vel_b
    )
    result = check_conflict(pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft)

    pos_a = np.array(pos_a, dtype=float)
    pos_b = np.array(pos_b, dtype=float)
    vel_a = np.array(vel_a, dtype=float)
    vel_b = np.array(vel_b, dtype=float)

    # Draw each aircraft's path from t=0 out to a bit past the closest-approach time
    t_end = max(time_to_closest_approach * 1.6, 0.05)
    t_vals = np.linspace(0, t_end, 50)

    path_a = np.array([pos_a + vel_a * t for t in t_vals])
    path_b = np.array([pos_b + vel_b * t for t in t_vals])

    closest_point_a = pos_a + vel_a * time_to_closest_approach
    closest_point_b = pos_b + vel_b * time_to_closest_approach

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(path_a[:, 0], path_a[:, 1], "--", color="#065A82", label="Aircraft A path")
    ax.plot(path_b[:, 0], path_b[:, 1], "--", color="#D85A30", label="Aircraft B path")

    ax.plot(*pos_a, "o", color="#065A82", markersize=10)
    ax.plot(*pos_b, "o", color="#D85A30", markersize=10)

    ax.plot(*closest_point_a, "x", color="#065A82", markersize=10, mew=2)
    ax.plot(*closest_point_b, "x", color="#D85A30", markersize=10, mew=2)

    # Separation minima circle around aircraft A's closest-approach position
    circle_color = "#E24B4A" if result["conflict"] else "#3FA66B"
    circle = plt.Circle(closest_point_a, HORIZONTAL_MINIMA_KM, fill=True,
                         alpha=0.15, color=circle_color)
    ax.add_patch(circle)
    circle_edge = plt.Circle(closest_point_a, HORIZONTAL_MINIMA_KM, fill=False,
                              linestyle="--", color=circle_color)
    ax.add_patch(circle_edge)

    status = "CONFLICT" if result["conflict"] else "SAFE"
    subtitle = (f"time to closest approach = {result['time_to_closest_approach_minutes']} min | "
                f"min horizontal sep = {result['minimum_horizontal_distance_nm']} nm | "
                f"vertical gap = {result['vertical_gap_ft']} ft | {status}")

    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlabel("km (east-west)")
    ax.set_ylabel("km (north-south)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}  ({status})")


if __name__ == "__main__":
    plot_pair(
        pos_a=(0, 0), vel_a=(300, 0), alt_a_ft=35000,
        pos_b=(60, 8), vel_b=(-300, 0), alt_b_ft=35000,
        title="Case 1: Converging aircraft, same altitude",
        filename="case1_conflict.png",
    )

    plot_pair(
        pos_a=(0, 0), vel_a=(300, 0), alt_a_ft=35000,
        pos_b=(60, 8), vel_b=(-300, 0), alt_b_ft=37000,
        title="Case 2: Converging aircraft, altitude-separated",
        filename="case2_safe_altitude.png",
    )