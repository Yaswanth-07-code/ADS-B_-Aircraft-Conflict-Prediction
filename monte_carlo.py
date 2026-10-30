"""
monte_carlo.py
---------------
Monte Carlo Simulation for Conflict Probability.

WHY: A single CPA check tells us "conflict" or "safe" based on
exact reported positions. But GPS isn't perfect! Monte Carlo runs
the same conflict check HUNDREDS of times, each time with slightly
different positions (simulating GPS noise). The percentage of runs
that result in conflict IS the conflict probability.

EXAMPLE:
  Run 500 simulations of the same aircraft pair.
  320 out of 500 say CONFLICT.
  Conflict probability = 320/500 = 64%

HOW IT CONNECTS:
  Uncertainty (add noise) -> Monte Carlo (run N simulations) -> Probability
"""

import numpy as np
from cpa import check_conflict
from uncertainty import add_uncertainty


def monte_carlo_conflict(pos_a, vel_a, alt_a_ft,
                         pos_b, vel_b, alt_b_ft,
                         n_simulations=500,
                         seed=42):
    """
    Runs N conflict checks with random GPS noise added each time.

    For each simulation:
      1. Add random noise to both aircraft's positions, velocities, and altitudes
      2. Run the CPA conflict check
      3. Record whether it's a conflict

    Returns a dict with:
      - conflict_probability: float 0.0 to 1.0
      - n_conflicts: how many simulations resulted in conflict
      - n_simulations: total simulations run
      - min_distances: list of minimum distances from each simulation
      - risk_level: "HIGH", "MEDIUM", or "LOW" based on probability
    """
    rng = np.random.default_rng(seed)

    n_conflicts = 0
    min_distances = []
    vertical_gaps = []

    for _ in range(n_simulations):
        # Add GPS noise to both aircraft
        noisy_pos_a, noisy_vel_a, noisy_alt_a = add_uncertainty(
            pos_a, vel_a, alt_a_ft, rng=rng
        )
        noisy_pos_b, noisy_vel_b, noisy_alt_b = add_uncertainty(
            pos_b, vel_b, alt_b_ft, rng=rng
        )

        # Run conflict check with noisy data
        result = check_conflict(
            noisy_pos_a, noisy_vel_a, noisy_alt_a,
            noisy_pos_b, noisy_vel_b, noisy_alt_b
        )

        if result["conflict"]:
            n_conflicts += 1

        min_distances.append(result["minimum_horizontal_distance_nm"])
        vertical_gaps.append(result["vertical_gap_ft"])

    probability = n_conflicts / n_simulations

    # Risk classification
    if probability >= 0.7:
        risk_level = "HIGH"
    elif probability >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "conflict_probability": round(probability, 4),
        "n_conflicts": n_conflicts,
        "n_simulations": n_simulations,
        "risk_level": risk_level,
        "mean_min_distance_nm": round(float(np.mean(min_distances)), 2),
        "std_min_distance_nm": round(float(np.std(min_distances)), 2),
        "mean_vertical_gap_ft": round(float(np.mean(vertical_gaps)), 1),
        "p95_min_distance_nm": round(float(np.percentile(min_distances, 5)), 2),
    }


def batch_monte_carlo(pairs_data, n_simulations=500, seed=42):
    """
    Runs Monte Carlo on multiple aircraft pairs.

    Input:
      pairs_data: list of dicts, each with:
        {id_a, id_b, pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft}

    Returns:
      list of dicts with Monte Carlo results added to each pair
    """
    results = []
    for i, pair in enumerate(pairs_data):
        mc_result = monte_carlo_conflict(
            pair["pos_a"], pair["vel_a"], pair["alt_a_ft"],
            pair["pos_b"], pair["vel_b"], pair["alt_b_ft"],
            n_simulations=n_simulations,
            seed=seed + i  # different seed per pair for independence
        )
        results.append({
            **pair,
            "monte_carlo": mc_result,
        })
    return results


# ---- Self-test ----
if __name__ == "__main__":
    print("=" * 60)
    print("Monte Carlo Conflict Simulation - Self Test")
    print("=" * 60)

    # Case 1: Clear conflict (converging, same altitude)
    print("\nCase 1: Head-on collision course, same altitude (2000 ft)")
    result1 = monte_carlo_conflict(
        pos_a=(0, 0), vel_a=(150, 0), alt_a_ft=2000,
        pos_b=(5, 0.2), vel_b=(-150, 0), alt_b_ft=2100,
        n_simulations=500
    )
    print(f"  Probability: {result1['conflict_probability']:.1%}")
    print(f"  Risk level:  {result1['risk_level']}")
    print(f"  Conflicts:   {result1['n_conflicts']}/{result1['n_simulations']}")
    print(f"  Mean min dist: {result1['mean_min_distance_nm']} nm")

    # Case 2: Safely separated by altitude (2000 ft gap)
    print("\nCase 2: Same path but 2000 ft altitude separation")
    result2 = monte_carlo_conflict(
        pos_a=(0, 0), vel_a=(150, 0), alt_a_ft=2000,
        pos_b=(5, 0.2), vel_b=(-150, 0), alt_b_ft=4000,
        n_simulations=500
    )
    print(f"  Probability: {result2['conflict_probability']:.1%}")
    print(f"  Risk level:  {result2['risk_level']}")
    print(f"  Conflicts:   {result2['n_conflicts']}/{result2['n_simulations']}")
    print(f"  Mean vert gap: {result2['mean_vertical_gap_ft']} ft")

    # Case 3: Borderline case (just at separation minimum)
    print("\nCase 3: Borderline - aircraft pass near the 5nm/1000ft boundary")
    result3 = monte_carlo_conflict(
        pos_a=(0, 0), vel_a=(150, 0), alt_a_ft=2000,
        pos_b=(8.5, 0.5), vel_b=(-150, 0), alt_b_ft=2950,
        n_simulations=500
    )
    print(f"  Probability: {result3['conflict_probability']:.1%}")
    print(f"  Risk level:  {result3['risk_level']}")
    print(f"  Conflicts:   {result3['n_conflicts']}/{result3['n_simulations']}")
    print(f"  This is where Monte Carlo adds value - the deterministic check")
    print(f"  says just 'conflict' or 'safe', but Monte Carlo tells us HOW LIKELY.")
