"""
uncertainty.py
---------------
Position Uncertainty Modeling for ADS-B Conflict Prediction.

WHY: GPS readings aren't perfect. An aircraft's reported position
could be off by 50-200 meters. If two aircraft are near a conflict
boundary, a small GPS error could mean the difference between
"safe" and "dangerous". We model this uncertainty.

WHAT IT DOES:
  Takes an aircraft's reported position and adds random noise
  based on known GPS accuracy specs for ADS-B:
    - Horizontal: 50-200m standard deviation
    - Vertical: 15-50m (45-150 ft) standard deviation
    - Velocity: 1-5 m/s standard deviation

  Each call to add_uncertainty() returns a DIFFERENT noisy version
  of the position. Run it many times = Monte Carlo simulation.

HOW IT CONNECTS:
  H3 (find nearby pairs) -> Uncertainty (add noise) -> Monte Carlo
"""

import numpy as np


# --- ADS-B Accuracy Specifications ---
# These come from aviation standards (DO-260B)

# Horizontal position accuracy (in km, since our dataset uses km)
HORIZONTAL_STD_KM = 0.1  # 100 meters = 0.1 km (typical ADS-B NACp >= 8)

# Vertical position accuracy (in feet, since we use feet for altitude)
VERTICAL_STD_FT = 100.0  # 100 feet (typical ADS-B NACp vertical)

# Velocity accuracy (in km/h, since our velocities are in km/h)
VELOCITY_STD_KMH = 5.0  # ~1.4 m/s = ~5 km/h


def add_uncertainty(pos, vel, alt_ft,
                    pos_std=HORIZONTAL_STD_KM,
                    vel_std=VELOCITY_STD_KMH,
                    alt_std=VERTICAL_STD_FT,
                    rng=None):
    """
    Adds Gaussian noise to an aircraft's state to simulate GPS uncertainty.

    Args:
        pos: (x_km, y_km) tuple - reported position
        vel: (vx_kmh, vy_kmh) tuple - reported velocity
        alt_ft: float - reported altitude in feet
        pos_std: float - position noise std dev in km (default 0.1 km = 100m)
        vel_std: float - velocity noise std dev in km/h
        alt_std: float - altitude noise std dev in feet
        rng: numpy random generator (for reproducibility)

    Returns:
        (noisy_pos, noisy_vel, noisy_alt_ft) - same format but with noise added
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy_pos = (
        pos[0] + rng.normal(0, pos_std),
        pos[1] + rng.normal(0, pos_std),
    )

    noisy_vel = (
        vel[0] + rng.normal(0, vel_std),
        vel[1] + rng.normal(0, vel_std),
    )

    noisy_alt = alt_ft + rng.normal(0, alt_std)

    return noisy_pos, noisy_vel, noisy_alt


def generate_samples(pos, vel, alt_ft, n_samples=100, seed=42,
                     pos_std=HORIZONTAL_STD_KM,
                     vel_std=VELOCITY_STD_KMH,
                     alt_std=VERTICAL_STD_FT):
    """
    Generates N noisy versions of an aircraft's state.
    Used by Monte Carlo to create many scenarios.

    Returns:
        List of (noisy_pos, noisy_vel, noisy_alt_ft) tuples
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        sample = add_uncertainty(pos, vel, alt_ft, pos_std, vel_std, alt_std, rng)
        samples.append(sample)
    return samples


# ---- Self-test ----
if __name__ == "__main__":
    print("=" * 60)
    print("Uncertainty Model - Self Test")
    print("=" * 60)

    # Original aircraft state
    pos = (1.5, 2.3)
    vel = (150.0, -50.0)
    alt = 3000.0

    print(f"\nOriginal state:")
    print(f"  Position: ({pos[0]:.3f}, {pos[1]:.3f}) km")
    print(f"  Velocity: ({vel[0]:.1f}, {vel[1]:.1f}) km/h")
    print(f"  Altitude: {alt:.0f} ft")

    print(f"\n5 noisy samples (pos_std={HORIZONTAL_STD_KM}km, "
          f"vel_std={VELOCITY_STD_KMH}km/h, alt_std={VERTICAL_STD_FT}ft):")

    samples = generate_samples(pos, vel, alt, n_samples=5, seed=42)
    for i, (np_, nv, na) in enumerate(samples):
        pos_err = ((np_[0]-pos[0])**2 + (np_[1]-pos[1])**2)**0.5 * 1000  # meters
        print(f"  Sample {i+1}: pos=({np_[0]:.3f}, {np_[1]:.3f}) "
              f"vel=({nv[0]:.1f}, {nv[1]:.1f}) "
              f"alt={na:.0f}ft  "
              f"[pos error: {pos_err:.0f}m, alt error: {na-alt:+.0f}ft]")

    # Show distribution stats over 1000 samples
    print(f"\nDistribution over 1000 samples:")
    big_samples = generate_samples(pos, vel, alt, n_samples=1000, seed=123)
    pos_errors = [((s[0][0]-pos[0])**2 + (s[0][1]-pos[1])**2)**0.5 * 1000 for s in big_samples]
    alt_errors = [abs(s[2] - alt) for s in big_samples]
    print(f"  Position error:  mean={np.mean(pos_errors):.0f}m, "
          f"max={np.max(pos_errors):.0f}m, 95th={np.percentile(pos_errors,95):.0f}m")
    print(f"  Altitude error:  mean={np.mean(alt_errors):.0f}ft, "
          f"max={np.max(alt_errors):.0f}ft, 95th={np.percentile(alt_errors,95):.0f}ft")
