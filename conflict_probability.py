"""
conflict_probability.py
------------------------
Combines ALL methods into a single conflict probability score.

This is the BRAIN of Phase 2. It takes an aircraft pair and produces
one final probability by combining:
  1. CPA deterministic check (baseline)
  2. LSTM neural network prediction (learned trajectory patterns over time)
  3. Monte Carlo simulation (accounts for GPS uncertainty)

The final probability is a weighted combination:
  P = 0.4 * LSTM_probability + 0.6 * MonteCarlo_probability

WHY THIS WEIGHTING:
  Monte Carlo gets more weight because it accounts for real-world
  uncertainty. The LSTM learned from trajectory data, but Monte Carlo
  physically simulates what happens with noisy positions.

HOW IT CONNECTS:
  H3 (find pairs) -> Uncertainty -> Monte Carlo -> THIS MODULE -> Dashboard
"""

import os
import sys
import math
import joblib
import numpy as np
import torch

from cpa import check_conflict
from monte_carlo import monte_carlo_conflict
from train_conflict_predictor import compute_features, ConflictLSTM, N_FEATURES


def load_lstm():
    """
    Loads the trained LSTM model, scaler, and config.
    Returns (model, scaler, config) or (None, None, None).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "lstm_model.pth")
    scaler_path = os.path.join(script_dir, "lstm_scaler.pkl")
    config_path = os.path.join(script_dir, "lstm_config.pkl")

    if (os.path.exists(model_path) and os.path.exists(scaler_path)
            and os.path.exists(config_path)):
        config = joblib.load(config_path)
        scaler = joblib.load(scaler_path)

        model = ConflictLSTM(
            input_size=config["n_features"],
            hidden_size=config["hidden_size"],
            fc_size=config["fc_size"],
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model, scaler, config
    return None, None, None


def lstm_predict_single_frame(model, scaler, config,
                               pos_a, vel_a, alt_a_ft,
                               pos_b, vel_b, alt_b_ft):
    """
    Runs the LSTM on a single snapshot by repeating the features across
    the sequence length. This gives a reasonable prediction even when
    only one frame is available (e.g., in the dashboard or Monte Carlo).

    For a proper multi-frame prediction, use lstm_predict_sequence().
    """
    features = compute_features(pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft)
    seq_len = config["seq_len"]

    # Repeat the same features across all timesteps
    # (approximation - assumes current state is representative)
    sequence = np.array([features] * seq_len)  # shape: (seq_len, n_features)

    # Scale features
    sequence_scaled = scaler.transform(sequence)  # shape: (seq_len, n_features)

    # Convert to tensor: (1, seq_len, n_features)
    x = torch.FloatTensor(sequence_scaled).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(x)  # shape: (1, 2)
        conflict_prob = float(probs[0, 1])

    return conflict_prob


def lstm_predict_sequence(model, scaler, config, feature_sequence):
    """
    Runs the LSTM on a proper multi-frame sequence.

    Args:
        feature_sequence: list of [14-feature vectors], one per timestep
                          length should be config["seq_len"]
    Returns:
        conflict_prob: float 0.0 to 1.0
    """
    seq_len = config["seq_len"]
    sequence = np.array(feature_sequence)

    # Pad or truncate to exact seq_len
    if len(sequence) < seq_len:
        # Pad by repeating the first frame
        pad_count = seq_len - len(sequence)
        padding = np.tile(sequence[0], (pad_count, 1))
        sequence = np.vstack([padding, sequence])
    elif len(sequence) > seq_len:
        sequence = sequence[-seq_len:]  # take the last seq_len frames

    # Scale
    sequence_scaled = scaler.transform(sequence)
    x = torch.FloatTensor(sequence_scaled).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(x)
        conflict_prob = float(probs[0, 1])

    return conflict_prob


def compute_conflict_probability(pos_a, vel_a, alt_a_ft,
                                  pos_b, vel_b, alt_b_ft,
                                  lstm_model=None, lstm_scaler=None, lstm_config=None,
                                  n_simulations=500,
                                  lstm_weight=0.4, mc_weight=0.6):
    """
    Computes a single conflict probability score combining all methods.

    Returns a dict with:
      - final_probability: weighted combination (0.0 to 1.0)
      - risk_level: "HIGH" / "MEDIUM" / "LOW"
      - cpa_result: deterministic CPA check result
      - lstm_probability: LSTM neural network prediction (if model loaded)
      - mc_probability: Monte Carlo simulation probability
      - mc_details: full Monte Carlo results
    """
    # 1. CPA deterministic baseline
    cpa_result = check_conflict(pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft)

    # 2. LSTM neural network prediction
    lstm_probability = None
    if lstm_model is not None and lstm_scaler is not None and lstm_config is not None:
        lstm_probability = lstm_predict_single_frame(
            lstm_model, lstm_scaler, lstm_config,
            pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft
        )

    # 3. Monte Carlo simulation
    mc_result = monte_carlo_conflict(
        pos_a, vel_a, alt_a_ft,
        pos_b, vel_b, alt_b_ft,
        n_simulations=n_simulations
    )
    mc_probability = mc_result["conflict_probability"]

    # 4. Combine into final probability
    if lstm_probability is not None:
        final_probability = (lstm_weight * lstm_probability) + (mc_weight * mc_probability)
    else:
        final_probability = mc_probability

    # Risk classification
    if final_probability >= 0.7:
        risk_level = "HIGH"
    elif final_probability >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "final_probability": round(final_probability, 4),
        "risk_level": risk_level,
        "cpa_conflict": cpa_result["conflict"],
        "cpa_min_distance_nm": cpa_result["minimum_horizontal_distance_nm"],
        "cpa_vertical_gap_ft": cpa_result["vertical_gap_ft"],
        "lstm_probability": round(lstm_probability, 4) if lstm_probability is not None else None,
        "mc_probability": mc_probability,
        "mc_details": mc_result,
    }


# ---- Self-test ----
if __name__ == "__main__":
    print("=" * 60)
    print("Conflict Probability Engine - Self Test")
    print("=" * 60)

    lstm_model, lstm_scaler, lstm_config = load_lstm()
    if lstm_model:
        print("  LSTM model loaded successfully")
    else:
        print("  WARNING: LSTM model not found, using Monte Carlo only")

    # Case 1: Clear conflict
    print("\n--- Case 1: Head-on, same altitude (should be HIGH) ---")
    r1 = compute_conflict_probability(
        pos_a=(0, 0), vel_a=(150, 0), alt_a_ft=2000,
        pos_b=(5, 0.2), vel_b=(-150, 0), alt_b_ft=2100,
        lstm_model=lstm_model, lstm_scaler=lstm_scaler, lstm_config=lstm_config
    )
    print(f"  Final probability: {r1['final_probability']:.1%}")
    print(f"  Risk level:        {r1['risk_level']}")
    print(f"  CPA says:          {'CONFLICT' if r1['cpa_conflict'] else 'SAFE'}")
    if r1["lstm_probability"] is not None:
        print(f"  LSTM says:         {r1['lstm_probability']:.1%}")
    print(f"  Monte Carlo says:  {r1['mc_probability']:.1%}")

    # Case 2: Safe separation
    print("\n--- Case 2: Altitude separated (should be LOW) ---")
    r2 = compute_conflict_probability(
        pos_a=(0, 0), vel_a=(150, 0), alt_a_ft=2000,
        pos_b=(5, 0.2), vel_b=(-150, 0), alt_b_ft=4000,
        lstm_model=lstm_model, lstm_scaler=lstm_scaler, lstm_config=lstm_config
    )
    print(f"  Final probability: {r2['final_probability']:.1%}")
    print(f"  Risk level:        {r2['risk_level']}")
    print(f"  CPA says:          {'CONFLICT' if r2['cpa_conflict'] else 'SAFE'}")
    if r2["lstm_probability"] is not None:
        print(f"  LSTM says:         {r2['lstm_probability']:.1%}")
    print(f"  Monte Carlo says:  {r2['mc_probability']:.1%}")

    # Case 3: Borderline
    print("\n--- Case 3: Borderline (where Monte Carlo matters most) ---")
    r3 = compute_conflict_probability(
        pos_a=(0, 0), vel_a=(150, 0), alt_a_ft=2000,
        pos_b=(8.5, 0.5), vel_b=(-150, 0), alt_b_ft=2950,
        lstm_model=lstm_model, lstm_scaler=lstm_scaler, lstm_config=lstm_config
    )
    print(f"  Final probability: {r3['final_probability']:.1%}")
    print(f"  Risk level:        {r3['risk_level']}")
    print(f"  CPA says:          {'CONFLICT' if r3['cpa_conflict'] else 'SAFE'}")
    if r3["lstm_probability"] is not None:
        print(f"  LSTM says:         {r3['lstm_probability']:.1%}")
    print(f"  Monte Carlo says:  {r3['mc_probability']:.1%}")
    print(f"  This borderline case shows Monte Carlo's value:")
    print(f"  CPA gives a binary answer, but Monte Carlo says {r3['mc_probability']:.1%}")
