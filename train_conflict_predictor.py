"""
Trains an LSTM (Long Short-Term Memory) neural network to predict
aircraft conflicts from trajectory SEQUENCES over time.


ARCHITECTURE:
  Input:       (batch_size, 10 timesteps, 14 features)
  LSTM Layer:  64 hidden units (learns temporal dynamics)
  Dense 1:     32 units with ReLU + Dropout
  Output:      2 classes (SAFE = 0, CONFLICT = 1) with Softmax

"""

import os
import glob
import math
import itertools
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from cpa import check_conflict
from data_utils import load_scene_file, KM_TO_FEET

# --- CONFIGURATION ---
DATA_FOLDER = r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data"
MAX_AIRCRAFT = 10
SEQ_LEN = 10          # 10 consecutive frames (10 seconds of trajectory history)
FRAME_STRIDE = 5      # Subsample sliding window every 5 frames
HIDDEN_SIZE = 64      # LSTM hidden units
FC_SIZE = 32          # Fully connected layer
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.002


def compute_features(pos_a, vel_a, alt_a_ft, pos_b, vel_b, alt_b_ft):
    """
    Computes 14 aviation-specific features for an aircraft pair at one timestep.
    """
    raw = [
        pos_a[0], pos_a[1], vel_a[0], vel_a[1], alt_a_ft,
        pos_b[0], pos_b[1], vel_b[0], vel_b[1], alt_b_ft,
    ]

    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    relative_distance = math.sqrt(dx * dx + dy * dy)

    rel_vx = vel_b[0] - vel_a[0]
    rel_vy = vel_b[1] - vel_a[1]
    if relative_distance > 0.001:
        closing_speed = -(dx * rel_vx + dy * rel_vy) / relative_distance
    else:
        closing_speed = 0.0

    altitude_diff = abs(alt_a_ft - alt_b_ft)

    heading_a = math.atan2(vel_a[1], vel_a[0])
    heading_b = math.atan2(vel_b[1], vel_b[0])
    heading_diff = abs(math.degrees(heading_a - heading_b))
    if heading_diff > 180:
        heading_diff = 360 - heading_diff

    return raw + [relative_distance, closing_speed, altitude_diff, heading_diff]


FEATURE_NAMES = [
    "pos_a_x", "pos_a_y", "vel_a_x", "vel_a_y", "alt_a",
    "pos_b_x", "pos_b_y", "vel_b_x", "vel_b_y", "alt_b",
    "relative_distance", "closing_speed", "altitude_diff", "heading_diff",
]
N_FEATURES = len(FEATURE_NAMES)


# ===== LSTM MODEL DEFINITION =====
class ConflictLSTM(nn.Module):
    """
    PyTorch LSTM for sequence-based conflict prediction.
    """

    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE, fc_size=FC_SIZE):
        super(ConflictLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc_size, 2)

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take the last hidden state representing the sequence summary
        last_hidden = h_n[-1]
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


# ===== DATASET CLASS =====
class ConflictSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ===== FAST PRECOMPUTED SEQUENCE EXTRACTION =====
def extract_sequences_from_folder(folder_path, seq_len=SEQ_LEN,
                                   frame_stride=FRAME_STRIDE,
                                   max_aircraft=MAX_AIRCRAFT):
    all_sequences = []
    all_labels = []

    files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    files_used = 0

    for file_path in files:
        df = load_scene_file(file_path)
        all_ids = df["ID"].unique()

        if len(all_ids) < 2:
            continue

        files_used += 1
        aircraft_ids = all_ids[:max_aircraft]

        # Fast precomputation of (pos, vel, alt) per aircraft per frame
        states = {}
        for ac_id in aircraft_ids:
            g = df[df["ID"] == ac_id].sort_values("Frame").reset_index(drop=True)
            frames = g["Frame"].values
            xs = g["x"].values
            ys = g["y"].values
            zs = g["z"].values * KM_TO_FEET

            vxs = np.zeros(len(g))
            vys = np.zeros(len(g))
            if len(g) > 1:
                dts = np.diff(frames).astype(float)
                dts[dts == 0] = 1.0
                vxs[:-1] = (np.diff(xs) / dts) * 3600.0
                vys[:-1] = (np.diff(ys) / dts) * 3600.0
                vxs[-1] = vxs[-2]
                vys[-1] = vys[-2]

            states[ac_id] = {
                frames[i]: ((xs[i], ys[i]), (vxs[i], vys[i]), zs[i])
                for i in range(len(frames))
            }

        # Pairwise sequence extraction
        for id_a, id_b in itertools.combinations(aircraft_ids, 2):
            shared_frames = sorted(set(states[id_a].keys()) & set(states[id_b].keys()))
            if len(shared_frames) < seq_len:
                continue

            for start_idx in range(0, len(shared_frames) - seq_len + 1, frame_stride):
                window = shared_frames[start_idx:start_idx + seq_len]

                seq_features = []
                for f in window:
                    pos_a, vel_a, alt_a = states[id_a][f]
                    pos_b, vel_b, alt_b = states[id_b][f]
                    feats = compute_features(pos_a, vel_a, alt_a, pos_b, vel_b, alt_b)
                    seq_features.append(feats)

                # Ground truth label at the end of the sequence
                last_f = window[-1]
                p_a, v_a, a_a = states[id_a][last_f]
                p_b, v_b, a_b = states[id_b][last_f]
                res = check_conflict(p_a, v_a, a_a, p_b, v_b, a_b)
                label = 1 if res["conflict"] else 0

                all_sequences.append(seq_features)
                all_labels.append(label)

    if len(all_sequences) == 0:
        return np.array([]).reshape(0, seq_len, N_FEATURES), np.array([]), files_used

    return np.array(all_sequences), np.array(all_labels), files_used


# ===== MAIN =====
if __name__ == "__main__":
    train_folder = os.path.join(DATA_FOLDER, "train")
    test_folder = os.path.join(DATA_FOLDER, "test")

    # ---- STEP 1: Extract training sequences ----
    print("=" * 60)
    print("STEP 1: Extract TRAINING sequences (from train/ folder)")
    print(f"        Sequence length: {SEQ_LEN} frames, Stride: {FRAME_STRIDE}")
    print("=" * 60)
    X_train, y_train, train_files = extract_sequences_from_folder(train_folder)
    train_conflicts = int(np.sum(y_train))
    train_safe = len(y_train) - train_conflicts
    print(f"  Files with 2+ aircraft: {train_files}")
    print(f"  Training sequences: {len(X_train)}")
    print(f"  Each sequence: {SEQ_LEN} timesteps x {N_FEATURES} features")
    print(f"  Conflicts: {train_conflicts}  |  Safe: {train_safe}")

    # ---- STEP 2: Extract test sequences ----
    print("\n" + "=" * 60)
    print("STEP 2: Extract TEST sequences (from test/ folder)")
    print("=" * 60)
    X_test, y_test, test_files = extract_sequences_from_folder(test_folder)
    test_conflicts = int(np.sum(y_test))
    test_safe = len(y_test) - test_conflicts
    print(f"  Files with 2+ aircraft: {test_files}")
    print(f"  Test sequences: {len(X_test)}")
    print(f"  Conflicts: {test_conflicts}  |  Safe: {test_safe}")

    if len(X_train) == 0 or len(X_test) == 0:
        raise SystemExit("No data found! Check DATA_FOLDER path.")

    # ---- STEP 3: Feature scaling ----
    print("\n" + "=" * 60)
    print("STEP 3: Scale features (standardization per-feature)")
    print("=" * 60)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_2d = X_train.reshape(-1, N_FEATURES)
    X_test_2d = X_test.reshape(-1, N_FEATURES)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_test_2d = scaler.transform(X_test_2d)

    X_train_scaled = X_train_2d.reshape(n_train, SEQ_LEN, N_FEATURES)
    X_test_scaled = X_test_2d.reshape(n_test, SEQ_LEN, N_FEATURES)

    print(f"  Features per timestep: {N_FEATURES} ({', '.join(FEATURE_NAMES)})")
    print(f"  Input tensor shape: ({SEQ_LEN}, {N_FEATURES}) per sequence")
    print("  Scaling complete (zero mean, unit variance)")

    # ---- STEP 4: Build and Train LSTM ----
    print("\n" + "=" * 60)
    print("STEP 4: Train LSTM neural network on train/ data")
    print(f"        Architecture: LSTM({HIDDEN_SIZE}) -> Dense({FC_SIZE}) -> 2 classes")
    print(f"        Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("=" * 60)

    train_dataset = ConflictSequenceDataset(X_train_scaled, y_train)
    test_dataset = ConflictSequenceDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ConflictLSTM(input_size=N_FEATURES, hidden_size=HIDDEN_SIZE, fc_size=FC_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("  Training LSTM across epochs:")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for seqs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc = correct / total
            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1:>2}/{EPOCHS}: Loss={avg_loss:.4f}, Train Acc={acc:.2%}")

    # Save model and config
    save_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(save_dir, "lstm_model.pth")
    scaler_path = os.path.join(save_dir, "lstm_scaler.pkl")
    config_path = os.path.join(save_dir, "lstm_config.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "fc_size": FC_SIZE,
    }, config_path)

    print(f"\n  Model saved to {model_path}")
    print(f"  Scaler saved to {scaler_path}")
    print(f"  Config saved to {config_path}")

    # ---- STEP 5: Test on test/ data ----
    print("\n" + "=" * 60)
    print("STEP 5: Test LSTM on test/ data (never seen during training)")
    print("=" * 60)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for seqs, labels in test_loader:
            outputs = model(seqs)
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")

    # ---- STEP 6: Comparison ----
    print("\n" + "=" * 60)
    print("STEP 6: Baseline vs LSTM Neural Network")
    print("=" * 60)
    majority = max(test_conflicts, test_safe) / len(y_test)
    print(f"  Baseline (always guess majority): {majority:.2%}")
    print(f"  LSTM neural network:              {accuracy:.2%}")
    improvement = accuracy - majority
    print(f"  Improvement:                      {improvement * 100:+.1f}%")

    print("\n  Confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"    Correctly said SAFE:                        {cm[0, 0]}")
    print(f"    False alarm (said CONFLICT, was SAFE):      {cm[0, 1]}")
    print(f"    Missed conflict (said SAFE, was CONFLICT):  {cm[1, 0]}")
    print(f"    Correctly said CONFLICT:                    {cm[1, 1]}")

    print("\n  Sample sequence predictions:")
    for i in range(min(5, len(y_true))):
        actual = "CONFLICT" if y_true[i] == 1 else "SAFE"
        predicted = "CONFLICT" if y_pred[i] == 1 else "SAFE"
        conf = y_probs[i, 1]
        match = "OK" if y_pred[i] == y_true[i] else "WRONG"
        print(f"    [{match}] Actual: {actual}, Predicted: {predicted} ({conf:.0%})")
