import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("MODEL EVALUATION & DATA LEAKAGE ANALYSIS - Sign Language Recognition")
print("="*80)

# Load data
print("\n[1/5] Loading data.csv...")
df = pd.read_csv("data.csv")
print(f"Dataset shape: {df.shape}")

# Prepare features and labels
X = df.iloc[:, :-1].values  # All columns except last
y = df["label"].values

# Load label encoder
print("\n[2/5] Loading model/label_encoder.pkl...")
with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Encode labels
y_encoded = encoder.transform(y)

# Load model
print("\n[3/5] Loading model/test_model.h5...")
model = tf.keras.models.load_model("model/test_model.h5")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# --- 1. Random Split Evaluation ---
print("\n[4/5] Recreating random train/test split (80/20, random_state=42)...")
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

y_pred_probs_rand = model.predict(X_test_rand, verbose=0)
y_pred_rand = np.argmax(y_pred_probs_rand, axis=1)
acc_rand = accuracy_score(y_test_rand, y_pred_rand)

def get_weakest_classes(y_true, y_pred, encoder_classes, top_k=5):
    class_accs = {}
    for i, class_name in enumerate(encoder_classes):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            acc = (y_pred[class_mask] == y_true[class_mask]).mean()
            class_accs[class_name] = acc
    return sorted(class_accs.items(), key=lambda item: item[1])[:top_k]

weak_rand = get_weakest_classes(y_test_rand, y_pred_rand, encoder.classes_)

# --- 2. Positional Split Evaluation ---
print("\n[5/5] Creating positional train/test split (First 80% train / Last 20% test per class)...")
X_train_pos, X_test_pos, y_train_pos, y_test_pos = [], [], [], []

for classId in range(len(encoder.classes_)):
    class_indices = np.where(y_encoded == classId)[0]
    # Assuming rows for a class are contiguous or at least sorted sequentially in time
    # Actually, as we established, they are grouped in blocks. We can just sort them by index to be safe.
    class_indices = sorted(class_indices)

    split_idx = int(0.8 * len(class_indices))
    train_idx = class_indices[:split_idx]
    test_idx = class_indices[split_idx:]

    X_train_pos.extend(X[train_idx])
    X_test_pos.extend(X[test_idx])
    y_train_pos.extend(y_encoded[train_idx])
    y_test_pos.extend(y_encoded[test_idx])

X_train_pos, np_y_train_pos = np.array(X_train_pos), np.array(y_train_pos)
X_test_pos, np_y_test_pos = np.array(X_test_pos), np.array(y_test_pos)

y_pred_probs_pos = model.predict(X_test_pos, verbose=0)
y_pred_pos = np.argmax(y_pred_probs_pos, axis=1)
acc_pos = accuracy_score(np_y_test_pos, y_pred_pos)

weak_pos = get_weakest_classes(np_y_test_pos, y_pred_pos, encoder.classes_)

# Print comparison
print("\n" + "="*80)
print("EVALUATION COMPARISON: Random vs Positional Split")
print("="*80)

print(f"{'Metric':<30} | {'Random Split':<20} | {'Positional Split':<20}")
print("-" * 75)
print(f"{'Overall Accuracy':<30} | {acc_rand*100:6.2f}%              | {acc_pos*100:6.2f}%")

print("\nWeakest Classes (Accuracy):")
print(f"[{'Random Split':^30}] | [{'Positional Split':^30}]")
for (cls_r, acc_r), (cls_p, acc_p) in zip(weak_rand, weak_pos):
    print(f" {cls_r:>13}: {acc_r*100:6.2f}%               |  {cls_p:>13}: {acc_p*100:6.2f}%")


print("\n" + "="*80)
print("DATA LEAKAGE ANALYSIS")
print("="*80)

# --- 3. Near-Duplicate Check (Leakage Confirmation) ---
print("\nChecking for near-duplicates between Random Train and Test sets...")
# Threshold for considering two samples "near-duplicates"
THRESHOLD = 0.05
duplicates_found = 0

for classId in range(len(encoder.classes_)):
    # Find all train and test samples for this class
    train_mask = y_train_rand == classId
    test_mask = y_test_rand == classId

    X_train_class = X_train_rand[train_mask]
    X_test_class = X_test_rand[test_mask]

    if len(X_train_class) == 0 or len(X_test_class) == 0:
        continue

    # Fit nearest neighbors on training data for this class
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(X_train_class)

    # Query test data
    distances, _ = nn.kneighbors(X_test_class)

    # Count how many test samples have a training neighbor within THRESHOLD
    duplicates_found += (distances < THRESHOLD).sum()

leak_pct = (duplicates_found / len(X_test_rand)) * 100
print(f"Test samples with near-duplicate in Train (dist < {THRESHOLD}): {duplicates_found}/{len(X_test_rand)} ({leak_pct:.2f}%)")


# --- 4. Consecutive Jump Detection (Detecting Recording Sessions) ---
print("\nDetecting consecutive jumps (session breaks) within class blocks...")
JUMP_THRESHOLD = 0.5
total_jumps = 0

for classId in range(len(encoder.classes_)):
    class_indices = np.where(y_encoded == classId)[0]
    class_indices = sorted(class_indices)

    if len(class_indices) < 2:
        continue

    X_class_ordered = X[class_indices]

    # Calculate euclidean distance between consecutive rows
    diffs = np.diff(X_class_ordered, axis=0)
    dists = np.linalg.norm(diffs, axis=1)

    jumps = np.where(dists > JUMP_THRESHOLD)[0]
    num_jumps = len(jumps)
    total_jumps += num_jumps

    if num_jumps > 0:
        print(f"Class '{encoder.classes_[classId]}': {num_jumps} significant jump(s) detected (max jump = {np.max(dists):.3f})")

print(f"Total session jumps detected across dataset: {total_jumps}")
print("\n" + "="*80)
