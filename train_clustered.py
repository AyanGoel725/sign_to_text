import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors

print("="*80)
print("TRAINING WITH SIMILARITY-BASED CLUSTERING")
print("="*80)

# 1. Load data
print("\n[1/6] Loading data...")
df = pd.read_csv("data.csv")
X = df.iloc[:, :-1].values
y = df["label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 2. Cluster samples within each class by landmark similarity
print("[2/6] Clustering samples by landmark similarity (distance < 0.05)...")
DISTANCE_THRESHOLD = 0.05
session_ids = np.zeros(len(df), dtype=int)
current_session_id = 0

cluster_stats = []

for classId in range(len(encoder.classes_)):
    class_name = encoder.classes_[classId]
    class_mask = y_encoded == classId
    class_indices = np.where(class_mask)[0]

    if len(class_indices) == 0:
        continue

    X_class = X[class_indices]

    # Use DBSCAN with precomputed distance metric
    # eps=DISTANCE_THRESHOLD means points within 0.05 are in same cluster
    # min_samples=1 allows singleton clusters (isolated samples)
    clustering = DBSCAN(eps=DISTANCE_THRESHOLD, min_samples=1, metric='euclidean')
    cluster_labels = clustering.fit_predict(X_class)

    # Assign global session IDs
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)

    for local_cluster_id in unique_clusters:
        cluster_mask = cluster_labels == local_cluster_id
        global_indices = class_indices[cluster_mask]
        session_ids[global_indices] = current_session_id
        current_session_id += 1

    cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]
    cluster_stats.append({
        'class': class_name,
        'num_clusters': num_clusters,
        'min_size': np.min(cluster_sizes),
        'max_size': np.max(cluster_sizes),
        'mean_size': np.mean(cluster_sizes),
        'total_samples': len(class_indices)
    })

    print(f"  {class_name:>6}: {num_clusters:4d} clusters | "
          f"size range [{np.min(cluster_sizes):4d}, {np.max(cluster_sizes):4d}] | "
          f"mean {np.mean(cluster_sizes):6.1f}")

    # Warning for classes with low diversity
    if num_clusters < 5:
        print(f"    ⚠️  WARNING: Only {num_clusters} clusters - low diversity!")

num_sessions = len(np.unique(session_ids))
print(f"\nTotal clusters across all classes: {num_sessions}")

# 3. Group Shuffle Split
print("\n[3/6] Creating train/test split (80/20 grouped by clusters)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups=session_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Grouped validation split
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, val_idx = next(gss_val.split(X_train, y_train, groups=session_ids[train_idx]))

X_train_final, X_val = X_train[train_val_idx], X_train[val_idx]
y_train_final = to_categorical(y_train[train_val_idx], num_classes=len(encoder.classes_))
y_val = to_categorical(y_train[val_idx], num_classes=len(encoder.classes_))

# 4. Model Training
print("\n[4/6] Compiling and training model...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train_final,
    y_train_final,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

print("\n[5/6] Saving clustered model...")
model.save("test_model_clustered.h5")

# 5. Evaluation and Leakage Check
print("\n[6/6] Evaluating and checking leakage...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*80)
print("CLUSTER-BASED SPLIT RESULTS")
print("="*80)
print(f"Overall Accuracy: {accuracy*100:.2f}%")

print("\nPer-class accuracy (5 weakest):")
class_accs = []
for i, class_name in enumerate(encoder.classes_):
    class_mask = y_test == i
    if class_mask.sum() > 0:
        class_acc = (y_pred[class_mask] == y_test[class_mask]).mean()
        class_accs.append((class_name, class_acc, class_mask.sum()))

class_accs.sort(key=lambda x: x[1])
for class_name, acc, count in class_accs[:5]:
    print(f"  {class_name:>6}: {acc*100:6.2f}% ({count:4d} samples)")

# 6. Near-duplicate leakage check
print("\n" + "-"*80)
print("LEAKAGE VERIFICATION")
print("-"*80)
print("Checking for near-duplicates between Train and Test sets...")
THRESHOLD = 0.05
duplicates_found = 0

for classId in range(len(encoder.classes_)):
    train_mask = y_train == classId
    test_mask = y_test == classId

    X_train_class = X_train[train_mask]
    X_test_class = X_test[test_mask]

    if len(X_train_class) == 0 or len(X_test_class) == 0:
        continue

    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(X_train_class)
    distances, _ = nn.kneighbors(X_test_class)
    duplicates_found += (distances < THRESHOLD).sum()

leak_pct = (duplicates_found / len(X_test)) * 100

print(f"Test samples with near-duplicate in Train (dist < {THRESHOLD}): {duplicates_found}/{len(X_test)} ({leak_pct:.2f}%)")

if leak_pct < 1.0:
    print("✓ Leakage successfully eliminated!")
elif leak_pct < 5.0:
    print("⚠️  Minimal leakage remaining (< 5%)")
else:
    print("❌ Significant leakage still present")

print("\n" + "="*80)
print("Cluster statistics saved. Model saved as test_model_clustered.h5")
print("="*80)

# Save cluster stats for analysis
cluster_df = pd.DataFrame(cluster_stats)
cluster_df.to_csv("cluster_statistics.csv", index=False)
print("\nCluster statistics saved to: cluster_statistics.csv")
