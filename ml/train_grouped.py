import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors

print("="*60)
print("TRAINING WITH GROUP-AWARE SPLIT")
print("="*60)

# 1. Load data
print("[1/5] Loading data and determining sessions...")
df = pd.read_csv("data.csv")
X = df.iloc[:, :-1].values
y = df["label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Construct session_ids based on jumps
JUMP_THRESHOLD = 0.5
session_ids = np.zeros(len(df), dtype=int)
current_session_id = 0

for classId in range(len(encoder.classes_)):
    # Ensure sequential index mapping
    class_indices = np.where(y_encoded == classId)[0]
    class_indices = np.sort(class_indices)

    if len(class_indices) == 0: continue

    X_class = X[class_indices]
    diffs = np.diff(X_class, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    jumps = dists > JUMP_THRESHOLD

    # First row of class gets current_session_id
    session_ids[class_indices[0]] = current_session_id

    for i, is_jump in enumerate(jumps):
        if is_jump:
            current_session_id += 1
        session_ids[class_indices[i+1]] = current_session_id
    current_session_id += 1 # Ensure completely disjoint sessions across classes

num_sessions = len(np.unique(session_ids))
print(f"Total extracted sessions (blocks): {num_sessions}")

# 2. Group Shuffle Split
print("[2/5] Creating train/test split (80/20 grouped)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups=session_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Early stopping validation split (also grouped to prevent leakage during val_loss monitoring)
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# Here session_ids[train_idx] are the original session ids relative to X_train array
train_val_idx, val_idx = next(gss_val.split(X_train, y_train, groups=session_ids[train_idx]))

X_train_final, X_val = X_train[train_val_idx], X_train[val_idx]
y_train_final = to_categorical(y_train[train_val_idx], num_classes=len(encoder.classes_))
y_val = to_categorical(y_train[val_idx], num_classes=len(encoder.classes_))

# 3. Model Training
print("[3/5] Compiling and training model...")
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

print("\n[4/5] Saving grouped model...")
model.save("model/test_model_grouped.h5")

# 4. Evaluation and Leakage report
print("[5/5] Evaluating and checking leakage...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*60)
print("GROUPED SPLIT RESULTS")
print("="*60)
print(f"Overall Accuracy: {accuracy*100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 5. Near duplicate check
print("\nChecking for near-duplicates between New Train and Test sets...")
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
print("="*60)
