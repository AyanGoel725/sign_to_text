import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("MODEL EVALUATION - Sign Language Recognition")
print("="*60)

# Load data
print("\n[1/5] Loading data.csv...")
df = pd.read_csv("data.csv")
print(f"Dataset shape: {df.shape}")

# Print class distribution
print("\n[2/5] Class distribution in data.csv:")
print("-" * 40)
class_counts = df['label'].value_counts().sort_index()
print(class_counts)
print(f"\nTotal samples: {len(df)}")
print(f"Number of classes: {len(class_counts)}")

# Prepare features and labels
X = df.iloc[:, :-1].values  # All columns except last
y = df["label"].values

# Load label encoder
print("\n[3/5] Loading label_encoder.pkl...")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Encode labels
y_encoded = encoder.transform(y)

# Recreate the same train/test split as training.py
# Using same random_state=42 and test_size=0.2
print("\n[4/5] Recreating train/test split (80/20, random_state=42)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Load model
print("\n[5/5] Loading test_model.h5...")
model = tf.keras.models.load_model("test_model.h5")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Run predictions on test set
print("\nRunning predictions on test set...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n" + "-"*60)
print("CLASSIFICATION REPORT")
print("-"*60)
report = classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_,
    digits=4
)
print(report)

print("\n" + "-"*60)
print("CONFUSION MATRIX")
print("-"*60)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save confusion matrix as image
print("\nGenerating confusion matrix visualization...")
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - ASL Recognition', fontsize=14, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved confusion matrix to: confusion_matrix.png")

# Per-class accuracy
print("\n" + "-"*60)
print("PER-CLASS ACCURACY")
print("-"*60)
for i, class_name in enumerate(encoder.classes_):
    class_mask = y_test == i
    if class_mask.sum() > 0:
        class_acc = (y_pred[class_mask] == y_test[class_mask]).mean()
        print(f"{class_name:>6}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_mask.sum()} samples")

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
