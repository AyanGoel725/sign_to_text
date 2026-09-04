import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
import glob as globmod
from collections import Counter, deque

# --- Model discovery & loading ---
# Find all .h5 model files in the project root (skip .venv/).
MODEL_PATHS = sorted(
    p for p in globmod.glob("*.h5")
)
if not MODEL_PATHS:
    raise FileNotFoundError("No .h5 model files found in the project root.")

current_model_idx = 0
model = tf.keras.models.load_model(MODEL_PATHS[current_model_idx])

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


def switch_model(idx):
    """Load a different model by index. Returns the new model and clears the buffer."""
    global model, current_model_idx
    idx = idx % len(MODEL_PATHS)
    current_model_idx = idx
    model = tf.keras.models.load_model(MODEL_PATHS[current_model_idx])
    print(f"\n  *** Switched to model [{current_model_idx + 1}/{len(MODEL_PATHS)}]: "
          f"{MODEL_PATHS[current_model_idx]} ***\n")
    return model


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Majority-vote smoothing: commit a letter only when one prediction
# holds >= SMOOTHING_THRESHOLD of the last SMOOTHING_WINDOW frames.
SMOOTHING_MIN = 3
SMOOTHING_MAX = 30
smoothing_window = 10
SMOOTHING_THRESHOLD = 0.70  # 70% of window

prediction_buffer = deque(maxlen=smoothing_window)
sentence = []
confirmed_word = None
last_sign_time = time.time()
pause_between_signs = 1.0  # seconds between accepting same sign


def set_smoothing_window(new_n):
    """Resize the prediction buffer, keeping as many recent predictions as fit."""
    global smoothing_window, prediction_buffer
    new_n = max(SMOOTHING_MIN, min(SMOOTHING_MAX, new_n))
    smoothing_window = new_n
    # Rebuild with new maxlen, preserving the tail of the old buffer.
    old_items = list(prediction_buffer)
    prediction_buffer = deque(old_items[-new_n:], maxlen=new_n)
    print(f"\n  *** Smoothing window set to N={smoothing_window} ***\n")


cap = cv2.VideoCapture(0)

print("=" * 60)
print(" Sign Language Recognition")
print("=" * 60)
print(f"  Model  [{current_model_idx + 1}/{len(MODEL_PATHS)}]: {MODEL_PATHS[current_model_idx]}")
print(f"  Smoothing window: N={smoothing_window}  (threshold {SMOOTHING_THRESHOLD:.0%})")
print()
print("  Controls:")
print("    1-9     Switch model (by number)")
print("    +/=     Increase smoothing window N")
print("    -       Decrease smoothing window N")
print("    ESC     Exit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            z_coords = [lm.z for lm in hand_landmarks.landmark]
            landmarks = x_coords + y_coords + z_coords

            if len(landmarks) == model.input_shape[1]:
                prediction = model.predict(np.array([landmarks]), verbose=0)
                predicted_class = np.argmax(prediction)
                predicted_label = encoder.inverse_transform([predicted_class])[0]

                prediction_buffer.append(predicted_label)

                # --- Majority-vote smoothing ---
                # Find the most common prediction in the buffer and its share.
                if len(prediction_buffer) == smoothing_window:
                    counts = Counter(prediction_buffer)
                    majority_label, majority_count = counts.most_common(1)[0]
                    majority_pct = majority_count / smoothing_window

                    if majority_pct >= SMOOTHING_THRESHOLD:
                        current_time = time.time()

                        if confirmed_word != majority_label or (current_time - last_sign_time > pause_between_signs):
                            confirmed_word = majority_label
                            last_sign_time = current_time
                            prediction_buffer.clear()

                            # Handle special gestures
                            if majority_label.lower() == "space":
                                sentence.append(" ")
                                print(f"  [raw: {predicted_label}]  >> COMMITTED: <space>")
                            elif majority_label.lower() == "del":
                                if sentence:
                                    removed = sentence.pop()
                                    print(f"  [raw: {predicted_label}]  >> COMMITTED: <del> (removed '{removed}')")
                            else:
                                sentence.append(majority_label)
                                print(f"  [raw: {predicted_label}]  >> COMMITTED: {majority_label}")
                        else:
                            # Same sign too soon — log but don't commit.
                            print(f"  [raw: {predicted_label}]  (majority {majority_label} {majority_pct:.0%}, waiting for pause)")
                    else:
                        # Buffer doesn't have a clear winner yet.
                        print(f"  [raw: {predicted_label}]  (no majority — top: {majority_label} {majority_pct:.0%})")
                else:
                    # Buffer not full yet.
                    print(f"  [raw: {predicted_label}]  (buffering {len(prediction_buffer)}/{smoothing_window})")

                # Show the raw per-frame prediction on screen.
                cv2.putText(frame, f'Raw: {predicted_label}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Show the last committed letter (if any).
                if confirmed_word:
                    cv2.putText(frame, f'Committed: {confirmed_word}', (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # add '.' in 3 seconds if nothing
        if time.time() - last_sign_time > 3 and sentence and sentence[-1] != ".":
            sentence.append(".")
            print("\n Sentence:", "".join(sentence))
            last_sign_time = time.time()

    # --- On-screen display ---
    # Sentence
    cv2.putText(frame, "".join(sentence[-25:]), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Current model and smoothing info
    model_name = MODEL_PATHS[current_model_idx]
    cv2.putText(frame, f'Model [{current_model_idx+1}/{len(MODEL_PATHS)}]: {model_name}',
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f'N={smoothing_window}  threshold={SMOOTHING_THRESHOLD:.0%}',
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Sign Language to Text", frame)

    # --- Key handling ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif ord('1') <= key <= ord('9'):
        # Switch model — keys 1-9 map to model indices 0-8.
        requested_idx = key - ord('1')
        if requested_idx < len(MODEL_PATHS):
            switch_model(requested_idx)
            prediction_buffer.clear()
        else:
            print(f"  (no model at index {requested_idx + 1}, only {len(MODEL_PATHS)} available)")
    elif key in (ord('+'), ord('=')):
        set_smoothing_window(smoothing_window + 1)
    elif key == ord('-'):
        set_smoothing_window(smoothing_window - 1)
cap.release()
cv2.destroyAllWindows()
hands.close()

print(" Program ended.")
