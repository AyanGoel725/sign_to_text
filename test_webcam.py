import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
from collections import deque

print("="*80)
print("MODEL COMPARISON - Real-time Testing")
print("="*80)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load all available models
models = {}
model_names = {
    "test_model.h5": "Original (Random Split)",
    "test_model_grouped.h5": "Grouped (Jump-based)",
    "test_model_clustered.h5": "Clustered (No Leakage)"
}

print("\nLoading models...")
for filename, display_name in model_names.items():
    try:
        models[display_name] = tf.keras.models.load_model(filename)
        print(f"  ✓ {display_name}")
    except:
        print(f"  ✗ {display_name} - not found")

if not models:
    print("\nNo models found! Train at least one model first.")
    exit(1)

# Select model
print("\n" + "="*80)
print("Available models:")
model_list = list(models.keys())
for i, name in enumerate(model_list, 1):
    print(f"  {i}. {name}")

while True:
    try:
        choice = input(f"\nSelect model (1-{len(model_list)}) or press Enter for default [1]: ").strip()
        if choice == "":
            choice = 1
        else:
            choice = int(choice)

        if 1 <= choice <= len(model_list):
            selected_model_name = model_list[choice - 1]
            model = models[selected_model_name]
            break
        else:
            print(f"Please enter a number between 1 and {len(model_list)}")
    except ValueError:
        print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\nExiting...")
        exit(0)

print(f"\nUsing: {selected_model_name}")
print("="*80)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Prediction smoothing
prediction_history = deque(maxlen=15)
sentence = []
confirmed_word = None
last_sign_time = time.time()
pause_between_signs = 1.0  # seconds

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

print("\nStarting Sign Language Recognition...")
print("Controls:")
print("  ESC - Exit")
print("  SPACE - Clear sentence")
print("  M - Switch model")
print("\n")

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
                confidence = np.max(prediction)
                predicted_label = encoder.inverse_transform([predicted_class])[0]

                prediction_history.append(predicted_label)

                if prediction_history.count(predicted_label) > 10:
                    current_time = time.time()

                    if confirmed_word != predicted_label or (current_time - last_sign_time > pause_between_signs):
                        confirmed_word = predicted_label
                        last_sign_time = current_time

                        # Handle special gestures
                        if predicted_label.lower() == "space":
                            sentence.append(" ")
                            print(" [Space added]")
                        elif predicted_label.lower() == "del":
                            if sentence:
                                removed = sentence.pop()
                                print(f" [Deleted: {removed}]")
                        else:
                            sentence.append(predicted_label)
                            print(f" Added: {predicted_label}")

                # Display prediction with confidence
                cv2.putText(frame, f'{predicted_label} ({confidence*100:.1f}%)',
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # Auto-add period after 3 seconds of no hand
        if time.time() - last_sign_time > 3 and sentence and sentence[-1] != ".":
            sentence.append(".")
            print(f"\n Sentence: {''.join(sentence)}")
            last_sign_time = time.time()

    # Display model name
    cv2.putText(frame, selected_model_name, (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Display sentence
    cv2.putText(frame, "".join(sentence[-25:]), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Sign Language to Text - Model Comparison", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE - clear sentence
        sentence = []
        print("\n [Sentence cleared]")
    elif key == ord('m') or key == ord('M'):  # M - switch model
        print("\n\nAvailable models:")
        for i, name in enumerate(model_list, 1):
            current = " (current)" if name == selected_model_name else ""
            print(f"  {i}. {name}{current}")

        try:
            choice = input(f"Select model (1-{len(model_list)}): ").strip()
            choice = int(choice)

            if 1 <= choice <= len(model_list):
                selected_model_name = model_list[choice - 1]
                model = models[selected_model_name]
                print(f"Switched to: {selected_model_name}\n")
            else:
                print("Invalid choice. Keeping current model.\n")
        except:
            print("Invalid input. Keeping current model.\n")

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n" + "="*80)
print("Final sentence:", "".join(sentence))
print("="*80)
print("Program ended.")
