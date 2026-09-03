import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
from collections import deque

model = tf.keras.models.load_model("test_model.h5")

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

prediction_history = deque(maxlen=15)
sentence = [] 
confirmed_word = None
last_sign_time = time.time()
pause_between_signs = 1.0  # seconds between accepting same sign


cap = cv2.VideoCapture(0)

print(" Starting Sign Language Recognition... Press ESC to exit.")

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

                prediction_history.append(predicted_label)

                if prediction_history.count(predicted_label) > 10:
                    current_time = time.time()

                    if confirmed_word != predicted_label or (current_time - last_sign_time > pause_between_signs):
                        confirmed_word = predicted_label
                        last_sign_time = current_time

                        # Handle special gestures
                        if predicted_label.lower() == "space":
                            sentence.append(" ")
                            print(" Space added")
                        elif predicted_label.lower() == "del":
                            if sentence:
                                removed = sentence.pop()
                                print(f" Deleted: {removed}")
                        else:
                            sentence.append(predicted_label)
                            print(" Added:", predicted_label)

                cv2.putText(frame, f'{predicted_label}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # add '.' in 3 seconds if nothing
        if time.time() - last_sign_time > 3 and sentence and sentence[-1] != ".":
            sentence.append(".")
            print("\n Sentence:", "".join(sentence))
            last_sign_time = time.time()

    cv2.putText(frame, "".join(sentence[-25:]), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Sign Language to Text", frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break
cap.release()
cv2.destroyAllWindows()
hands.close()

print(" Program ended.")
