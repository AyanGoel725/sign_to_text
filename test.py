import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

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


cap = cv2.VideoCapture(0)

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
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  

           
            if len(landmarks) == model.input_shape[1]:
                prediction = model.predict(np.array([landmarks]), verbose=0)
                predicted_class = np.argmax(prediction)
                predicted_label = encoder.inverse_transform([predicted_class])[0]

                cv2.putText(frame, f'{predicted_label}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
