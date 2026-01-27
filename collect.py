import cv2
import mediapipe as mp
import pandas as pd
import os
import string


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

csv_file = 'sign_data.csv'
if not os.path.exists(csv_file):
    header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
    pd.DataFrame(columns=header).to_csv(csv_file, index=False)


cap = cv2.VideoCapture(0)
print("Press a key A–Z to label the hand gesture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            key = cv2.waitKey(10)
            if 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
                label = chr(key).upper()
                x = [lm.x for lm in hand_landmarks.landmark]
                y = [lm.y for lm in hand_landmarks.landmark]
                z = [lm.z for lm in hand_landmarks.landmark]
                row = x + y + z + [label]

                pd.DataFrame([row]).to_csv(csv_file, mode='a', header=False, index=False)
                print(f"✅ Saved gesture for '{label}'")

    cv2.imshow("Sign Data Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
