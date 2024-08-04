import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('asl_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = hand_img.reshape(1, 64, 64, 1) / 255.0

            prediction = model.predict(hand_img)
            predicted_label = chr(np.argmax(prediction) + ord('A'))
            cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('ASL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
