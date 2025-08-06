import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# Load model & label
model = load_model("model/gesture_model.h5")
with open("model/labels.txt", "r") as f:
    labels = f.read().splitlines()

# Setup MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                smooth_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Buffer buat 30 frame
seq_length = 30
buffer = deque(maxlen=seq_length)

# Ambil keypoints dalam bentuk (108,)
def extract_keypoints(results):
    keypoints = []

    # Face: 468 titik
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0.0] * 468 * 2)

    # Left Hand: 21 titik
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0.0] * 21 * 2)

    # Right Hand: 21 titik
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0.0] * 21 * 2)

    # Pose (6 titik): [11,12,13,14,23,24]
    selected = [11, 12, 13, 14, 23, 24]
    if results.pose_landmarks:
        for i in selected:
            lm = results.pose_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0.0] * 6 * 2)

    return np.array(keypoints)

# Kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    # Gambar landmark
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Ambil keypoints per frame
    keypoints = extract_keypoints(results)
    buffer.append(keypoints)

    # Kalau cukup 30 frame → prediksi
    if len(buffer) == seq_length:
        input_data = np.expand_dims(buffer, axis=0)  # shape: (1, 30, 108)
        pred = model.predict(input_data, verbose=0)
        label = labels[np.argmax(pred)]
        prob = np.max(pred)

        if prob > 0.8:
            cv2.putText(frame, f"{label} ({prob:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Dynamic Sign Language", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
