import cv2
import numpy as np
import os
import time
import mediapipe as mp
from tensorflow.keras.models import load_model

# Konstanta
MODEL_PATH = "model/gesture_model.h5"
LABEL_PATH = "model/labels.txt"
OUTPUT_FILE = "result.txt"
SEQ_LEN = 30
PREDICT_INTERVAL = 10
CONFIDENCE_THRESHOLD = 0.95

# Load model dan label
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Setup MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

# Kamera
cap = cv2.VideoCapture(0)
print("🖐️ Jalankan gesture... akan otomatis ditulis ke result.txt jika confident. Tekan 'q' untuk keluar.")

frame_buffer = []
frame_count = 0
latest_prediction = "Belum terdeteksi"
last_label_written = ""
# last_valid_label = ""

# Hapus isi file result.txt di awal (opsional)
open(OUTPUT_FILE, "w").close()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(image_rgb)

    keypoints = []

    # Wajah (mata & mulut)
    if result.face_landmarks:
        face_ids = [33, 133, 362, 263, 61, 291]
        for i in face_ids:
            lm = result.face_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * (6 * 2))

    # Tangan kiri
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * 21 * 2)

    # Tangan kanan
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * 21 * 2)

    # Pose
    if result.pose_landmarks:
        pose_ids = [11, 12, 13, 14, 15, 16]
        for i in pose_ids:
            lm = result.pose_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * 6 * 2)

    frame_buffer.append(keypoints)
    if len(frame_buffer) > SEQ_LEN:
        frame_buffer.pop(0)

    if len(frame_buffer) == SEQ_LEN and frame_count % PREDICT_INTERVAL == 0:
        input_seq = np.expand_dims(frame_buffer, axis=0)
        pred = model.predict(input_seq)[0]  # shape: (num_labels,)
        pred_idx = np.argmax(pred)
        confidence = pred[pred_idx]
        label = labels[pred_idx]

        if confidence >= CONFIDENCE_THRESHOLD and label.lower() != "idle":
            latest_prediction = f"{label} ({confidence:.2f})"

            # Cegah duplikasi langsung (misal: "halo halo")
            if label != last_label_written:
                with open(OUTPUT_FILE, "a") as f:
                    f.write(label + " ")
                print(f"✔️ Ditulis: {label}")
                last_label_written = label
        else:
            latest_prediction = f"[Diabaikan] {label} ({confidence:.2f})"

        # if confidence >= CONFIDENCE_THRESHOLD:
        #     if label.lower() == "idle":
        #         if last_label_written != last_valid_label and last_valid_label:
        #             with open(OUTPUT_FILE, "a") as f:
        #                 f.write(last_valid_label + " ")
        #             print(f"📝 Ditulis saat idle: {last_valid_label}")
        #             last_label_written = last_valid_label
        #         latest_prediction = f"[Idle] ({confidence:.2f})"
        #     else:
        #         last_valid_label = label
        #         latest_prediction = f"{label} ({confidence:.2f})"
        # else:
        #     latest_prediction = f"[Confidence rendah] {label} ({confidence:.2f})"



    frame_count += 1

    # Gambar landmark
    mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Tampilkan prediksi
    cv2.putText(frame, f"Prediksi: {latest_prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
