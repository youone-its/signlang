import cv2
import numpy as np
import os
import mediapipe as mp

# Konstanta
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

gesture_name = input("Nama gesture: ")
num_samples = int(input("Berapa sample gesture ini? (tiap sample = 30 klik space): "))
sample_id = 0
frame_buffer = []
frames_per_sample = 30

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

# Kamera
cap = cv2.VideoCapture(0)
print("🎥 Tekan SPASI untuk simpan 1 frame, 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(image_rgb)

    keypoints = []

    # Ambil landmark wajah
    face_ids = [33, 133, 362, 263, 61, 291]
    if result.face_landmarks:
        for i in face_ids:
            lm = result.face_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * (len(face_ids) * 2))

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
    pose_ids = [11, 12, 13, 14, 15, 16]
    if result.pose_landmarks:
        for i in pose_ids:
            lm = result.pose_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * len(pose_ids) * 2)

    # UI
    mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    cv2.putText(frame, f"{gesture_name} | Sample: {sample_id}/{num_samples} | Frame: {len(frame_buffer)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Kamera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == 32:  # SPASI
        if len(keypoints) == 108:
            frame_buffer.append(keypoints)
            print(f"🟩 Frame {len(frame_buffer)} ditambahkan")

        if len(frame_buffer) == frames_per_sample:
            filename = f"{gesture_name}{sample_id}.npy"
            np.save(os.path.join(DATA_DIR, filename), np.array(frame_buffer))
            print(f"✔️ Sample disimpan: {filename}")
            sample_id += 1
            frame_buffer = []

            if sample_id >= num_samples:
                print("✅ Semua sample selesai direkam.")
                break

cap.release()
cv2.destroyAllWindows()
