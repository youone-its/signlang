import cv2
import numpy as np
import os
import mediapipe as mp
import time

# Setup MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Parameter
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

gesture_name = input("Nama gesture: ")
num_samples = int(input("Berapa kali rekam gesture ini? "))

frames_per_sample = 30
sample_id = 0
frame_buffer = []

cap = cv2.VideoCapture(0)
print("Tekan SPASI untuk mulai, 'x' untuk pause, 'z' untuk resume, 'q' untuk keluar.")

recording = False
paused = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(image_rgb)

    keypoints = []

    # ==== AMBIL TITIK-TITIK LANDMARK ====

    # Ambil titik penting di wajah: mata dan mulut
    if result.face_landmarks:
        selected_ids = [
            33, 133,     # Mata kanan
            362, 263,    # Mata kiri
            61, 291,     # Mulut kanan & kiri
        ]
        for i in selected_ids:
            lm = result.face_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * (len(selected_ids) * 2))

    # Tangan kiri
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * 42)

    # Tangan kanan
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * 42)

    # Pose (bahu & lengan)
    if result.pose_landmarks:
        selected_pose_ids = [11, 12, 13, 14, 15, 16]  # shoulder & arms
        for i in selected_pose_ids:
            lm = result.pose_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * 12)

    # Simpan data jika sedang merekam
    if recording and not paused and keypoints:
        frame_buffer.append(keypoints)
        if len(frame_buffer) == frames_per_sample:
            np.save(f"{DATA_DIR}/{gesture_name}{sample_id}.npy", np.array(frame_buffer))
            print(f"✔️ Disimpan: {gesture_name}{sample_id}.npy")
            sample_id += 1
            frame_buffer = []
            time.sleep(0.3)

            if sample_id >= num_samples:
                print("✅ Semua sample selesai direkam!")
                break
            else:
                print("⏺️ Merekam... lakukan gesture sekarang!")

    # ==== TAMPILKAN VISUAL ====

    # Gambar tangan dan pose seperti biasa
    mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # UI Info
    status = f"{gesture_name}: {sample_id}/{num_samples} sample"
    if paused:
        status += " [PAUSED]"
    elif recording:
        status += " [RECORDING]"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Kamera", frame)

    # Control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32 and not recording:  # SPASI
        recording = True
        print("⏺️ Merekam... lakukan gesture sekarang!")
    elif key == ord('x'):
        paused = True
        print("⏸️ Pause")
    elif key == ord('z'):
        paused = False
        print("▶️ Resume")

cap.release()
cv2.destroyAllWindows()
