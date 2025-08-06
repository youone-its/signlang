import cv2
import numpy as np
import os
import mediapipe as mp
import time

# Setup MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Parameter direktori
DATA_DIR = "data"
VIDEO_DIR = "vid"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

gesture_name = input("Nama gesture: ")
num_samples = int(input("Berapa kali rekam gesture ini? "))

frames_per_sample = 30
sample_id = 0
frame_buffer = []

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

print("Tekan SPASI untuk mulai, 'x' untuk pause, 'z' untuk resume, 'q' untuk keluar.")

recording = False
paused = False
video_writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(image_rgb)

    keypoints = []

    # ==== AMBIL LANDMARK ====
    selected_ids = [
        33, 133,   # Mata kanan
        362, 263,  # Mata kiri
        61, 291    # Mulut kanan & kiri
    ]

    if result.face_landmarks:
        for i in selected_ids:
            lm = result.face_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * (len(selected_ids) * 2))

    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * 42)

    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * 42)

    if result.pose_landmarks:
        selected_pose_ids = [11, 12, 13, 14, 15, 16]  # shoulder & arms
        for i in selected_pose_ids:
            lm = result.pose_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints.extend([0] * 12)

    # ==== SIMPAN DATA ====
    if recording and not paused and keypoints:
        frame_buffer.append(keypoints)
        if video_writer:
            video_writer.write(frame)

        if len(frame_buffer) == frames_per_sample:
            # Simpan landmark ke file .npy
            np.save(f"{DATA_DIR}/{gesture_name}{sample_id}.npy", np.array(frame_buffer))
            print(f"✔️ Disimpan: {gesture_name}{sample_id}.npy")

            # Simpan video
            if video_writer:
                video_writer.release()
                print(f"🎥 Video disimpan: {VIDEO_DIR}/{gesture_name}{sample_id}.mp4")

            sample_id += 1
            frame_buffer = []

            if sample_id >= num_samples:
                print("✅ Semua sample selesai direkam!")
                break
            else:
                print("⏺️ Merekam... lakukan gesture sekarang!")

                # Mulai video baru untuk sample berikutnya
                video_path = os.path.join(VIDEO_DIR, f"{gesture_name}{sample_id}.mp4")
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            time.sleep(0.3)

    # ==== TAMPILKAN UI ====
    mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    status = f"{gesture_name}: {sample_id}/{num_samples} sample"
    if paused:
        status += " [PAUSED]"
    elif recording:
        status += " [RECORDING]"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Kamera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32 and not recording:  # SPASI
        recording = True
        print("⏺️ Merekam... lakukan gesture sekarang!")

        # Mulai video pertama
        video_path = os.path.join(VIDEO_DIR, f"{gesture_name}{sample_id}.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    elif key == ord('x'):
        paused = True
        print("⏸️ Pause")
    elif key == ord('z'):
        paused = False
        print("▶️ Resume")

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
