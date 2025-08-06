import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from glob import glob

# Konstanta path
DATA_DIR = "data"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "labels.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X, y = [], []
for file in glob(f"{DATA_DIR}/*.npy"):
    label = ''.join([c for c in os.path.basename(file) if not c.isdigit()]).replace('.npy', '')
    sequence = np.load(file)  # (30, 108)
    
    if sequence.shape == (30, 108):  # ✅ validasi dengan 108 fitur/frame
        X.append(sequence)
        y.append(label)
    else:
        print(f"❌ Skip {file}, shape: {sequence.shape}")

X = np.array(X)  # shape: (N, 30, 108)
y = np.array(y)

# Encode label ke angka
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save mapping label
with open(LABEL_PATH, "w") as f:
    for label in le.classes_:
        f.write(label + "\n")

# Build LSTM model
model = models.Sequential([
    layers.Input(shape=(30, 108)),  # ✅ disesuaikan
    layers.LSTM(256, return_sequences=True),
    layers.LSTM(256),
    layers.Dense(256, activation="relu"),
    layers.Dense(len(le.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train model
model.fit(X, y_encoded, epochs=500, batch_size=4)

# Simpan model
model.save(MODEL_PATH)
print(f"✔️ Model disimpan ke: {MODEL_PATH}")
