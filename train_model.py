import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers, metrics, callbacks

# Konstanta
DATA_DIR = "data"
MODEL_PATH = "model/gesture_model.h5"
LABEL_ENCODER_PATH = "model/labels.txt"

# Augmentasi: tambah noise kecil
def add_noise(sequence, noise_level=0.0):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

# Load data dan augmentasi
X, y = [], []
for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        label = ''.join([c for c in os.path.splitext(file)[0] if not c.isdigit()])
        seq = np.load(os.path.join(DATA_DIR, file))
        if seq.shape == (30, 108):
            X.append(add_noise(seq))
            y.append(label)
        else:
            print(f"❌ Skip {file} shape {seq.shape}")

X = np.array(X)
y = np.array(y)

# Encode label ke angka
le = LabelEncoder()
y_encoded = le.fit_transform(y)
with open(LABEL_ENCODER_PATH, "w") as f:
    for label in le.classes_:
        f.write(label + "\n")

# Build LSTM model
model = models.Sequential([
    layers.Input(shape=(30, 108)),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(128),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(le.classes_), activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        metrics.SparseCategoricalAccuracy(name="accuracy"),
    ]
)

# Callback: EarlyStopping
# early_stop = callbacks.EarlyStopping(
#     monitor="loss", patience=10, restore_best_weights=True
# )

# Train
history = model.fit(
    X, y_encoded,
    epochs=500,
    batch_size=64,
    validation_split=0.3,
    # callbacks=[early_stop]
)

# Save model
model.save(MODEL_PATH)
print(f"✔️ Model saved to {MODEL_PATH}")

# Plot training result
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["accuracy"], label="Accuracy")
plt.title("Training Loss & Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_result.png")
plt.show()
