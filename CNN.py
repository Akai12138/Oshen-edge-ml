import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# path
base_dir = "./AudioWAV"

# convert audio to a spectrogram with consistent shape
def audio_to_spectrogram(file_path, max_pad_len):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    padded_S_DB = np.pad(S_DB, ((0, 0), (0, max_pad_len - S_DB.shape[1])), mode='constant') if S_DB.shape[1] < max_pad_len else S_DB[:, :max_pad_len]
    return padded_S_DB[..., np.newaxis]

def determine_max_length(base_dir):
    max_len = 0
    for filename in os.listdir(base_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(base_dir, filename)
            y, sr = librosa.load(file_path)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            if S.shape[1] > max_len:
                max_len = S.shape[1]
    return max_len

def load_data(base_dir, max_pad_len):
    X = []
    y = []
    for filename in os.listdir(base_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(base_dir, filename)
            label = 0 if "IN" in filename else 1
            spectrogram = audio_to_spectrogram(file_path, max_pad_len)
            X.append(spectrogram)
            y.append(label)
    return np.array(X), np.array(y)

max_length = determine_max_length(base_dir)
X, y = load_data(base_dir, max_length)
y = LabelBinarizer().fit_transform(y)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# simple CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")
