import os
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Define the paths
base_dir = "./AudioWAV"

# convert audio to a flattened spectrogram with consistent length
def audio_to_flattened_spectrogram(file_path, max_pad_len):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    padded_S_DB = np.pad(S_DB, ((0, 0), (0, max_pad_len - S_DB.shape[1])), mode='constant') if S_DB.shape[1] < max_pad_len else S_DB[:, :max_pad_len]
    return padded_S_DB.flatten()

# determine the max length for padding/truncating by analyzing audio data
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
            if "IN" in filename:
                label = 0  # Category "IN"
            elif "NOT" in filename:
                label = 1  # Category "NOT"
            spectrogram = audio_to_flattened_spectrogram(file_path, max_pad_len)
            X.append(spectrogram)
            y.append(label)
    return np.array(X), np.array(y)

# maximum spectrogram length
max_length = determine_max_length(base_dir)
X, y = load_data(base_dir, max_length)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Training a KNN model
knn = KNeighborsClassifier(n_neighbors=10) 
knn.fit(X_train, y_train)

# Evaluate the model on validation and test sets
y_val_pred = knn.predict(X_val)
y_test_pred = knn.predict(X_test)

print("Validation Set Evaluation:")
print(classification_report(y_val, y_val_pred))
print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred))
