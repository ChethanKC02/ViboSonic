# genre_predictor.py

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize

model = tf.keras.models.load_model(".\Trained_model.h5")
classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    return np.array(data)

def predict_genre(file_path):
    X_test = load_and_preprocess_data(file_path)
    y_pred = model.predict(X_test)
    avg_probs = np.mean(y_pred, axis=0)
    return {classes[i]: round(avg_probs[i] * 100, 2) for i in range(len(classes))}
