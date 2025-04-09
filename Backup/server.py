from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("Trained_model.h5")
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Preprocess audio
def load_and_preprocess_data(audio_path, target_shape=(150, 150)):
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    chunk_duration, overlap_duration = 4, 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = max(1, (len(audio_data) - chunk_samples) // (chunk_samples - overlap_samples) + 1)
    
    data = []
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

@app.route('/predict', methods=['POST'])
def predict():
    # file = request.files['file']
    # file_path = "temp_audio.mp3"
    # file.save(file_path)
    file_path = "./TestAudio/jazz.mp3"

    X_test = load_and_preprocess_data(file_path)
    y_pred = model.predict(X_test)
    avg_probs = np.mean(y_pred, axis=0)
    genre_probabilities = {classes[i]: round(avg_probs[i] * 100, 1) for i in range(len(classes))}
    
    return jsonify(genre_probabilities)

if __name__ == '__main__':
    app.run(debug=True)
