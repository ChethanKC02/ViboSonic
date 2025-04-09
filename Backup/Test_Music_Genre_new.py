import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.image import resize
import seaborn as sns

#Loading Model
model = tf.keras.models.load_model("Trained_model.h5")

classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
        # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
        # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
        # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
        #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)


# Model Prediction
def model_prediction(X_test):
    y_pred = model.predict(X_test)  # Get probability distributions
    avg_probs = np.mean(y_pred, axis=0)  # Average across all chunks
    class_probabilities = {classes[i]: round(avg_probs[i] * 100, 2) for i in range(len(classes))}
    return class_probabilities

# Test with an audio file
file_path = "./Data/genres_original/blues/blues.00099.wav"
X_test = load_and_preprocess_data(file_path)
genre_probabilities = model_prediction(X_test)

# Print genre probabilities
print("\nGenre Probabilities:")
for genre, prob in genre_probabilities.items():
    print(f"{genre.ljust(10)}: {prob:.1f}%")