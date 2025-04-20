from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from genre_predictor import predict_genre
import mimetypes
import librosa
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = ".uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define allowed extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav','ogg'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB in bytes

def allowed_file(filename):
    # Check the file extension
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_silent(audio_path, silence_threshold=0.01):
    # Load the audio file using librosa
    y, sr = librosa.load(audio_path)

    # Check if the average amplitude is below a threshold (silent audio)
    rms = librosa.feature.rms(y=y)[0]  # Root mean square of audio signal
    avg_rms = np.mean(rms)

    # If average RMS is below the silence threshold, consider it silent
    return avg_rms < silence_threshold

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    elif not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload an audio file."}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        file_size = os.path.getsize(file_path)

        if file_size > MAX_FILE_SIZE:
            os.remove(file_path)  # Clean up the uploaded file
            return jsonify({"error": "File size too large. Please upload a file smaller than 100MB."}), 400
        
        if is_silent(file_path):
            os.remove(file_path)  # Clean up the uploaded file
            return jsonify({"error": "The audio file is silent. No meaningful features detected."}), 400

        genre_probs = predict_genre(file_path)
        genre_probs = {k: round(float(v), 2) for k, v in genre_probs.items()}
        os.remove(file_path)  # Clean up the uploaded file
        return jsonify(genre_probs)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
