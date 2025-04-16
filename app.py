
from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from genre_predictor import predict_genre

app = Flask(__name__)
UPLOAD_FOLDER = ".uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        genre_probs = predict_genre(file_path)
        genre_probs = {k: round(float(v), 2) for k, v in genre_probs.items()}
        os.remove(file_path)  # Clean up the uploaded file
        return jsonify(genre_probs)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)