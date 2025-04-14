# ViboSonic : Music Genre Classification System

ViboSonic is a deep learning-based music genre classification system that uses 
**Convolutional Neural Networks (CNN)** to analyze and classify audio tracks into 10 different genres. It leverages a trained model to predict genre probabilities from user-uploaded audio files.

### Features:
- **Audio Genre Classification**: Classifies audio into genres like Classical, Jazz, Pop, Rock, etc.
- **User-Friendly Web Interface**: Allows users to upload audio files and view results in real-time.
- **Pie Chart Visualization**: Displays the percentage match of the uploaded audio file's genres in an intuitive pie chart.
  
---

### Getting Started

Follow these steps to set up the ViboSonic project locally.

#### Prerequisites
Make sure you have Python 3.x and **pip** installed. You'll also need `git` for cloning the repository.

#### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ChethanKC02/ViboSonic.git
   cd vibosonic

2. **Create a Virtual Environment **(recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Run the Server**:
   ```bash
   python app.py
   ```
   This will start a local development server at http://127.0.0.1:5000/.
   
6. Upload an Audio File:
  - On the homepage, click the "Upload an audio file" button.
  - Select any audio file from your device (MP3, WAV, etc.).
  - Wait for the system to process the file, and it will display the genre predictions.

View the Results: The results will appear in a pie chart showing the confidence of different genres.
