from flask import Flask, request, jsonify
import librosa
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from moviepy.editor import VideoFileClip

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and RFE selector
with open('model/dog_bark_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/rfe_selector.pkl', 'rb') as rfe_file:
    rfe = pickle.load(rfe_file)

# Function to extract features from audio
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.concatenate((np.mean(mfccs.T, axis=0),
                           np.mean(chroma.T, axis=0),
                           np.mean(spectral_contrast.T, axis=0)))

# Default route to show backend is running
@app.route('/')
def index():
    return "<h1>Backend is running</h1>"
@app.route('/predict', methods=['POST'])
def predict():
    if 'videofile' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files['videofile']
    print(f"Received file: {file.filename}, mimetype: {file.mimetype}")

    if file and (file.mimetype in ['video/mp4', 'application/octet-stream'] or file.filename.endswith('.mp4')):
        try:
            # Save and process the file
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            # Extract audio
            video = VideoFileClip(filepath)
            audio_path = filepath.replace(".mp4", ".wav")
            video.audio.write_audiofile(audio_path)
            print(f"Audio saved at: {audio_path}")

            # Load and extract features
            try:
                y, sr = librosa.load(audio_path, sr=22050)
                features = extract_features(y, sr)
                features_rfe = rfe.transform(features.reshape(1, -1))
                print(f"Extracted features: {features_rfe}")
            except Exception as e:
                print(f"Error during feature extraction: {e}")
                return jsonify({"status": "error", "message": f"Feature extraction failed: {e}"}), 500

            # Make prediction
            try:
                prediction = model.predict(features_rfe)[0]
                print(f"Prediction: {prediction}")
                return jsonify({"status": "success", "prediction": prediction})
            except Exception as e:
                print(f"Error during prediction: {e}")
                return jsonify({"status": "error", "message": f"Prediction failed: {e}"}), 500

        except Exception as e:
            print(f"Error processing video: {e}")
            return jsonify({"status": "error", "message": f"Processing failed: {e}"}), 500

    return jsonify({"status": "error", "message": "Invalid file type."}), 400




if __name__ == '__main__':
    # Ensure uploads folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, host='0.0.0.0', port=5000)



#ooooo