from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import os
import pickle

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Trained Model and Label Encoder ---
try:
    model = load_model('emotion_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    model = None
    label_encoder = None

def extract_features(file_path):
    """
    Extracts audio features from a given file path.
    This function MUST be identical to the one used for training.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        features = np.concatenate((mfccs, chroma, mel))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Define API Routes ---

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the audio file upload and returns the prediction."""
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model or label encoder not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Save the file temporarily
            temp_path = "temp_audio.wav"
            file.save(temp_path)

            # Extract features from the temporary file
            features = extract_features(temp_path)
            
            if features is not None:
                # Reshape features for the model
                features = np.expand_dims(features, axis=0) # For a single sample
                if len(model.input_shape) == 3: # Check if model is CNN
                    features = np.expand_dims(features, axis=2)

                # Get the prediction
                prediction = model.predict(features)
                predicted_index = np.argmax(prediction, axis=1)[0]
                
                # Decode the prediction back to the emotion name
                predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
                
                os.remove(temp_path)
                return jsonify({'emotion': predicted_emotion})
            else:
                os.remove(temp_path)
                return jsonify({'error': 'Failed to extract features from audio.'}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)

