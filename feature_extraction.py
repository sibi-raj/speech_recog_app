import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

def load_data(data_path):
    """
    Scans the data directory, extracts file paths and emotion labels.
    (This function is the same as in the previous step)
    """
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    data = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                try:
                    parts = file.split('-')
                    emotion_code = parts[2]
                    emotion_label = emotion_map.get(emotion_code)
                    if emotion_label:
                        full_path = os.path.join(root, file)
                        data.append({'path': full_path, 'label': emotion_label})
                except IndexError:
                    print(f"Skipping file with unexpected format: {file}")
    return pd.DataFrame(data)

def extract_features(file_path):
    """
    Extracts audio features from a given file path.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        numpy.ndarray: A numpy array containing the concatenated mean of
                       MFCCs, Chroma, and Mel-spectrogram features.
    """
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        
        # Extract Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        
        # Extract Mel-spectrogram features
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        
        # Concatenate all features into a single feature vector
        features = np.concatenate((mfccs, chroma, mel))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    DATA_PATH = './data/'
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory not found at '{DATA_PATH}'")
    else:
        # Load the initial data (paths and labels)
        audio_df = load_data(DATA_PATH)
        print("Initial data loaded. Starting feature extraction...")
        
        # Use tqdm for a progress bar as this process can take a while
        # We apply the extract_features function to each file path in the DataFrame
        features_list = [extract_features(path) for path in tqdm(audio_df['path'])]
        
        # Create a new DataFrame from the list of extracted features
        features_df = pd.DataFrame(features_list)
        
        # Concatenate the original DataFrame (with paths and labels) with the new features DataFrame
        final_df = pd.concat([audio_df, features_df], axis=1)
        
        # Drop rows with any missing values (in case some files failed to process)
        final_df.dropna(inplace=True)
        
        # Save the final DataFrame to a pickle file for easy loading later
        output_path = 'features.pkl'
        final_df.to_pickle(output_path)
        
        print(f"\nFeature extraction complete. Data saved to '{output_path}'")
        print("Here is a sample of the final DataFrame with features:")
        print(final_df.head())
