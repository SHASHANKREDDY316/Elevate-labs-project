import streamlit as st
import librosa
import numpy as np
import pandas as pd
import os
import joblib
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 22050
DURATION = 5  # seconds for recording
MODEL_PATH = 'emotion_model.pkl'
SCALER_PATH = 'scaler.pkl'
SESSION_DATA_PATH = 'session_data.csv'
ZIP_PATH = r'C:\Users\ssrnd\Downloads\archive (8).zip'  # Your provided zip file path

# Extract zip file
def extract_zip(zip_path, extract_path='ravdess'):
    try:
        if not os.path.exists(extract_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            st.write(f"Extracted {zip_path} to {extract_path}")
        else:
            st.write(f"Directory {extract_path} already exists")
        return extract_path
    except Exception as e:
        st.error(f"Error extracting zip file: {e}")
        return None

# Feature extraction function
def extract_features(audio_path, sample_rate=SAMPLE_RATE):
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate)
        # Extract MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        # Extract Chroma
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        # Extract Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        # Extract Spectral Contrast
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        # Combine features
        features = np.hstack([mfcc, chroma, mel, contrast])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Train model (runs only if model doesn't exist)
def train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    
    # Extract zip file
    data_path = extract_zip(ZIP_PATH)
    if data_path is None:
        return None, None
    
    # Placeholder for RAVDESS dataset processing
    features = []
    labels = []
    
    # Emotion mapping (RAVDESS labels)
    emotion_dict = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    try:
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    # Extract emotion from filename (RAVDESS format: modality-vocal_channel-emotion-...)
                    parts = file.split('-')
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        if emotion_code in emotion_dict:
                            feature = extract_features(os.path.join(root, file))
                            if feature is not None:
                                features.append(feature)
                                labels.append(emotion_dict[emotion_code])
        
        if not features:
            raise Exception("No features extracted. Check dataset path and file format.")
        
        # Prepare data
        X = np.array(features)
        y = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
        # Print accuracy
        accuracy = model.score(X_test_scaled, y_test)
        st.write(f"Model trained with accuracy: {accuracy:.2f}")
        
        return model, scaler
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

# Record audio
def record_audio(filename='temp.wav', duration=DURATION, sample_rate=SAMPLE_RATE):
    st.write("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    wavfile.write(filename, sample_rate, recording)
    st.write("Recording finished")
    return filename

# Predict emotion
def predict_emotion(audio_path, model, scaler):
    features = extract_features(audio_path)
    if features is not None:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        return prediction, probabilities
    return None, None

# Save session data
def save_session_data(timestamp, emotion, probabilities):
    data = {
        'Timestamp': [timestamp],
        'Emotion': [emotion],
        'Probabilities': [str(probabilities)]
    }
    df = pd.DataFrame(data)
    if os.path.exists(SESSION_DATA_PATH):
        df.to_csv(SESSION_DATA_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(SESSION_DATA_PATH, index=False)

# Plot emotional trend
def plot_emotional_trend():
    if os.path.exists(SESSION_DATA_PATH):
        df = pd.read_csv(SESSION_DATA_PATH)
        if not df.empty:
            fig, ax = plt.subplots()
            emotions = df['Emotion'].value_counts()
            emotions.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
            ax.set_title('Emotional Trend')
            ax.set_xlabel('Emotion')
            ax.set_ylabel('Count')
            st.pyplot(fig)

# Streamlit UI
def main():
    st.title("Emotion Detection from Voice")
    st.write("This app detects emotions from voice using the RAVDESS dataset and a Random Forest classifier.")
    
    # Train or load model
    with st.spinner("Loading/Training model..."):
        model, scaler = train_model()  # Fixed typo here
    
    if model is None or scaler is None:
        st.error("Failed to load or train model. Please check zip file and dataset format.")
        return
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    record_button = st.sidebar.button("Record Audio")
    
    # Session state for predictions
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    
    # Record and predict
    if record_button:
        audio_file = record_audio()
        if audio_file:
            prediction, probabilities = predict_emotion(audio_file, model, scaler)
            if prediction:
                st.success(f"Detected Emotion: {prediction}")
                # Display probability distribution
                prob_df = pd.DataFrame({
                    'Emotion': model.classes_,
                    'Probability': probabilities
                })
                st.bar_chart(prob_df.set_index('Emotion'))
                
                # Save session data
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                save_session_data(timestamp, prediction, probabilities)
                st.session_state.predictions.append((timestamp, prediction))
                
                # Clean up
                os.remove(audio_file)
    
    # Display emotional trend
    st.header("Emotional Trend")
    plot_emotional_trend()
    
    # Display session history
    st.header("Session History")
    if st.session_state.predictions:
        history_df = pd.DataFrame(st.session_state.predictions, columns=['Timestamp', 'Emotion'])
        st.table(history_df)
    else:
        st.write("No predictions yet. Record audio to start.")

if __name__ == "__main__":
    main()