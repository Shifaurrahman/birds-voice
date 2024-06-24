import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import os
import tempfile
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = tf.keras.models.load_model('bird_sound_classification_model2.h5')

# Define label encoder
classes = ['Brown_Tinamou', 'Cinereous_Tinamou', 'Great_Tinamou']
label_encoder = LabelEncoder()
label_encoder.fit(classes)

# Feature extraction function
def extract_features(file_path, max_length=216):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Pad or truncate MFCCs
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    
    # Pad or truncate Mel spectrogram
    if mel_db.shape[1] < max_length:
        pad_width = max_length - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_length]
    
    return mfccs, mel_db

# Prediction function
def predict_bird_species(file_path):
    mfccs, _ = extract_features(file_path)
    mfccs = mfccs[..., np.newaxis]
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    predictions = model.predict(mfccs)
    predicted_label = np.argmax(predictions, axis=1)
    return label_encoder.inverse_transform(predicted_label)[0]

# Streamlit app
st.title("Bird Sound Classification")
st.write("Upload an audio file to classify the bird species.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3"])

if uploaded_file is not None:
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(file_path, format='audio/mp3')
        
        # Make prediction
        prediction = predict_bird_species(file_path)
        st.write(f"Predicted Bird Species: **{prediction}**")

