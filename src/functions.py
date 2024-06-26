import numpy as np
import librosa

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_path, sr=None)
    if chroma:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    return result

def predict_genre(file_path, model, label_encoder):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)  # Reshape for LSTM input
    prediction = model.predict(features)
    genre_label = np.argmax(prediction)
    return label_encoder.inverse_transform([genre_label])[0]