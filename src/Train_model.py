from functions import extract_features, load_data
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional

script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = 'PATH_TO_THE_GTZAN_DATASET'  # Path to the GTZAN dataset directory
X, y = load_data(data_path)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
np.save(os.path.join(script_dir,"..","Models", 'classes.npy'), label_encoder.classes_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Reshape input data for LSTM
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Define the model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1))),
    Dropout(0.3),
    BatchNormalization(),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save(os.path.join(script_dir,"..","Models", 'music_genre_rnn_classifier.h5'))
model.save(os.path.join(script_dir,"..","Models", 'music_genre_rnn_classifier.keras'))