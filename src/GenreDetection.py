import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from styles import style1, style2, style3, style4, style5, mainstyle
import os
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
warnings.filterwarnings("ignore")

def load_model_and_load_label_encoder() -> None:
    # Determine the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the paths to the model and classes files
    model_path = os.path.join(script_dir, '..', 'Models', 'music_genre_rnn_classifier.keras')
    classes_path = os.path.join(script_dir, '..', 'Models', 'classes.npy')
    
    # Load the model
    model = load_model(model_path)
    
    # Load the label encoder classes
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(classes_path)
    
    return model, label_encoder

if __name__ == "__main__":
    model, label_encoder = load_model_and_load_label_encoder()
    mainstyle(model=model, label_encoder=label_encoder)