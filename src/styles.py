import tensorflow as tf
import streamlit as st
import os
import librosa
import numpy as np
from functions import extract_features, predict_genre
import matplotlib.pyplot as plt

def versions():
    print(tf.__version__)
    import keras
    print(keras.__version__)


def style1(model, label_encoder):
    # Streamlit interface
    st.title("Music Genre Classification")

    uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Ensure the temporary directory exists
        os.makedirs("../temp", exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("../temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded file name
        st.write(f"Uploaded file: {uploaded_file.name}")
        
        # Predict genre
        predicted_genre = predict_genre(file_path=temp_file_path, model=model, label_encoder=label_encoder)
        
        # Display the predicted genre
        st.write(f"Predicted genre: {predicted_genre}")

        # Optional: Play the audio file
        st.audio(uploaded_file, format='audio/mp3')

def style2(model, label_encoder):
    st.title("🎵 Music Genre Classification")
    st.markdown("Upload an audio file and let our model predict its genre. 🎶")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Ensure the temporary directory exists
        os.makedirs("../temp", exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("../temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded file name
        st.write(f"Uploaded file: **{uploaded_file.name}**")
        
        # Predict genre
        predicted_genre = predict_genre(file_path=temp_file_path, model=model, label_encoder=label_encoder)
        
        # Display the predicted genre
        st.markdown(f"### Predicted Genre: **{predicted_genre}**")
        
        # Optional: Play the audio file
        st.audio(uploaded_file, format='audio/mp3')

        # Load the audio file for feature extraction and visualization
        y, sr = librosa.load(temp_file_path)
        
        # Waveform plot
        st.markdown("#### Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        st.pyplot(fig)
        
        # Spectrogram plot
        st.markdown("#### Spectrogram")
        fig, ax = plt.subplots()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        st.pyplot(fig)
        
        # MFCC plot
        st.markdown("#### MFCC (Mel-frequency cepstral coefficients)")
        fig, ax = plt.subplots()
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
        plt.colorbar()
        plt.title('MFCC')
        st.pyplot(fig)

        # Tempo and beat tracking
        st.markdown("#### Tempo and Beat Tracking")
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        st.write(f"Estimated tempo: **{tempo:.2f} BPM**")

        fig, ax = plt.subplots()
        times = librosa.frames_to_time(beats, sr=sr)
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.vlines(times, -1, 1, color='r', alpha=0.75, linestyle='--', label='Beats')
        ax.legend()
        plt.title('Tempo and Beat Tracking')
        st.pyplot(fig)
    else:
        st.info("Please upload an audio file to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("Developed by [Rishi Shah](https://my-website.com) | © 2024")

def style3(model, label_encoder):
    st.title("🎵 Music Genre Classification")
    st.markdown("Upload an audio file and let our model predict its genre. 🎶")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Ensure the temporary directory exists
        os.makedirs("../temp", exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("../temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded file name
        st.write(f"Uploaded file: **{uploaded_file.name}**")
        
        # Predict genre
        predicted_genre = predict_genre(file_path=temp_file_path, model=model, label_encoder=label_encoder)
        
        # Display the predicted genre
        st.markdown(f"### Predicted Genre: **{predicted_genre}**")
        
        # Optional: Play the audio file
        st.audio(uploaded_file, format='audio/mp3')

        # Load the audio file for feature extraction and visualization
        y, sr = librosa.load(temp_file_path)
        
        # Waveform plot
        st.markdown("#### Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        st.pyplot(fig)
        
        # Spectrogram plot
        st.markdown("#### Spectrogram")
        fig, ax = plt.subplots()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.title('Spectrogram')
        st.pyplot(fig)
        
        # MFCC plot
        st.markdown("#### MFCC (Mel-frequency cepstral coefficients)")
        fig, ax = plt.subplots()
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
        plt.colorbar(img, ax=ax)
        plt.title('MFCC')
        st.pyplot(fig)

        # Tempo and beat tracking
        st.markdown("#### Tempo and Beat Tracking")
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        st.write(f"Estimated tempo: **{tempo:.2f} BPM**")

        fig, ax = plt.subplots()
        times = librosa.frames_to_time(beats, sr=sr)
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.vlines(times, -1, 1, color='r', alpha=0.75, linestyle='--', label='Beats')
        ax.legend()
        plt.title('Tempo and Beat Tracking')
        st.pyplot(fig)
    else:
        st.info("Please upload an audio file to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("Developed by [Rishi Shah](https://my-website.com) | © 2024")

def style4(model, label_encoder):
    st.title("🎵 Music Genre Classification")
    st.markdown("Upload an audio file and let our model predict its genre. 🎶")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Ensure the temporary directory exists
        os.makedirs("../temp", exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("../temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded file name
        st.write(f"Uploaded file: **{uploaded_file.name}**")
        
        # Predict genre
        predicted_genre = predict_genre(file_path=temp_file_path, model=model, label_encoder=label_encoder)
        
        # Display the predicted genre
        st.markdown(f"### Predicted Genre: **{predicted_genre}**")
        
        # Optional: Play the audio file
        st.audio(uploaded_file, format='audio/mp3')

        # Load the audio file for feature extraction and visualization
        y, sr = librosa.load(temp_file_path)
        
        # Waveform plot
        st.markdown("#### Waveform")
        fig, ax = plt.subplots(facecolor='none', edgecolor='none')
        ax.set_facecolor('none')
        librosa.display.waveshow(y, sr=sr, ax=ax)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        st.pyplot(fig)
        
        # Spectrogram plot
        st.markdown("#### Spectrogram")
        fig, ax = plt.subplots(facecolor='none', edgecolor='none')
        ax.set_facecolor('none')
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.title('Spectrogram')
        st.pyplot(fig)
        
        # MFCC plot
        st.markdown("#### MFCC (Mel-frequency cepstral coefficients)")
        fig, ax = plt.subplots(facecolor='none', edgecolor='none')
        ax.set_facecolor('none')
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
        plt.colorbar(img, ax=ax)
        plt.title('MFCC')
        st.pyplot(fig)

        # Tempo and beat tracking
        st.markdown("#### Tempo and Beat Tracking")
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)  # Ensure tempo is a scalar
        st.write(f"Estimated tempo: **{tempo:.2f} BPM**")

        fig, ax = plt.subplots(facecolor='none', edgecolor='none')
        ax.set_facecolor('none')
        times = librosa.frames_to_time(beats, sr=sr)
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.vlines(times, -1, 1, color='r', alpha=0.75, linestyle='--', label='Beats')
        ax.legend()
        plt.title('Tempo and Beat Tracking')
        st.pyplot(fig)
    else:
        st.info("Please upload an audio file to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("Developed by [Rishi Shah](https://my-website.com) | © 2024")

def style5(model, label_encoder):
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .title {
            font-family: 'Trebuchet MS', sans-serif;
            color: #4B8BBE;
        }
        .subtitle {
            font-family: 'Arial', sans-serif;
            color: #306998;
        }
        .prediction {
            font-family: 'Courier New', monospace;
            color: #FF4500;
        }
        .footer {
            font-family: 'Verdana', sans-serif;
            color: #666666;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App title and description
    st.markdown("<h1 class='title'>🎵 Music Genre Classification</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subtitle'>Upload an audio file and let our model predict its genre. 🎶</h3>", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Ensure the temporary directory exists
        os.makedirs("../temp", exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("../temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded file name
        st.write(f"Uploaded file: **{uploaded_file.name}**")
        
        # Predict genre
        predicted_genre = predict_genre(file_path=temp_file_path, model=model, label_encoder=label_encoder)
        
        # Display the predicted genre
        st.markdown(f"<h3 class='prediction'>Predicted Genre: {predicted_genre}</h3>", unsafe_allow_html=True)
        
        # Optional: Play the audio file
        st.audio(uploaded_file, format='audio/mp3')

        # Load the audio file for feature extraction and visualization
        y, sr = librosa.load(temp_file_path)

        col1, col2 = st.columns(2)
        with col1:
            # Waveform plot
            st.markdown("<h4 class='subtitle'>Waveform</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(facecolor='none', edgecolor='none')
            ax.set_facecolor('none')
            librosa.display.waveshow(y, sr=sr, ax=ax)
            plt.title('Waveform')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            st.pyplot(fig)
        
        with col2:
            # Spectrogram plot
            st.markdown("<h4 class='subtitle'>Spectrogram</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(facecolor='none', edgecolor='none')
            ax.set_facecolor('none')
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.title('Spectrogram')
            st.pyplot(fig)
        
        col3, col4 = st.columns(2)
        with col3:
            # MFCC plot
            st.markdown("<h4 class='subtitle'>MFCC (Mel-frequency cepstral coefficients)</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(facecolor='none', edgecolor='none')
            ax.set_facecolor('none')
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
            plt.colorbar(img, ax=ax)
            plt.title('MFCC')
            st.pyplot(fig)

        with col4:
            # Tempo and beat tracking
            st.markdown("<h4 class='subtitle'>Tempo and Beat Tracking</h4>", unsafe_allow_html=True)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)  # Ensure tempo is a scalar
            st.write(f"Estimated tempo: **{tempo:.2f} BPM**")

            fig, ax = plt.subplots(facecolor='none', edgecolor='none')
            ax.set_facecolor('none')
            times = librosa.frames_to_time(beats, sr=sr)
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.vlines(times, -1, 1, color='r', alpha=0.75, linestyle='--', label='Beats')
            ax.legend()
            plt.title('Tempo and Beat Tracking')
            st.pyplot(fig)
    else:
        st.info("Please upload an audio file to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("<p class='footer'>Developed by <a href='https://my-website.com'>Rishi Shah</a> | © 2024</p>", unsafe_allow_html=True)

def mainstyle(model, label_encoder):
    st.set_page_config(page_icon="🎵", page_title="Music Genre Detection", layout="wide")
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .title {
            font-family: 'Trebuchet MS', sans-serif;
            color: #4B8BBE;
        }
        .subtitle {
            font-family: 'Arial', sans-serif;
            color: #306998;
        }
        .prediction {
            font-family: 'Courier New', monospace;
            color: #FF4500;
        }
        .footer {
            font-family: 'Verdana', sans-serif;
            color: #666666;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App title and description
    st.markdown("<h1 class='title'>🎵 Music Genre Classification</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subtitle'>Upload an audio file and let our model predict its genre. 🎶</h3>", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Ensure the temporary directory exists
        os.makedirs("../temp", exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("../temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded file name
        st.write(f"Uploaded file: **{uploaded_file.name}**")
        
        # Predict genre
        predicted_genre = predict_genre(file_path=temp_file_path, model=model, label_encoder=label_encoder)
        
        # Display the predicted genre
        st.markdown(f"<h3 class='prediction'>Predicted Genre: {predicted_genre}</h3>", unsafe_allow_html=True)
        
        # Optional: Play the audio file
        st.audio(uploaded_file, format='audio/mp3')

        # Load the audio file for feature extraction and visualization
        y, sr = librosa.load(temp_file_path)
        
        # col1, col2 = st.columns(2)
        # with col1:
        #     # Waveform plot
        #     st.markdown("<h4 class='subtitle'>Waveform</h4>", unsafe_allow_html=True)
        #     fig, ax = plt.subplots(facecolor='none', edgecolor='none')
        #     ax.set_facecolor('none')
        #     ax.tick_params(colors='white', which='both')  # Change the color of ticks
        #     ax.xaxis.label.set_color('white')  # Change the color of the x axis label
        #     ax.yaxis.label.set_color('white')  # Change the color of the y axis label
        #     librosa.display.waveshow(y, sr=sr, ax=ax)
        #     plt.title('Waveform', color='white')
        #     plt.xlabel('Time (s)', color='white')
        #     plt.ylabel('Amplitude', color='white')
        #     st.pyplot(fig)
        
        # with col2:
        #     # Spectrogram plot
        #     st.markdown("<h4 class='subtitle'>Spectrogram</h4>", unsafe_allow_html=True)
        #     fig, ax = plt.subplots(facecolor='none', edgecolor='none')
        #     ax.set_facecolor('none')
        #     ax.tick_params(colors='white', which='both')  # Change the color of ticks
        #     ax.xaxis.label.set_color('white')  # Change the color of the x axis label
        #     D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        #     img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        #     plt.colorbar(img, ax=ax, format='%+2.0f dB')
        #     plt.title('Spectrogram', color='white')
        #     st.pyplot(fig)
        
        # col3, col4 = st.columns(2)
        # with col3:
        #     # MFCC plot
        #     st.markdown("<h4 class='subtitle'>MFCC (Mel-frequency cepstral coefficients)</h4>", unsafe_allow_html=True)
        #     fig, ax = plt.subplots(facecolor='none', edgecolor='none')
        #     ax.set_facecolor('none')
        #     ax.tick_params(colors='white', which='both')  # Change the color of ticks
        #     ax.xaxis.label.set_color('white')  # Change the color of the x axis label
        #     ax.yaxis.label.set_color('white')  # Change the color of the y axis label
        #     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        #     img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
        #     plt.colorbar(img, ax=ax)
        #     plt.title('MFCC', color='white')
        #     st.pyplot(fig)

        # with col4:
        #     # Tempo and beat tracking
        #     st.markdown("<h4 class='subtitle'>Tempo and Beat Tracking</h4>", unsafe_allow_html=True)
        #     tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        #     tempo = float(tempo)  # Ensure tempo is a scalar
        #     st.write(f"Estimated tempo: **{tempo:.2f} BPM**")

        #     fig, ax = plt.subplots(facecolor='none', edgecolor='none')
        #     ax.set_facecolor('none')
        #     ax.tick_params(colors='white', which='both')  # Change the color of ticks
        #     ax.xaxis.label.set_color('white')  # Change the color of the x axis label
        #     ax.yaxis.label.set_color('white')  # Change the color of the y axis label
        #     times = librosa.frames_to_time(beats, sr=sr)
        #     librosa.display.waveshow(y, sr=sr, ax=ax)
        #     ax.vlines(times, -1, 1, color='r', alpha=0.75, linestyle='--', label='Beats')
        #     ax.legend()
        #     plt.title('Tempo and Beat Tracking', color='white')
        #     st.pyplot(fig)
        col1, col2 = st.columns(2)
        with col1:
            # Waveform plot
            st.markdown("<h4 class='subtitle'>Waveform</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(facecolor='none', edgecolor='none')
            ax.set_facecolor('none')
            ax.tick_params(colors='white', which='both')  # Change the color of ticks
            ax.xaxis.label.set_color('white')  # Change the color of the x axis label
            ax.yaxis.label.set_color('white')  # Change the color of the y axis label
            librosa.display.waveshow(y, sr=sr, ax=ax)
            plt.title('Waveform', color='white')
            plt.xlabel('Time (s)', color='white')
            plt.ylabel('Amplitude', color='white')
            st.pyplot(fig)

        with col2:
            # Spectrogram plot
            st.markdown("<h4 class='subtitle'>Spectrogram</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(facecolor='none', edgecolor='none')
            ax.set_facecolor('none')
            ax.tick_params(colors='white', which='both')  # Change the color of ticks
            ax.xaxis.label.set_color('white')  # Change the color of the x axis label
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
            cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
            cbar.ax.tick_params(colors='white')  # Change the color of the color bar ticks
            cbar.ax.yaxis.label.set_color('white')  # Change the color of the color bar label
            plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')  # Change the color of the color bar tick labels
            plt.title('Spectrogram', color='white')
            st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            # MFCC plot
            st.markdown("<h4 class='subtitle'>MFCC (Mel-frequency cepstral coefficients)</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(facecolor='none', edgecolor='none')
            ax.set_facecolor('none')
            ax.tick_params(colors='white', which='both')  # Change the color of ticks
            ax.xaxis.label.set_color('white')  # Change the color of the x axis label
            ax.yaxis.label.set_color('white')  # Change the color of the y axis label
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
            cbar = plt.colorbar(img, ax=ax)
            cbar.ax.tick_params(colors='white')  # Change the color of the color bar ticks
            cbar.ax.yaxis.label.set_color('white')  # Change the color of the color bar label
            plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')  # Change the color of the color bar tick labels
            plt.title('MFCC', color='white')
            st.pyplot(fig)

        with col4:
            # Tempo and beat tracking
            st.markdown("<h4 class='subtitle'>Tempo and Beat Tracking</h4>", unsafe_allow_html=True)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)  # Ensure tempo is a scalar
            st.write(f"Estimated tempo: **{tempo:.2f} BPM**")

            fig, ax = plt.subplots(facecolor='none', edgecolor='none')
            ax.set_facecolor('none')
            ax.tick_params(colors='white', which='both')  # Change the color of ticks
            ax.xaxis.label.set_color('white')  # Change the color of the x axis label
            ax.yaxis.label.set_color('white')  # Change the color of the y axis label
            times = librosa.frames_to_time(beats, sr=sr)
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.vlines(times, -1, 1, color='r', alpha=0.75, linestyle='--', label='Beats')
            ax.legend()
            plt.title('Tempo and Beat Tracking', color='white')
            st.pyplot(fig)
    else:
        st.info("Please upload an audio file to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("<p class='footer'>Developed by <a href='https://my-website.com'>Rishi Shah</a> | © 2024</p>", unsafe_allow_html=True)