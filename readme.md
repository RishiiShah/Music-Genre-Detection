## Music Genre Detection/Classification

This project demonstrates an advanced approach to music genre detection and classification using TensorFlow, trained on the GTZAN dataset—a widely recognized benchmark in music genre recognition. The methodology involves feature extraction using Librosa for MFCC (Mel Frequency Cepstral Coefficients) and chroma representations, pivotal for capturing timbral and harmonic attributes essential to genre classification.

### Features and Libraries Used:
- **TensorFlow**: Employed for constructing and training the classification model, leveraging deep learning capabilities.
- **Librosa**: Facilitated music and audio analysis, enabling extraction of crucial audio features.
- **NumPy**: Used for efficient numerical computations and data manipulation.
- **Scikit-learn**: Applied for label encoding categorical variables, ensuring compatibility with TensorFlow.
- **Streamlit**: Utilized to develop interactive web applications for visualizing model predictions.
- **Matplotlib**: Employed for generating visual representations of audio features and model performance metrics.

### Dataset
The GTZAN dataset is utilized, consisting of 1000 audio tracks each 30 seconds long. These tracks are further divided into 10 genres, each containing 100 tracks. The genres included are:

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

This diverse dataset provides a comprehensive basis for training and evaluating the model's ability to classify different musical genres accurately.

### Methodology
1. **Feature Extraction**:
   - **MFCC**: Captures spectral characteristics crucial for discerning different musical timbres.
   - **Chroma**: Represents harmonic content, aiding in genre differentiation.

2. **Model Development**:
   - **Bidirectional RNN**: Implemented to enhance model accuracy by capturing both past and future context in the sequence data. This architecture contributed to achieving approximately 80% accuracy on the validation set.
   - TensorFlow architecture optimized with bidirectional recurrent layers for sequential data processing.

3. **Training and Evaluation**:
   - The dataset was split into training (80%) and validation (20%) subsets for model training and performance evaluation.
   - Metrics such as accuracy, precision, recall, and F1-score computed to gauge classification performance across diverse genres.

4. **Visualization and Deployment**:
   - **Streamlit** integrated with **Matplotlib** for real-time visualization of model predictions and classification results.
   - Users can interactively input audio samples or select predefined examples to observe genre predictions based on the trained model.

### Installation and Usage
To clone and run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Project**:
   ```bash
   streamlit run app.py
   ```
   This command starts the Streamlit server and opens the application in your default web browser.

### Conclusion
This project showcases an effective use of TensorFlow and Librosa for music genre detection and classification tasks, leveraging advanced techniques like bidirectional RNNs to achieve high accuracy. The integration with Streamlit and Matplotlib enhances usability, providing intuitive insights into the model's genre predictions. It contributes to advancing applications of machine learning in music analysis and underscores the potential of deep learning in audio signal processing.

For detailed implementation and usage instructions, refer to the documentation and source code available in this repository.