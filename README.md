# Emotion-Detection-from-Voice

This project detects emotions in human voice using audio analysis and machine learning. It uses the **RAVDESS dataset**, extracts MFCC features, and classifies audio into one of 8 emotions using a Random Forest Classifier.

---

## 📁 Dataset: RAVDESS

- 🎧 **1440 audio samples**
- 🧑‍🎤 24 actors (12 male, 12 female)
- 🗣️ Emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- 📝 Filename structure provides metadata (emotion, intensity, gender, etc.)

📥 [Download from Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

---

## 🔍 Step-by-Step Workflow

---

### ✅ Step 1: Parsing Metadata

We parsed all filenames to extract:

- Emotion  
- Intensity  
- Gender  
- Actor ID  

### Parsing example filename: 03-01-06-01-02-01-12.wav

## 📊 Step 2: Exploratory Data Analysis (EDA)
📌 Emotion Distribution

![image](https://github.com/user-attachments/assets/76025481-5024-4aa3-b96c-8cb5765d24dd)

📌 Gender-wise Emotion Distribution

![image](https://github.com/user-attachments/assets/81333714-a130-4ba5-989e-c28da19aa5ca)

![image](https://github.com/user-attachments/assets/2185161f-9150-4942-a6be-90c71a58ed96)

![image](https://github.com/user-attachments/assets/58a883d9-baa5-4c76-af2d-9fd95aecacea)



## 🎛️ Step 3: Feature Extraction
Extracted 40 MFCC features per file

Stored as .npy for fast access

📂 Saved files:

X_features.npy

y_labels.npy

genders.npy

## 🤖 Step 4: Model Training
🧠 Algorithm Used: RandomForestClassifier
Training Set Size: 80%
Test Set Size: 20%

📈 Model Metrics:

yaml
Copy
Edit
✅ Accuracy: 0.60
✅ F1 Score: 0.60
📌 Classification Report

markdown
Copy
Edit
               precision    recall  f1-score   support

       angry       0.75      0.63      0.69        38
        calm       0.67      0.87      0.76        38
     disgust       0.54      0.71      0.61        38
     fearful       0.58      0.72      0.64        39
       happy       0.61      0.44      0.51        39
     neutral       0.50      0.32      0.39        19
         sad       0.51      0.47      0.49        38
   surprised       0.62      0.54      0.58        39

    accuracy                           0.60       288
   macro avg       0.60      0.59      0.58       288
weighted avg       0.60      0.60      0.60       288
📷

📦 Saved Files
emotion_model.pkl ✅

label_encoder.pkl ✅

## 🌐 Streamlit Web App
🔧 Features
Feature	Description
🎙️ Upload .wav file	Accepts voice audio files
📈 Waveform & Spectrogram	Visualizes audio in real-time
📊 Prediction Chart	Emotion prediction bar chart
🧠 Output Display	Predicted label shown clearly

![image](https://github.com/user-attachments/assets/6656193c-221a-46e5-a284-d46458a38974)

![image](https://github.com/user-attachments/assets/56973757-9d0c-4ae3-848f-9258e59de601)

![image](https://github.com/user-attachments/assets/19abd635-d9b9-4855-a31a-6056e8c5b4c7)

![image](https://github.com/user-attachments/assets/c8e56c6d-461b-4407-b9ce-9215e592dc3a)

## ▶️ How to Run
bash
Copy
Edit
streamlit run "D:\ML PROJECTS\Emotion Detection from Voice\admission_app.py"
Make sure these files exist in the same directory:

emotion_model.pkl

label_encoder.pkl

admission_app.py

Audio sample for testing

## 🧰 Tech Stack
Python 3.x

Scikit-learn

Librosa

Pandas / NumPy

Matplotlib / Seaborn

Streamlit

## 📁 Folder Structure
csharp
Copy
Edit
Emotion Detection from Voice/
│
├── archive/                 # RAVDESS audio
├── emotion_model.pkl        # Trained ML model
├── label_encoder.pkl        # Label encoder
├── X_features.npy           # Feature matrix
├── y_labels.npy             # Labels
├── genders.npy              # Gender array
├── admission_app.py         # Streamlit frontend
├── ravdess_metadata.csv     # Metadata file
└── README.md                # This file

## 🚀 Future Work

🎤 Real-time microphone inference

🧠 CNN-based deep learning model

☁️ Deploy to Hugging Face Spaces / Streamlit Cloud

🙌 Developed By
V. Yuvan Krishnan
SRM Institute of Science and Technology
