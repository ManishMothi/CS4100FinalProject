# Emotion-to-MBTI Prediction: Fusion Model and App

This repository contains two main components:

1. **Model Training**: Scripts for training the fusion model (Audio CNN + FNN).
2. **Web Application**: A Next.js frontend and Flask backend to interact with the trained model.

## Getting Started

### Prerequisites

#### Python Environments

You need two separate Python virtual environments:

- **Training Environment**: For training models.
- **Backend Environment**: For running the Flask backend. (make sure you are in the backend directory when creating)

Create the environments as follows:

##### Training Environment:

    python3 -m venv training_env
    source training_env/bin/activate  # On Windows: training_env\Scripts\activate
    pip install -r requirements.txt

##### Backend Environment:

    cd app/backend
    python3 -m venv backend_env
    source backend_env/bin/activate  # On Windows: backend_env\Scripts\activate
    pip install -r requirements.txt

##### Frontend Setup:

    cd app/frontend
    npm install

---

## Part 1: Fusion Model Creation & Training

### Step 1: Audio Convolutional Neural Network (CNN)

#### Feature Extraction: Extract audio features from the RAVDESS emotion audio dataset (takes a ~5 minutes to complete)

    source training_env/bin/activate  # Activate the training environment
    python cnn/ravdess_feat_extraction.py

### Step 2: Fusion Feed-Forward Neural Network (FNN)

#### Feature Extraction: Extract features from the CREMA-D dataset (5-10 minutes to complete)

    python fnn/01-cremad_feat_extraction.py

#### Train the FNN (~10 minutes to complete):

    python fnn/02-cremad_FNN.py
    deactivate

---

## Part 2: Web Application

### Backend Setup (Flask)

    cd app/backend
    source backend_env/bin/activate  # Activate the backend environment
    flask run --port=5000  # Start the Flask backend server

### Frontend Setup (Next.js)

    cd ../frontend
    npm run dev  # Start the Next.js frontend server

### Access the Application

Open your browser and go to:

    http://localhost:3000/

---

## Usage Instructions

1. You will be automatically redirected to `http://localhost:3000/AudioRecorder` within 20 seconds.
2. Press the "Record" button and talk about your day for 20 seconds.
3. After recording, you will be redirected to the MBTI Prediction Page in approximately 30 seconds.

---

## High Level File Structure

    .
    ├── app/
    │   ├── frontend/             # Next.js frontend
    │   ├── backend/              # Flask backend
    │   └── requirements.txt      # Python dependencies for backend
    ├── cnn/                      # CNN-related scripts
    │   └── ravdess_CNN.py
    │   └── ravdess_feat_extraction.py
    ├── fnn/                      # FNN-related scripts
    │   ├── 01-cremad_feat_extraction.py
    │   └── 02-cremad_FNN.py
    └── requirements.txt          # Python dependencies for training

---

## Notes

- Ensure you activate the correct virtual environment before running scripts or servers.
- The training environment is for running CNN and FNN scripts.
- The backend environment is for running the Flask backend.
- Both the Flask backend and Next.js frontend must run simultaneously for the application to work.
