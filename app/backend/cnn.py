import numpy as np
import opensmile
import os
from pathlib import Path
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def cnn_workflow(filename):
    '''
    CNN workflow which takes mp3 file, converts it to vocal features via openSmile,
    and is fed to trained CNN which predicts emotion scores. 

    Returns an emotion vector consisting of: 
    happy, angry, disgust, neutral, calm, sad, surprised, fearful       

    Sum of values add up to 1 and range from 0 to 1. 
    '''

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,  
        feature_level=opensmile.FeatureLevel.Functionals 
    )

    current_dir = Path(__file__).parent
    audio_file = current_dir / filename
    audio_file = str(audio_file)

    features = smile.process_file(audio_file)

    current_file_path = os.path.abspath(__file__)
    file_path = os.path.join(os.path.dirname(current_file_path), '..', '..', 'cnn', 'audio_cnn.pkl')
    file_path = os.path.abspath(file_path)

    with open(file_path, 'rb') as file:
        model = pickle.load(file)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_reshaped = np.expand_dims(X_scaled, axis=-1)

    emotion_predictions = model.predict(X_reshaped)

    return emotion_predictions