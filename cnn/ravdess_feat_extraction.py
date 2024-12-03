import opensmile
import pandas as pd
import os

# paths to the dataset and output csv
dataset_dir = "C:\\repos\\CS4100FinalProject\\cnn\\RAVDESS"  # path where RAVDESS dataset is located
output_csv = "cnn\\ravdess_feature_extraction.csv"   


# initializing the OpenSMILE feature extractor
# using eGeMAPSv02 feature set and functionals feature level, useful for emotion recognition
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

print("initialized feature extraction library")

# emotion map using RAVDESS label naming conventions
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

audio_features = []


# iterating through all audio files in the RAVDESS folder
for root, _, files in os.walk(dataset_dir):
    for filename in files:
        if filename.endswith(".wav"):

            # extracting emotion label from the filename
            parts = filename.split("-")
            emotion_code = parts[2]
            emotion_label = emotion_map.get(emotion_code, "unknown")

            file_path = os.path.join(root, filename)

            features = smile.process_file(file_path) # extracting features with opensmile

            # adding emotion label and filename
            features["emotion"] = emotion_label
            features["filename"] = filename

            audio_features.append(features)


features_df = pd.concat(audio_features, ignore_index=True)
features_df.to_csv(output_csv, index=False)
print(f"Feature extraction of RAVDESS Dataset complete and saved to {output_csv}")


df = pd.read_csv("cnn\\ravdess_feature_extraction.csv")

print(df.head())
