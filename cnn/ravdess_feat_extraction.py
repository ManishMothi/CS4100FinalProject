import opensmile
import pandas as pd
import os


base_dir = "cnn"

dataset_dir = os.path.join(base_dir, "RAVDESS")  # Path where RAVDESS dataset is located
output_csv = os.path.join(base_dir, "ravdess_feature_extraction.csv")  # Path for the output CSV

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



# Combine all extracted features into a DataFrame
if audio_features: 
    features_df = pd.concat(audio_features, ignore_index=True)
    features_df.to_csv(output_csv, index=False)
    print(f"Feature extraction of RAVDESS dataset complete and saved to {output_csv}")
else:
    print("No audio features extracted. Please check the dataset directory.")

# Load the saved CSV to confirm the data
if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
    print(df.head())
else:
    print(f"CSV file {output_csv} not found. Extraction may have failed.")