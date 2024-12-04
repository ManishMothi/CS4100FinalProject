import opensmile
import pandas as pd
import os

base_dir = "fnn"

dataset_dir = os.path.join(base_dir, "CREMA-D")  
output_csv = os.path.join(base_dir, "cremad_feature_extraction.csv") 


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

# emotion mapping using CREMA-D filename convention
cremad_emotion_mapping = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "FEA": "fear",
    "DIS": "disgust",
    "NEU": "neutral"
}

# sentence mapping using CREMA-D filename convention
cremad_sentence_mapping = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes"
}

def map_cremad_audio_features(path):
    audio_feature=path.split("/")[-1].split("_")
    emotion=cremad_emotion_mapping.get(audio_feature[2])
    text=cremad_sentence_mapping.get(audio_feature[1])
    return path,emotion,text


cremad_features = []
# count = 0

for root, _, files in os.walk(dataset_dir):
    for filename in files:
        if filename.endswith(".wav"):
            parts = filename.split("_")
            speaker_id = parts[0]  # Speaker ID
            sentence_code = parts[1]  # Sentence code
            emotion_code = parts[2]  # Emotion code

            # mapping emotion and sentence labels
            emotion_label = cremad_emotion_mapping.get(emotion_code, "unknown")
            sentence_text = cremad_sentence_mapping.get(sentence_code, "unknown sentence")

            file_path = os.path.join(root, filename)

            # extract features using OpenSMILE
            features = smile.process_file(file_path)

            # adding metadata (emotion, sentence, filename, speaker ID) to the features
            features["emotion"] = emotion_label
            features["sentence"] = sentence_text
            features["filename"] = filename
            features["speaker_id"] = speaker_id

            cremad_features.append(features)
            # print(count)
            # count += 1

features_df = pd.concat(cremad_features, ignore_index=True)
features_df.to_csv(output_csv, index=False)

print(f"Feature extraction completed and saved to {output_csv}")
print(features_df.head())
