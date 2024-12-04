import pandas as pd
import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertForSequenceClassification

with open(os.path.join("cnn", "audio_cnn.pkl"), 'rb') as file:
    cnn_model = pickle.load(file)

crema_data = pd.read_csv(os.path.join("fnn", "cremad_feature_extraction.csv"))
ravdess_data = pd.read_csv(os.path.join("cnn", "ravdess_feature_extraction.csv"))

X = crema_data.drop(columns=['filename', 'emotion', 'sentence','speaker_id'])
y = crema_data['emotion']
print(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_reshaped = np.expand_dims(X_scaled, axis = -1)
predictions = cnn_model.predict(X_reshaped)

# for i in range(3):
#     print(predictions[i])

emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise"]

mapped_emotions = []
for prediction in predictions:
    one_hot = np.zeros_like(prediction)
    one_hot[np.argmax(prediction)] = 1

    max_index = np.argmax(one_hot)
    emotion = emotions[max_index]
    mapped_emotions.append(emotion)
for i, emotion in enumerate(mapped_emotions):
    print(f"Prediction {i + 1}: {emotion}")

print(y.value_counts())


X['preds'] = mapped_emotions
print(X['preds'].value_counts())



# passing CREMA-D data through BERT model:

tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")

def personality_detection(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.squeeze().detach().numpy()

    label_names = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    result = {label_names[i]: predictions[i] for i in range(len(label_names))}

    return result,


text_input = "I am feeling excited about the upcoming event."
personality_prediction = personality_detection(text_input)

print(personality_prediction)


# applying personality_detection to all the text within the crema_dataset

crema_data['Big5'] = crema_data['sentence'].apply(personality_detection)
crema_data['ListBig5'] = crema_data['Big5'].apply(lambda d: list(d[0].values()) if isinstance(d, tuple) else [])
crema_data['ListBig5'].head()

predictions_df = pd.DataFrame(predictions)
predictions_df['cnn_predictions'] = predictions_df.apply(lambda row: row.tolist(), axis=1)
crema_data['cnn_predictions'] = predictions_df['cnn_predictions']
crema_data['cnn_predictions'].head()
crema_data['combined_vector'] = crema_data.apply(lambda row: row['ListBig5'] + row['cnn_predictions'], axis=1)

crema_data['combined_vector'].head()

# add back into csv
# crema_data.to_csv('C:\\repos\\CS4100FinalProject\\fnn\\cremad_features_preprocessed.csv', index=False)



# function to assign mbti labels to the OCEAN-Emotion combined vector for CREMA-D dataset
def assign_mbti_label_balanced(vector):
    
    # splitting vector into Big Five and emotion scores
    big_five = vector[:5]
    emotions = vector[5:]
    emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

    # normalizing Big Five traits logarithmically
    big_five = np.log(np.array(big_five) + 1) / np.log(2)  # Normalize to [0, 1]
    extraversion, agreeableness, conscientiousness, neuroticism, openness = big_five

    # dynamic thresholds based on median
    sn_threshold = np.median(big_five)  # Dynamic threshold for S/N
    jp_threshold = np.median(big_five)  # Dynamic threshold for J/P

    # basing the MBTI values from Big Five traits
    ei = 'E' if extraversion > 0.5 else 'I'
    sn = 'N' if openness > sn_threshold else 'S'
    tf = 'F' if agreeableness > 0.5 else 'T'
    jp = 'J' if conscientiousness > jp_threshold else 'P'

    # Determine dominant emotion
    dominant_emotion_index = emotions.index(max(emotions))
    dominant_emotion = emotion_labels[dominant_emotion_index]

    # Adjust MBTI based on the dominant emotion
    if dominant_emotion in ["happy", "surprised"]:
        ei = 'E'
    elif dominant_emotion in ["sad", "calm", "fearful"]:
        ei = 'I'

    if dominant_emotion in ["angry", "disgust"]:
        tf = 'T'
    elif dominant_emotion in ["neutral", "happy"]:
        tf = 'F'

    sn = 'N' if openness > sn_threshold else 'S'
    jp = 'J' if conscientiousness > jp_threshold else 'P'

    return ei + sn + tf + jp


# cremad_features = pd.read_csv('C:\\repos\\CS4100FinalProject\\fnn\\cremad_features_preprocessed.csv')
crema_data['ground_truths_balanced'] = crema_data['combined_vector'].apply(assign_mbti_label_balanced)
print(crema_data['ground_truths_balanced'].value_counts())




# normalizing combined vector 
def logarithmic_normalization(vector):
    big_five = vector[:5]
    emotions = vector[5:]

    big_five = np.log(np.array(big_five) + 1) / np.log(2)
    emotions = np.log(np.array(emotions) + 1) / np.log(2)

    return np.concatenate((big_five, emotions))


crema_data['normalized_combined_vector'] = crema_data['combined_vector'].apply(logarithmic_normalization)
crema_extracted_df = crema_data[['ListBig5', 'cnn_predictions', 'combined_vector', 'ground_truths_balanced', 'normalized_combined_vector']]
crema_extracted_df.shape
crema_extracted_df.to_csv(os.path.join("fnn", "cremad_extracted_df.csv"), index=False)

# crema_extracted_df['ground_truths'].value_counts()
crema_extracted_df['ground_truths_balanced'].value_counts()

crema_model_df = pd.read_csv(os.path.join("fnn", "cremad_extracted_df.csv"))
crema_model_df 

fnn_df = crema_model_df[['normalized_combined_vector', 'ground_truths_balanced']]
fnn_df





def parse_array_string(s):
    s = s.strip('[]').replace('\n', ' ')
    items = s.split()
    float_items = [float(item) for item in items]
    return torch.tensor(float_items, dtype=torch.float32)

fnn_df['normalized_combined_vector'] = fnn_df['normalized_combined_vector'].apply(parse_array_string)
print(fnn_df['normalized_combined_vector'].iloc[0])
print(type(fnn_df['normalized_combined_vector'].iloc[0]))


fnn_df['ground_truths_balanced'] = fnn_df['ground_truths_balanced'].astype('category')
fnn_df['label_codes'] = fnn_df['ground_truths_balanced'].cat.codes
print(fnn_df[['ground_truths_balanced', 'label_codes']].head())





class PersonalityDataset(Dataset):
    def __init__(self, dataframe):
        self.vectors = dataframe['normalized_combined_vector'].tolist()  # List of tensors
        self.labels = torch.tensor(dataframe['label_codes'].tolist(), dtype=torch.long)  # Numeric labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]

dataset = PersonalityDataset(fnn_df)
print(dataset[0])



train_df, test_df = train_test_split(fnn_df, test_size=0.2, random_state=42)
train_dataset = PersonalityDataset(train_df)
test_dataset = PersonalityDataset(test_df)




batch_size = 64

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 8)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x





feedforward_net = FNN()
criterion = nn.CrossEntropyLoss()
optimizer_fnn = torch.optim.Adam(feedforward_net.parameters(), lr=0.001)


loss_values_fnn = []   # loss for each epoch 


# num_epochs_ffn = 25

for epoch in range(25):  # loop over the dataset multiple times
    running_loss_ffn = 0.0

    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Flatten inputs for ffn
        inputs = inputs.flatten(start_dim=1)
        
        # zero the parameter gradients
        optimizer_fnn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_fnn.step()
        running_loss_ffn += loss.item()
    loss_values_fnn.append(running_loss_ffn)    

    print(f"Training loss: {running_loss_ffn}")

print('Finished Training')


torch.save(feedforward_net.state_dict(), os.path.join("fnn", "fnn.pth"))  # Saves model file (upload with submission)

with open(os.path.join("fnn", "loss_values_fnn.pkl"), 'wb') as f:
    pickle.dump(loss_values_fnn, f)




feedforward_net.load_state_dict(torch.load(os.path.join("fnn", "fnn.pth")))

correct_ffn = 0
total_ffn = 0

with torch.no_grad():           # since we're not training, we don't need to calculate the gradients for our outputs
    for data in test_loader:
        inputs, labels = data
        inputs_flattened = inputs.flatten(start_dim=1)

        outputs_ffn = feedforward_net(inputs_flattened)
        correct_ffn += (torch.argmax(outputs_ffn, dim=1) == labels).sum().item()
        total_ffn += labels.size(0)


print('Accuracy for feedforward network: ', correct_ffn/total_ffn)




import matplotlib.pyplot as plt
import pickle



def plot_loss(loss_history, model):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label=f"{model} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for {model}")
    plt.legend()
    plt.savefig('loss_char_' + model + '.png') 
    plt.show()


with open(os.path.join("fnn", "loss_values_fnn.pkl"), 'rb') as f:
    loss_values_fnn = pickle.load(f)


plot_loss(loss_values_fnn, "FNN")
