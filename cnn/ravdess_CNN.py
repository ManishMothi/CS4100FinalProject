import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle


# preprocessing
data = pd.read_csv('cnn\\ravdess_feature_extraction.csv')

X = data.drop(columns=['filename', 'emotion'])
y = data['emotion']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

print(f"training data shape: {X_train.shape}")
print(f"test data shape: {X_test.shape}")

# Audio CNN MOdel architecture
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 1 dimensional convolution
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)


# training CNN
model = create_cnn_model((X_train_reshaped.shape[1], 1), len(label_encoder.classes_))
history = model.fit(X_train_reshaped, y_train, epochs=25, batch_size=64, validation_data=(X_test_reshaped, y_test))


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# getting predictions for test data
predictions = model.predict(X_test_reshaped)

emotion_labels = label_encoder.classes_  # getting original class labels

# for i in range(len(predictions)):
#     emotion_probs = {emotion_labels[j]: predictions[i][j] for j in range(len(emotion_labels))}
#     print(f"Sample {i + 1}: {emotion_probs}")

model.summary()


with open('cnn\\audio_cnn_FAKE.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully as audio_cnn.pkl")