import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

df = pd.read_csv("data.csv")

X = df.iloc[:, :-1].values
y = df["label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(X_train,
          y_train,
          epochs=50,
          batch_size=128,
          validation_split=0.2,
          callbacks=[early_stop])

model.save("test_model.h5")
print("Model saved as 'test_model.h5'")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("Label encoder saved as 'label_encoder.pkl'")