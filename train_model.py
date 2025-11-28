import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import os

# --- Load and Prepare Data ---
try:
    features_df = pd.read_pickle('features.pkl')
except FileNotFoundError:
    print("Error: 'features.pkl' not found. Please run feature_extraction.py first.")
    exit()

if features_df.empty:
    print("Error: The 'features.pkl' file is empty. Cannot train the model.")
    exit()

X = features_df.drop(columns=['path', 'label'])
y = features_df['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Reshape features for the 1D CNN
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)


# --- Build the CNN Model ---
print("\n--- Building the CNN Model ---")
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))
num_classes = y_categorical.shape[1]
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# --- Train the Model ---
print("\n--- Training the Model ---")
# This is the part that takes a long time
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    verbose=1
)


# --- Evaluate the Model ---
print("\n--- Evaluating the Model ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# --- Save the Trained Model (with Error Handling) ---
print("\n--- Attempting to save the model ---")
try:
    model.save('emotion_model.h5')
    print("--- Successfully saved model to 'emotion_model.h5' ---")

    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("--- Successfully saved label encoder to 'label_encoder.pkl' ---")

except Exception as e:
    print("\n!!!!!!!! AN ERROR OCCURRED DURING FILE SAVING !!!!!!!!")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")