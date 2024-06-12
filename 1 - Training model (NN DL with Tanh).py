# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 02:11:42 2024

@author: ahpompom
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import pickle 

print("Loading dataset...")
# Load the dataset
file_path = 'C:\Project 2/ComponentData_1 (1mil).csv'  # Adjust to your file path
data = pd.read_csv(file_path)
print("Dataset loaded successfully.")

print("Preparing the data...")
try:
    X = data.drop('Core-Package conflict', axis=1)
    y = data['Core-Package conflict']
except KeyError as e:
    print(f"Column not found: {e}")
    import sys
    sys.exit(f"Script terminated: {e}")

# Define 'input_dim' here
input_dim = X.shape[1]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
scaler_path = 'C:\Project 2\scaler_NN_Rev1b.pkl'  # Replace with the actual path you want to use
with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)
print("Features scaled and scaler saved.")

# Now proceed to build and train your model
# Define the neural network model

def build_model(input_dim):
    print("Building the model...")
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(32, activation='tanh'),
        Dense(16, activation='tanh'),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# 'input_dim' is defined above
model = build_model(input_dim)

# Set up early stopping to interrupt the training process when the model's performance on the validation set stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# Training the model
print("Training the model with early stopping...")
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=2)

# Save the model
model.save('C:/Project 2/NN_Model_Rev1b.h5')

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Generate predictions and evaluate
print("Generating predictions and evaluating...")
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Process completed.")