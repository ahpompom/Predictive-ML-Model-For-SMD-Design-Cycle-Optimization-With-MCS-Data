import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Input, MaxPooling1D, Flatten
import pickle

input_shape = (17, 1)  # The input shape should be a tuple

def build_conv_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Change to linear for continuous output
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error',  # Change to MSE or another suitable regression loss
                  metrics=['mae'])  # Consider using Mean Absolute Error for regression metrics
    return model

print("Loading dataset...")
# Load the dataset
file_path = 'C:/Project 2/ComponentData_1 (1mil).csv'  # Adjust to your file path
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

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the features for training and testing
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 17, 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 17, 1))

# Set up early stopping to interrupt the training process when the model's performance on the validation set stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training the model
print("Training the model with early stopping...")
model = build_conv_model(input_shape)
history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=2)

# Save the model in the recommended native Keras format
model.save('C:/Project 2/NN_Model_Rev1b.keras')

# Evaluate the model
print("Evaluating the model...")
loss, mae = model.evaluate(X_test_reshaped, y_test, verbose=2)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Generate predictions and evaluate
print("Generating predictions and evaluating...")
y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Process completed.")

# Save the scaler
scaler_path = 'C:/Project 2/scaler_NN_Rev1b.pkl'
with open(scaler_path, 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Reshape the features for training and testing
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 17, 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 17, 1))