
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'C:/Project 2/ComponentData_1 (1mil).csv'  # Use the correct path
data = pd.read_csv(file_path)

# Prepare the data
X = data.drop('Core-Package conflict', axis=1)
y = data['Core-Package conflict'].astype(int)  # Convert to int for scikit-learn

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test_scaled)

# Get probabilities instead of predicted classes for NN input
rf_probs = rf_classifier.predict_proba(X_test_scaled)

# Define a simple Neural Network model
nn_model = Sequential([
    Input(shape=(rf_probs.shape[1],)),  # Specify input shape here
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Neural Network on the Random Forest probabilities
nn_model.fit(rf_probs, y_test, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the Neural Network model
loss, accuracy = nn_model.evaluate(rf_probs, y_test)
print(f'Neural Network Model Accuracy: {accuracy * 100:.2f}%')

# Save the Neural Network model in the Keras format (recommended)
nn_model.save('C:/Project 2/NN_model_Rev1b.keras')

# Save the Random Forest model
joblib.dump(rf_classifier, 'C:/Project 2/RF_model_Rev1b.joblib')

# Random Forest Confusion Matrix
rf_cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

print("Random Forest Model Evaluation")
print(classification_report(y_test, y_pred))

# Neural Network predictions (as probabilities)
nn_probs = nn_model.predict(rf_probs)
nn_preds = (nn_probs > 0.5).astype(int)

# Neural Network Confusion Matrix
nn_cm = confusion_matrix(y_test, nn_preds)

plt.figure(figsize=(8,6))
sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
plt.title('Neural Network Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

print("Neural Network Model Evaluation")
print(f"Accuracy: {accuracy_score(y_test, nn_preds) * 100:.2f}%")
print(classification_report(y_test, nn_preds))