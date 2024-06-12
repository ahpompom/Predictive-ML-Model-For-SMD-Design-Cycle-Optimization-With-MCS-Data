# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:38:12 2023
@author: ahpompom
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
file_path = 'C:/Project 2/ComponentData_1 (1mil).csv'
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop('Core-Package conflict', axis=1)
y = data['Core-Package conflict'].astype(int)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Address class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE, counts of label '1':", sum(y_train_smote == 1))
print("After SMOTE, counts of label '0':", sum(y_train_smote == 0))

# Initialize Random Forest classifier with chosen parameters
rf_classifier = RandomForestClassifier(
    n_estimators=200,    # Example: set the number of trees to 200
    max_depth=None,      # Example: set the maximum depth of the tree to None
    min_samples_split=2, # Example: minimum number of samples required to split a node
    min_samples_leaf=1,  # Example: minimum number of samples required at a leaf node
    random_state=42
)

# Train the Random Forest classifier
rf_classifier.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate the classifier
print("Random Forest Model Evaluation")
print("Accuracy on Test Set: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print(classification_report(y_test, y_pred))

# Save the Random Forest model
model_filename = 'C:/Project 2/RF_model_Rev1b.joblib'
joblib.dump(rf_classifier, model_filename)

# Save the scaler
scaler_filename = 'C:/Project 2/scaler_Rev1b.joblib'
joblib.dump(scaler, scaler_filename)

print(f"Model saved as {model_filename}")
print(f"Scaler saved as {scaler_filename}")