import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math
import random

# Load the model and scaler
def load_model_and_scaler(model_path, scaler_path):
    model = load_model(model_path)
    with open(scaler_path, 'rb') as handle:
        scaler = pickle.load(handle)
    return model, scaler

# Predict the error rate given a set of dimensions with detailed debugging
def predict_error_rate_with_debugging(model, scaler, dimensions):
    feature_names = [
        'Core coor X', 'Core coor Y', 'Core width', 'Core length',
        'Core hole coor X', 'Core hole coor Y', 'Core hole diameter',
        'Rod coor X', 'Rod coor Y', 'Rod diameter',
        'Base-plate coor X', 'Base-plate coor Y', 'Base-plate width', 'Base-plate length',
        'Package width', 'Package length', 'Core rotation'
    ]
    
    input_df = pd.DataFrame([dimensions], columns=feature_names)
    print("Input dimensions (unscaled):", input_df)
    
    scaled_dimensions = scaler.transform(input_df)
    print("Scaled dimensions:", scaled_dimensions)
    
    prediction = model.predict(scaled_dimensions)
    print("Model prediction (raw):", prediction)
    
    error_rate = abs(prediction[0][0])  # Ensure non-negative error rate
    print("Computed error rate:", error_rate)
    
    return error_rate

# Propose new dimensions by randomly adjusting within bounds using a normal distribution
def propose_new_dimensions(current_dimensions, bounds, std_dev_factor=0.1):
    new_dimensions = []
    for dim, (lower, upper) in zip(current_dimensions, bounds):
        std_dev = (upper - lower) * std_dev_factor
        new_dim = np.random.normal(dim, std_dev)
        new_dim = max(min(new_dim, dim + upper), dim + lower)
        new_dimensions.append(new_dim)
    return new_dimensions

# Simulated Annealing optimization function with adjusted parameters
def optimize_dimensions_with_sa_adjusted(model, scaler, initial_dimensions, bounds, iterations=1000, initial_temp=0.5, cooling_rate=0.995, tolerance=1e-5, patience=300):
    current_dimensions = initial_dimensions
    current_error = predict_error_rate_with_debugging(model, scaler, current_dimensions)
    best_error = current_error
    best_dimensions = current_dimensions
    error_history = [current_error]
    temperature = initial_temp
    no_improvement_count = 0

    for i in range(iterations):
        new_dimensions = propose_new_dimensions(current_dimensions, bounds)
        new_error = predict_error_rate_with_debugging(model, scaler, new_dimensions)
        error_history.append(new_error)

        # Apply Simulated Annealing acceptance criterion with adjusted parameters
        if new_error < current_error or math.exp((current_error - new_error) / temperature) > random.random():
            current_dimensions = new_dimensions
            current_error = new_error

            if new_error < best_error - tolerance:
                best_error = new_error
                best_dimensions = new_dimensions
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1

        temperature *= cooling_rate

        if no_improvement_count >= patience:
            print(f'No improvement for {patience} consecutive iterations, stopping early.')
            break

    return best_dimensions, best_error, error_history

# Load the model and scaler
model_path = 'C:/Project 2/NN_Model_Rev1b.keras'
scaler_path = 'C:/Project 2/scaler_NN_Rev1b.pkl'
model, scaler = load_model_and_scaler(model_path, scaler_path)

# Define initial dimensions based on nominal values
initial_dimensions = [
    4.0, 3.771, 3.85, 3.94, 2.56, 1.6, 2.71, 1.75, 8.0, 8.0, 2.56, 1.6, 2.71, 1.75, 8.0, 8.0, 0.0
]

# Define bounds for each feature based on given tolerances
bounds = [
    (-0.1, 0.1), (-0.1, 0.1), (-0.05, 0.05), (-0.05, 0.05), 
    (-0.05, 0.05), (-0.05, 0.05), (-0.03, 0.03), (-0.03, 0.03), 
    (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 
    (-0.03, 0.03), (-0.03, 0.03), (-0.05, 0.05), (-0.05, 0.05), 
    (-1.0, 1.0)
]

# Run the optimization with adjusted parameters
optimized_dimensions, optimized_error, error_history = optimize_dimensions_with_sa_adjusted(
    model, scaler, initial_dimensions, bounds, iterations=1000
)

# Print optimized dimensions and error rate
print("Optimized Dimensions:", optimized_dimensions)
print("Optimized Error Rate:", optimized_error)

# Plot the error history
plt.plot(error_history)
plt.xlabel('Iteration')
plt.ylabel('Error Rate')
plt.title('Error Rate Over Optimization Iterations with Adjusted Simulated Annealing')
plt.show()