import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model and scaler
model = load_model('C:/Project 2/NN_Model_Rev1b.keras')

# Print the model summary to understand the layer structure
model.summary()

# Function to extract and visualize weights
def visualize_weights(model):
    for layer in model.layers:
        if 'conv' in layer.name:  # Identify convolutional layers
            weights, biases = layer.get_weights()
            # Sum weights across all filters and kernels
            feature_importance = np.sum(np.abs(weights), axis=(0, 1))
            # Normalize for better visualization
            feature_importance_normalized = feature_importance / np.max(feature_importance)
            plt.bar(range(len(feature_importance_normalized)), feature_importance_normalized)
            plt.xlabel('Feature Index')
            plt.ylabel('Normalized Feature Importance')
            plt.title(f'Feature Importance from {layer.name}')
            plt.show()

# Call the function to visualize weights
visualize_weights(model)
