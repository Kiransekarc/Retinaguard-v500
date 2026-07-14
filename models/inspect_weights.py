import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the uncorrupted model
print("Loading model...")
model = load_model('RetinaGuard_Clinical_Balanced.h5')

# 2. Get all the layers in the model
print("\n=== MODEL ARCHITECTURE ===")
model.summary()

# 3. Let's inspect the weights of the very first Convolutional Layer
first_layer = model.layers[0]

# get_weights() returns a list: [weights_matrix, biases_array]
weights_and_biases = first_layer.get_weights()

if len(weights_and_biases) > 0:
    weights = weights_and_biases[0]
    biases = weights_and_biases[1]
    
    print("\n=== INSPECTING WEIGHTS (First Layer) ===")
    print(f"Layer Name: {first_layer.name}")
    print(f"Weight Matrix Shape: {weights.shape}")
    print(f"Bias Array Shape: {biases.shape}")
    
    print("\nHere are the actual mathematical weights (first 5 values) that the AI learned:")
    print(weights.flatten()[:5])
else:
    print("\nThis layer does not have trainable weights.")
