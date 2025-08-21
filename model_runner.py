import tensorflow as tf
import numpy as np
import pandas as pd

# Configuration: Update these based on your project setup
n_oscillators = 25  # From your trainer inits (e.g., 5)
model_type = 'genpot3'  # e.g., 'mecpot', 'genpot1', 'genpot2', 'genpot3'
use_merged_models = True  # Set to True if using merged models; False for individual
params = [1.0, 0.1, 0.1]  # [k, kc, lambda_] from your trainers; update as needed
batch_size = 10  # Number of samples to predict on (e.g., time steps)
use_real_data = False  # Set to True to load from CSV; False for dummy data

# Paths (update if different)
if use_merged_models:
    pinn_path = f'models/merged_pinn_{model_type}.keras'
    dnn_path = f'models/merged_dnn_model.keras'  # Assuming one merged DNN across types; adjust if per-type
else:
    pinn_paths = [f'models/{model_type}_pinns/pinn_{i+1}.keras' for i in range(n_oscillators)]
    dnn_path = f'models/{model_type}_dnn.keras'  # Per-type DNN

# Load models with custom objects if needed (e.g., for pinn_loss; define it here if custom)
# custom_objects = {'pinn_loss': your_pinn_loss_function}  # Uncomment and define if required
if use_merged_models:
    try:
        merged_pinn = tf.keras.models.load_model(pinn_path)  # Add custom_objects=custom_objects if needed
        print(f"Loaded merged PINN for {model_type}")
        merged_pinn.summary()  # Print summary for debugging
    except Exception as e:
        print(f"Error loading merged PINN: {e}. Falling back to individual PINNs.")
        use_merged_models = False
else:
    pinns = [tf.keras.models.load_model(path) for path in pinn_paths]  # Add custom_objects if needed
    print(f"Loaded {n_oscillators} individual PINNs for {model_type}")

dnn = tf.keras.models.load_model(dnn_path)  # Add custom_objects if needed
print(f"Loaded DNN for system-level predictions")
dnn.summary()  # Print summary for debugging

# Prepare input data
if use_real_data:
    # Load from your CSV (assumes time and x1 to xn columns)
    data = pd.read_csv(f'data/{model_type}_data.csv')
    t = data['time'].values.astype(np.float32)[:batch_size]  # Take first batch_size samples
    x_data = np.array([data[f'x{i+1}'].values.astype(np.float32)[:batch_size] for i in range(n_oscillators)]).T  # Shape: (batch_size, n_oscillators)
else:
    # Dummy data: random t and x for testing
    t = np.random.random((batch_size,)).astype(np.float32)  # Shape: (batch_size,)
    x_data = np.random.random((batch_size, n_oscillators)).astype(np.float32)  # Shape: (batch_size, n_oscillators)

# Step 1: Run PINN predictions for each SHO (oscillator)
pinn_outputs = np.zeros((batch_size, n_oscillators))  # To store predictions: (batch_size, n_oscillators)

if use_merged_models:
    try:
        # For merged PINN: Try inputs as [t (batch_size, 1), x (batch_size, n_oscillators)]
        t_input = np.expand_dims(t, axis=1)  # Shape: (batch_size, 1)
        x_input = x_data  # Shape: (batch_size, n_oscillators)
        pinn_outputs = merged_pinn.predict([t_input, x_input])  # Expected outputs: (batch_size, n_oscillators)
    except ValueError as ve:
        print(f"Shape mismatch in merged PINN predict: {ve}")
        print("Attempting alternative input structure (single input of (batch_size, n_oscillators + 1))...")
        try:
            # Alternative: Concat t and x into single input (batch_size, n_oscillators + 1)
            alt_input = np.concatenate([t_input, x_input], axis=1)  # Shape: (batch_size, n_oscillators + 1)
            pinn_outputs = merged_pinn.predict(alt_input)
        except Exception as e:
            print(f"Alternative failed: {e}. Falling back to individual PINN predictions.")
            use_merged_models = False

if not use_merged_models:
    # Fallback: Loop over individual PINNs
    try:
        pinns = [tf.keras.models.load_model(path) for path in pinn_paths]
    except:
        raise ValueError("Could not load individual PINNs. Check paths and model files.")
    for i in range(n_oscillators):
        x_i = x_data[:, i]  # Shape: (batch_size,)
        pinn_input = np.stack([t, x_i], axis=1)  # Shape: (batch_size, 2)
        pinn_outputs[:, i] = pinns[i].predict(pinn_input).squeeze()  # Predict and store

print(f"PINN predictions for {n_oscillators} SHOs (first sample): {pinn_outputs[0]}")
print(f"PINN output shape: {pinn_outputs.shape}")

# Step 2: Prepare DNN input: Concat PINN outputs + repeated params
params_repeated = np.tile(params, (batch_size, 1))  # Shape: (batch_size, 3)
dnn_input = np.concatenate([pinn_outputs, params_repeated], axis=1)  # Shape: (batch_size, n_oscillators + 3)

# Step 3: Run DNN for system-level prediction (e.g., potential energy)
potential_predictions = dnn.predict(dnn_input)

print(f"System-level DNN predictions (potential energy): {potential_predictions.squeeze()}")
print(f"DNN output shape: {potential_predictions.shape}")  # Should be (batch_size, 1) or adjusted for merged DNN

# Optional: Sanity checks
if np.any(np.isnan(pinn_outputs)) or np.any(np.isnan(potential_predictions)):
    print("Warning: Predictions contain NaN values - check model training or inputs.")
else:
    print("All predictions look good - no NaNs detected.")
