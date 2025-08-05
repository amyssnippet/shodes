import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

# Define parameters from your PINNTrainer (update n_oscillators and model_type as needed)
n_oscillators = 2  # e.g., 5; set this to match your trainer init
model_type = 'genpot3'  # Change to 'genpot1', 'genpot2', etc., and run the script for each type

# Array of PINN model file paths for the selected model_type (generated dynamically)
pinn_model_paths = [f'models/{model_type}_pinns/pinn_{i+1}.keras' for i in range(n_oscillators)]

# Load models into an array
pinn_models = [tf.keras.models.load_model(path) for path in pinn_model_paths]

# Optional: Freeze layers to retain pre-trained weights
for model in pinn_models:
    for layer in model.layers:
        layer.trainable = False

# Define inputs: shared t (shape (batch, 1)) and x array (shape (batch, n_oscillators))
input_t = Input(shape=(1,), name='pinn_input_t')
input_x = Input(shape=(n_oscillators,), name='pinn_input_x')

# Get outputs from each PINN model (store in array)
pinn_outputs = []
for i, model in enumerate(pinn_models):
    # Extract x_i from input_x (shape (batch, 1))
    x_i = input_x[:, i:i+1]
    # Create input for this PINN: concatenate t and x_i (shape (batch, 2))
    pinn_input = concatenate([input_t, x_i], axis=1)
    # Get prediction
    out = model(pinn_input)
    pinn_outputs.append(out)

# Concatenate all outputs into (batch, n_oscillators)
if len(pinn_outputs) > 1:
    merged_pinn_output = concatenate(pinn_outputs, axis=1, name='merged_pinn_layer')
else:
    merged_pinn_output = pinn_outputs[0]  # Handle single model case

# Optional: Add a final layer if you want further processing (e.g., for a single output; adjust as needed)
# final_pinn_output = Dense(1, activation='linear', name='final_pinn_output')(merged_pinn_output)
# Use merged_pinn_output directly if you just want the concatenated predictions

# Create the merged PINN model (no compilation needed if only for inference; add custom loss if training)
merged_pinn_model = Model(inputs=[input_t, input_x], outputs=merged_pinn_output, name=f'merged_pinn_{model_type}')

# Summary and save
merged_pinn_model.summary()
merged_pinn_model.save(f'models/merged_pinn_{model_type}.keras')

# Example usage: Predict with sample input data
# sample_t = tf.random.normal((1, 1))  # Shape: (batch, 1)
# sample_x = tf.random.normal((1, n_oscillators))  # Shape: (batch, n_oscillators)
# predictions = merged_pinn_model.predict([sample_t, sample_x])  # Outputs: (1, n_oscillators)
# print(predictions)
