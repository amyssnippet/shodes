import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

# Define parameters from your DNNTrainer (update n_oscillators as needed)
n_oscillators = 25  # e.g., 5; set this to match your trainer init

# Array of DNN model file paths (add/remove based on your trained models)
dnn_model_paths = [
    'models/mecpot_dnn.keras',
    'models/genpot1_dnn.keras',
    'models/genpot2_dnn.keras',
    'models/genpot3_dnn.keras'
]  # Update with actual paths if different

# Load models into an array
dnn_models = [tf.keras.models.load_model(path) for path in dnn_model_paths]

# Optional: Freeze layers to retain pre-trained weights
for model in dnn_models:
    for layer in model.layers:
        layer.trainable = False

# Define a shared input based on your DNN input shape
input_shape = (n_oscillators + 3,)  # From your build_dnn: n oscillators + 3 params
shared_input = Input(shape=input_shape, name='dnn_shared_input')

# Get outputs from each DNN model (store in array)
dnn_outputs = [model(shared_input) for model in dnn_models]

# Concatenate all outputs (assumes compatible shapes; use Average() for averaging if preferred)
if len(dnn_outputs) > 1:
    merged_dnn_output = concatenate(dnn_outputs, name='merged_dnn_layer')
else:
    merged_dnn_output = dnn_outputs[0]  # Handle single model case

# Add a final output layer (matches your DNN: 1 unit, linear for potential energy)
final_dnn_output = Dense(1, activation='linear', name='final_dnn_output')(merged_dnn_output)

# Create and compile the merged DNN model (use your original loss/optimizer)
merged_dnn_model = Model(inputs=shared_input, outputs=final_dnn_output, name='merged_dnn')
merged_dnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Summary and save
merged_dnn_model.summary()
merged_dnn_model.save('models/merged_dnn_model.keras')

# Example usage: Predict with sample input data
# sample_input = tf.random.normal((1, n_oscillators + 3))  # Shape: (batch, n+3)
# predictions = merged_dnn_model.predict(sample_input)
# print(predictions)
