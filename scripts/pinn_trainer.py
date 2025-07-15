import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(filename='logs/shodes.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class PINNTrainer:
    def __init__(self, n_oscillators, k=1.0, kc=0.1, lambda_=0.1, m=1.0, layers=None):
        self.n = n_oscillators
        self.k = k
        self.kc = kc
        self.lambda_ = lambda_
        self.m = m
        self.layers = layers if layers is not None else [50, 50, 50]  # default architecture
        self.models = ['mecpot', 'genpot1', 'genpot2', 'genpot3']
        self.pinns = {model: [self.build_pinn() for _ in range(self.n)] for model in self.models}
        os.makedirs('models', exist_ok=True)

    def build_pinn(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(2,)))  # Input layer for (t, x)
        for units in self.layers:
            model.add(tf.keras.layers.Dense(units, activation='tanh'))
        model.add(tf.keras.layers.Dense(1))  # Output layer
        return model

    def pinn_loss(self, pinn, t, x_true, model_type, i):
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        x_true = tf.convert_to_tensor(x_true, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            input_data = tf.stack([t, x_true], axis=1)
            x_pred = pinn(input_data)
            dx_dt = tape.gradient(x_pred, t)
            d2x_dt2 = tape.gradient(dx_dt, t)
            x_pred = tf.squeeze(x_pred)
        if model_type == 'mecpot':
            coupling = 0
            if i > 0:
                x_prev = tf.convert_to_tensor(x_true[i-1], dtype=tf.float32)
                coupling += self.kc * (x_pred - x_prev)
            if i < self.n-1:
                x_next = tf.convert_to_tensor(x_true[i+1], dtype=tf.float32)
                coupling += self.kc * (x_pred - x_next)
            physics = self.m * d2x_dt2 + self.k * x_pred + coupling
        elif model_type == 'genpot1':
            coupling = 0
            if i > 0:
                x_prev = tf.convert_to_tensor(x_true[i-1], dtype=tf.float32)
                coupling += self.lambda_ * x_prev
            if i < self.n-1:
                x_next = tf.convert_to_tensor(x_true[i+1], dtype=tf.float32)
                coupling += self.lambda_ * x_next
            physics = self.m * d2x_dt2 + self.k * x_pred + coupling
        elif model_type == 'genpot2':
            x_sum = tf.reduce_sum(x_true, axis=0)
            physics = self.m * d2x_dt2 + self.k * x_pred + self.lambda_ * x_sum
        elif model_type == 'genpot3':
            physics = self.m * d2x_dt2 + self.k * x_pred + self.lambda_
        physics_loss = tf.reduce_mean(tf.square(physics))
        data_loss = tf.reduce_mean(tf.square(x_pred - x_true))
        return data_loss + physics_loss

    def train(self, epochs=1000):
        for model_type in self.models:
            logger.info(f'Training PINNs for {model_type}')
            print(f'Training PINNs for {model_type}')
            
            data = pd.read_csv(f'data/{model_type}_data.csv')
            t = data['time'].values
            x_data = [data[f'x{i+1}'].values for i in range(self.n)]

            for i, pinn in enumerate(self.pinns[model_type]):
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                loss_history = []

                for epoch in range(epochs):
                    with tf.GradientTape() as tape:
                        loss = self.pinn_loss(pinn, t, x_data[i], model_type, i)
                    gradients = tape.gradient(loss, pinn.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, pinn.trainable_variables))

                    loss_history.append(loss.numpy())

                    if epoch % 100 == 0:
                        logger.info(f'{model_type} PINN{i+1}, Epoch {epoch}, Loss: {loss.numpy():.4f}')
                        print(f'{model_type} PINN{i+1}, Epoch {epoch}, Loss: {loss.numpy():.4f}')

                # Save model and loss curve
                os.makedirs(f'models/{model_type}_pinns', exist_ok=True)
                pinn.save(f'models/{model_type}_pinns/pinn_{i+1}.keras')

                df_loss = pd.DataFrame({'epoch': list(range(epochs)), 'loss': loss_history})
                df_loss.to_csv(f'models/{model_type}_pinns/loss_pinn_{i+1}.csv', index=False)

                logger.info(f'Saved PINN {i+1} for {model_type}')
