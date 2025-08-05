import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(filename='logs/shodes.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class DNNTrainer:
    def __init__(self, n_oscillators, k=1.0, kc=0.1, lambda_=0.1, layers=None):
        self.n = n_oscillators
        self.k = k
        self.kc = kc
        self.lambda_ = lambda_
        self.layers = layers if layers is not None else [100, 100, 100]
        self.models = ['mecpot', 'genpot1', 'genpot2', 'genpot3']
        self.dnns = {model: self.build_dnn() for model in self.models}
        os.makedirs('models', exist_ok=True)

    def build_dnn(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.n + 3,)))
        for units in self.layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        return model

    def compute_potential(self, x, model_type):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        V = 0.5 * self.k * tf.reduce_sum(x**2, axis=1)
        if model_type == 'mecpot':
            V += 0.5 * self.kc * tf.reduce_sum((x[:, :-1] - x[:, 1:])**2, axis=1)
        elif model_type == 'genpot1':
            V += self.lambda_ * tf.reduce_sum(x[:, :-1] - x[:, 1:], axis=1)
        elif model_type == 'genpot2':
            V += 0.5 * self.lambda_ * (tf.reduce_sum(x, axis=1))**2
        elif model_type == 'genpot3':
            V += self.lambda_ * tf.reduce_sum(x, axis=1)
        return V

    def train(self, epochs=1000):
        for model_type in self.models:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            logger.info(f'Training DNN for {model_type}')
            print(f'Training DNN for {model_type}')

            data = pd.read_csv(f'data/{model_type}_data.csv')
            x_data = np.array([data[f'x{i+1}'] for i in range(self.n)]).T
            V_true = data['potential_energy'].values
            t = np.expand_dims(data['time'].values, axis=1)

            pinns = [tf.keras.models.load_model(f'models/{model_type}_pinns/pinn_{i+1}.keras') for i in range(self.n)]
            loss_history = []

            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    pinn_outputs = []
                    for i, pinn in enumerate(pinns):
                        x_i = x_data[:, i:i+1]
                        inp = np.hstack([t, x_i])
                        out = pinn.predict(inp, verbose=0)
                        pinn_outputs.append(out)

                    pinn_outputs = tf.concat(pinn_outputs, axis=1)
                    params = tf.repeat([[self.k, self.kc, self.lambda_]], repeats=len(x_data), axis=0)
                    dnn_input = tf.concat([pinn_outputs, params], axis=1)

                    V_pred = self.dnns[model_type](dnn_input)
                    V_true_tensor = tf.convert_to_tensor(V_true, dtype=tf.float32)
                    loss = tf.reduce_mean(tf.square(V_pred - tf.expand_dims(V_true_tensor, axis=1)))

                gradients = tape.gradient(loss, self.dnns[model_type].trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.dnns[model_type].trainable_variables))
                loss_history.append(loss.numpy())

                if epoch % 100 == 0:
                    logger.info(f'{model_type} DNN, Epoch {epoch}, Loss: {loss.numpy():.4f}')
                    print(f'{model_type} DNN, Epoch {epoch}, Loss: {loss.numpy():.4f}')

            # Save model and loss
            os.makedirs(f'models/{model_type}_dnn', exist_ok=True)
            self.dnns[model_type].save(f'models/{model_type}_dnn.keras')

            df_loss = pd.DataFrame({'epoch': list(range(epochs)), 'loss': loss_history})
            df_loss.to_csv(f'models/{model_type}_dnn/loss_dnn.csv', index=False)

            logger.info(f'Saved DNN for {model_type}')
