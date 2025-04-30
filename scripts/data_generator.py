import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os
import logging

logging.basicConfig(filename='logs/shodes.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class DataGenerator:
    def __init__(self, n_oscillators, k=1.0, kc=0.1, lambda_=0.1, m=1.0, t_span=(0,10), n_points=100):
        self.n = n_oscillators
        self.k = k
        self.kc = kc
        self.lambda_ = lambda_
        self.m = m
        self.t_span = t_span
        self.n_points = n_points
        self.models = ['mecpot', 'genpot1', 'genpot2', 'genpot3']
        os.makedirs('data', exist_ok=True)

    def equations(self, state, t, model_type):
        x = state[:self.n]
        v = state[self.n:]
        dxdt= v
        dvdt = np.zeros(self.n)
        if model_type == 'mecpot':
            for i in range(self.n):
                coupling = 0
                if i > 0:
                    coupling += self.kc * (x[i] - x[i-1])
                if i < self.n-1:
                    coupling += self.kc * (x[i] - x[i+1])
                dvdt[i] = -(self.k * x[i] + coupling) / self.m

        elif model_type == 'genpot1':
            for i in range(self.n):
                coupling = 0
                if i > 0:
                    coupling += self.lambda_ * x[i-1]
                if i < self.n-1:
                    coupling += self.lambda_ * x[i+1]
                dvdt[i] = -(self.k * x[i] + coupling) / self.m

        elif model_type == 'genpot2':
            sum_x = np.sum(x)
            for i in range(self.n):
                dvdt[i] = -(self.k * x[i] + self.lambda_ * sum_x) / self.m

        elif model_type == 'genpot3':
            for i in range(self.n):
                dvdt[i] = -(self.k * x[i] + self.lambda_ ) / self.m

        return np.concatenate([dxdt, dvdt])

    def compute_potential(self, x, model_type):
        V = 0.5 * self.k * np.sum(x**2)
        if model_type == 'mecpot':
            V += 0.5 * self.kc * np.sum((x[:-1] - x[1:])**2)
        elif model_type == 'genpot1':
            V += self.lambda_ * np.sum(x[:-1] - x[1:])
        elif model_type == 'genpot2':
            V += 0.5 * self.lambda_ * (np.sum(x))**2
        elif model_type == 'genpot3':
            V += self.lambda_ * np.sum(x)
        return V
    
    def generate(self):
        t = np.linspace(self.t_span[0], self.t_span[1], self.n_points)
        initial_conditions = np.random.randn(2*self.n)*0.1
        for model_type in self.models:
            logger.info(f'Generated data for {model_type}')
            print(f'Generated data for {model_type}')
            sol=odeint(self.equations, initial_conditions,t,args=(model_type,))
            x=sol[:,:self.n]
            V = np.array([self.compute_potential(x[i], model_type) for i in range(self.n_points)])
            data=pd.DataFrame({
                'time': t,
                **{f'x{i+1}': x[:,i] for i in range(self.n)},
                'potential_energy': V
            })

            data_path = f'data/{model_type}_data.csv'
            data.to_csv(data_path, index=False)
            logger.info(f'Saved Data to {data_path}')
            print(data.head())