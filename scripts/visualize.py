import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

logging.baseConfig(filename='logs/shodes.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class Visualizer:
    def __init__(self,n_oscillators):
        self.n=n_oscillators
        self.models=['mecpot','genpot1','genpot2','genpot3']
        os.makedirs('plots',exist_ok=True)
    
    def plot_trajectories(self):
        for model_type in self.models:
            data=pd.read_csv(f'data/{model_type}_data.csv')
            t=data['time']
            for i in range(self.n):
                plt.plot(t,data[f'x{i+1}'],label=f'x{i+1}')
            plt.xlabel('Time')
            plt.ylabel('Displacement')
            plt.title(f'{model_type} Oscillator Trajectories')
            plt.legend()
            plt.savefig(f'plots/{model_type}_trajectories.png')
            plt.close()
            logger.info(f'Saved trajectory plot for {model_type}')
            print(f'Saved trajectory plot for {model_type}')

    def plot_potential(self):
        for model_type in self.models:
            data=pd.read_csv(f'data/{model_type}_data.csv')
            t=data['time']
            V=data['potential_energy']
            plt.figure(figsize=(10,6))
            plt.plot(t,V,label='potential energy')
            plt.xlabel('Time')
            plt.ylabel('Potential Energy')
            plt.title(f'{model_type} Potential Energy')
            plt.legend()
            plt.savefig(f'plots/{model_type}_potential.png')
            plt.close()
            logger.info(f'Saved Potential plot for {model_type}')
            print(f'Saved Potential plot for {model_type}')