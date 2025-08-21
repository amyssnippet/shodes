import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

logging.basicConfig(
    filename='logs/shodes.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

class Visualizer:
    def __init__(self, n_oscillators):
        self.n = n_oscillators
        # expand to 25 models; adjust names if needed
        self.models = ['mecpot'] + [f'genpot{i}' for i in range(1, 25)]
        os.makedirs('new-plots', exist_ok=True)

    def plot_trajectories(self):
        for model_type in self.models:
            data_file = f'data/{model_type}_data.csv'
            if not os.path.exists(data_file):
                logger.warning(f"Data file not found: {data_file}")
                continue

            data = pd.read_csv(data_file)
            t = data['time']

            plt.figure(figsize=(10, 6))
            for i in range(self.n):
                plt.plot(t, data[f'x{i+1}'], label=f'x{i+1}', linewidth=1)

            plt.xlabel('Time')
            plt.ylabel('Displacement')
            plt.title(f'{model_type} Oscillator Trajectories')
            plt.legend(ncol=4, fontsize='small', loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.6)

            outpath = f'new-plots/{model_type}_trajectories.png'
            plt.savefig(outpath, bbox_inches='tight')
            plt.close()
            logger.info(f'Saved trajectory plot for {model_type}')
            print(f'Saved trajectory plot for {model_type}')

    def plot_potential(self):
        for model_type in self.models:
            data_file = f'data/{model_type}_data.csv'
            if not os.path.exists(data_file):
                logger.warning(f"Data file not found: {data_file}")
                continue

            data = pd.read_csv(data_file)
            t = data['time']
            V = data['potential_energy']

            plt.figure(figsize=(10, 6))
            plt.plot(t, V, label='Potential Energy', color='darkred', linewidth=1.5)
            plt.xlabel('Time')
            plt.ylabel('Potential Energy')
            plt.title(f'{model_type} Potential Energy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            outpath = f'new-plots/{model_type}_potential.png'
            plt.savefig(outpath, bbox_inches='tight')
            plt.close()
            logger.info(f'Saved potential plot for {model_type}')
            print(f'Saved potential plot for {model_type}')

    def plot_loss_curves(self):
        for model_type in self.models:
            plt.figure(figsize=(10, 6))
            found = False

            # plot all PINN loss curves
            for i in range(self.n):
                loss_file = f'models/{model_type}_pinns/loss_pinn_{i+1}.csv'
                if os.path.exists(loss_file):
                    df = pd.read_csv(loss_file)
                    plt.plot(df['epoch'], df['loss'], label=f'PINN {i+1}', linewidth=1)
                    found = True

            # plot DNN curve if available
            dnn_file = f'models/{model_type}_dnn/loss_dnn.csv'
            if os.path.exists(dnn_file):
                df_dnn = pd.read_csv(dnn_file)
                plt.plot(df_dnn['epoch'], df_dnn['loss'], label='DNN',
                         linestyle='--', color='black', linewidth=1.5)
                found = True

            if not found:
                logger.warning(f"No loss data found for {model_type}")
                plt.close()
                continue

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss vs Epochs - {model_type}')
            # legend outside if too many curves
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.grid(True, linestyle='--', alpha=0.6)

            outpath = f'new-plots/{model_type}_loss_comparison.png'
            plt.savefig(outpath, bbox_inches='tight')
            plt.close()
            logger.info(f'Saved loss plot for {model_type}')
            print(f'Saved loss plot for {model_type}')


if __name__ == "__main__":
    vis = Visualizer(n_oscillators=25)
    vis.plot_trajectories()
    vis.plot_potential()
    vis.plot_loss_curves()
