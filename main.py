import os
from scripts.data_generator import DataGenerator
from scripts.pinn_trainer import PINNTrainer
from scripts.dnn_trainer import DNNTrainer
from scripts.visualize import Visualizer
import logging

logging.basicConfig(filename='logs/shodes.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def main():
    n_oscillators = 3
    k = 1.0
    kc = 0.1
    lambda_ = 0.1
    m = 1.0
    pinn_epochs = 1500
    pinn_layers = [64, 64, 64, 64]
    dnn_epochs = 1000
    dnn_layers = [128, 128, 64]

    logger.info('Starting SHODes Framework')
    print('Starting SHODes Framework...')

    data_gen = DataGenerator(n_oscillators, k, kc, lambda_, m)
    data_gen.generate()

    pinn_trainer = PINNTrainer(n_oscillators, k, kc, lambda_, m, pinn_layers)
    pinn_trainer.train(pinn_epochs)

    dnn_trainer = DNNTrainer(n_oscillators, k, kc, lambda_, dnn_layers)
    dnn_trainer.train(dnn_epochs)

    visualizer = Visualizer(n_oscillators)
    visualizer.plot_trajectories()
    visualizer.plot_potential()
    visualizer.plot_loss_curves()

    logger.info('SHODes Framework completed')
    print('SHODes Framework completed')


if __name__=='__main__':
    os.makedirs('logs',exist_ok=True)
    os.makedirs('plots',exist_ok=True)
    for model in ['mecpot','genpot1','genpot2','genpot3']:
        os.makedirs(f'models/{model}_pinns',exist_ok=True)
        os.makedirs(f'models/{model}_dnn',exist_ok=True)
    main()