import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

# --- Kuramoto and Strogatz Models ---
def kuramoto_rhs(theta, omega, K):
    diff = theta[None, :] - theta[:, None]
    coupling = np.sum(np.sin(diff), axis=1) - np.sin(0)
    return omega + (K / 2) * coupling

def simulate_kuramoto_nosc(T=100.0, dt=0.1, K=0.1, n=25, omega=None, theta0=None):
    steps = int(T / dt)
    theta = np.zeros((steps + 1, n))
    if omega is None:
        omega = np.linspace(1.0, 1.5, n)  # frequencies across oscillators
    if theta0 is None:
        theta0 = np.random.uniform(-np.pi, np.pi, size=n)
    theta[0] = theta0
    for t in range(steps):
        k1 = kuramoto_rhs(theta[t], omega, K)
        k2 = kuramoto_rhs(theta[t] + 0.5*dt*k1, omega, K)
        k3 = kuramoto_rhs(theta[t] + 0.5*dt*k2, omega, K)
        k4 = kuramoto_rhs(theta[t] + dt*k3, omega, K)
        theta[t+1] = theta[t] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    time = np.linspace(0, T, steps + 1)
    return time, theta

def kuramoto_order_parameter(theta):
    z = np.mean(np.exp(1j * theta))
    return np.abs(z)

def watts_strogatz_adjacency(n, k=1, p=0.5, seed=42):
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    np.fill_diagonal(A, 0.0)
    return A

def kuramoto_rhs_network(theta, omega, K, A):
    diff = theta[None, :] - theta[:, None]
    coupling = (A * np.sin(diff)).sum(axis=1)
    return omega + (K / max(1, A.shape[0])) * coupling

def simulate_kuramoto_strogatz_nosc(T=100.0, dt=0.1, K=0.1, n=25, omega=None, theta0=None, p=0.5):
    A = watts_strogatz_adjacency(n=n, k=2, p=p, seed=42)
    steps = int(T / dt)
    theta = np.zeros((steps + 1, n))
    if omega is None:
        omega = np.linspace(1.0, 1.5, n)
    if theta0 is None:
        theta0 = np.random.uniform(-np.pi, np.pi, size=n)
    theta[0] = theta0
    for t in range(steps):
        k1 = kuramoto_rhs_network(theta[t], omega, K, A)
        k2 = kuramoto_rhs_network(theta[t] + 0.5*dt*k1, omega, K, A)
        k3 = kuramoto_rhs_network(theta[t] + 0.5*dt*k2, omega, K, A)
        k4 = kuramoto_rhs_network(theta[t] + dt*k3, omega, K, A)
        theta[t+1] = theta[t] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    time = np.linspace(0, T, steps + 1)
    return time, theta

# --- SHODes PINN and DNN model runner ---
def run_shodes_pinn_dnn(batch_size=1001, n_oscillators=25, model_type='genpot3', params=[1.0, 0.1, 0.1]):
    pinn_path = f'models/merged_pinn_{model_type}.keras'
    dnn_path = f'models/merged_dnn_model.keras'
    merged_pinn = tf.keras.models.load_model(pinn_path)
    dnn = tf.keras.models.load_model(dnn_path)

    # Use the n-oscillator simulation
    time_ml, theta_ml = simulate_kuramoto_nosc(
        T=100.0, dt=0.1, K=params[2], n=n_oscillators
    )

    # Prepare input data
    x_data = np.sin(theta_ml[:batch_size]).astype(np.float32)   # shape (batch_size, n_oscillators)
    t_data = time_ml[:batch_size].astype(np.float32)
    t_input = np.expand_dims(t_data, axis=1)

    # Predict with PINN
    pinn_outputs = merged_pinn.predict([t_input, x_data])

    # Feed into DNN
    params_repeated = np.tile(params, (batch_size, 1))
    dnn_input = np.concatenate([pinn_outputs, params_repeated], axis=1)
    potential_predictions = dnn.predict(dnn_input)

    return time_ml[:batch_size], pinn_outputs, potential_predictions

# --- Plotting function including DNN energy ---
def plot_benchmark_with_dnn(time_k, R_kuramoto, time_s, R_strogatz, time_ml, pinn_outputs, dnn_energy):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    # Plot order parameters
    axs[0].plot(time_k, R_kuramoto, label='Kuramoto R(t)', linewidth=2)
    axs[0].plot(time_s, R_strogatz, label='Strogatz R(t) (Small-world)', linewidth=2)
    axs[0].set_ylabel('Order Parameter R(t)')
    axs[0].legend()
    axs[0].set_title('Synchronization: Kuramoto vs Strogatz')
    axs[0].grid(True)

    # Plot all 25 oscillators
    for i in range(25):
        axs[1].plot(time_ml, pinn_outputs[:, i], '--', alpha=0.7, label=f'PINN Oscillator {i+1}')

    # Add DNN predicted energy
    axs[1].plot(time_ml, dnn_energy.squeeze(), ':', linewidth=2, label='SHODes DNN Predicted Energy')

    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude / Energy')
    axs[1].legend(ncol=3, fontsize=8)
    axs[1].set_title('SHODes PINN Dynamics and DNN Energy')
    axs[1].grid(True)

    plt.savefig('benchmark_with_dnn.png')
    plt.show()

# --- Main procedure for 100s simulation ---
def main():
    n = 25
    # Simulate Kuramoto with n oscillators
    time_k, theta_k = simulate_kuramoto_nosc(T=100.0, dt=0.1, K=0.1, n=n)
    R_kuramoto = np.array([kuramoto_order_parameter(theta_k[t]) for t in range(len(time_k))])

    # Simulate Strogatz with n oscillators
    time_s, theta_s = simulate_kuramoto_strogatz_nosc(T=100.0, dt=0.1, K=0.1, n=n, p=0.5)
    R_strogatz = np.array([kuramoto_order_parameter(theta_s[t]) for t in range(len(time_s))])

    # PINN + DNN predictions
    batch_size = len(time_k)
    time_ml, pinn_outputs, dnn_energy = run_shodes_pinn_dnn(batch_size=batch_size, n_oscillators=n)

    # Plot benchmark
    plot_benchmark_with_dnn(time_k, R_kuramoto, time_s, R_strogatz, time_ml, pinn_outputs, dnn_energy)

if __name__ == "__main__":
    main()
