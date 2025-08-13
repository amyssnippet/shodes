import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

# ---- Kuramoto global coupling (classic) ----

def kuramoto_rhs(theta, omega, K):
    diff = theta[None, :] - theta[:, None]
    coupling = np.sum(np.sin(diff), axis=1) - np.sin(0)
    return omega + (K / 2) * coupling

def simulate_kuramoto_2osc(T=10.0, dt=0.1, K=0.1, omega=None, theta0=None):
    steps = int(T / dt)
    theta = np.zeros((steps + 1, 2))
    if omega is None:
        omega = np.array([1.0, 1.1])
    if theta0 is None:
        theta0 = np.random.uniform(-np.pi, np.pi, size=2)
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

# ---- Strogatz: Watts-Strogatz small-world network ----

def watts_strogatz_adjacency(n, k=1, p=0.5, seed=42):
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    np.fill_diagonal(A, 0.0)
    return A

def kuramoto_rhs_network(theta, omega, K, A):
    diff = theta[None, :] - theta[:, None]
    coupling = (A * np.sin(diff)).sum(axis=1)
    return omega + (K / max(1, A.shape[0])) * coupling

def simulate_kuramoto_strogatz_2osc(T=10.0, dt=0.1, K=0.1, omega=None, theta0=None, p=0.5):
    n = 2
    A = watts_strogatz_adjacency(n=n, k=1, p=p, seed=42)
    steps = int(T / dt)
    theta = np.zeros((steps + 1, n))
    if omega is None:
        omega = np.array([1.0, 1.1])
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

# ---- SHODes ML Model runner section ----

def run_shodes_pinn_dnn(batch_size=100, n_oscillators=2, model_type='genpot3', params=[1.0, 0.1, 0.1]):
    pinn_path = f'models/merged_pinn_{model_type}.keras'
    dnn_path = f'models/merged_dnn_model.keras'
    merged_pinn = tf.keras.models.load_model(pinn_path)
    dnn = tf.keras.models.load_model(dnn_path)
    # Use Kuramoto's phases as dummy displacements for ML model
    time_ml, theta_ml = simulate_kuramoto_2osc(T=10, dt=0.1, K=params[2])
    x_data = np.sin(theta_ml[:batch_size]).astype(np.float32) # Shape: (batch_size, n_oscillators)
    t_data = time_ml[:batch_size].astype(np.float32)
    t_input = np.expand_dims(t_data, axis=1)
    pinn_outputs = merged_pinn.predict([t_input, x_data])
    params_repeated = np.tile(params, (batch_size, 1))
    dnn_input = np.concatenate([pinn_outputs, params_repeated], axis=1)
    potential_predictions = dnn.predict(dnn_input)
    return time_ml[:batch_size], pinn_outputs, potential_predictions

# ---- Main: run all and plot ----

def main():
    # Run Kuramoto (global coupling)
    time_k, theta_k = simulate_kuramoto_2osc(T=10, dt=0.1, K=0.1)
    R_kuramoto = np.array([kuramoto_order_parameter(theta_k[t]) for t in range(len(time_k))])

    # Run Strogatz (Watts-Strogatz small-world)
    time_s, theta_s = simulate_kuramoto_strogatz_2osc(T=10, dt=0.1, K=0.1, p=0.5)
    R_strogatz = np.array([kuramoto_order_parameter(theta_s[t]) for t in range(len(time_s))])

    # Run SHODes model
    batch_size = 100
    time_ml, pinn_outputs, potential_predictions = run_shodes_pinn_dnn(batch_size=batch_size)

    # Plot all curves
    plt.figure(figsize=(10,6))
    plt.plot(time_k[:batch_size], R_kuramoto[:batch_size], label='Kuramoto R(t)', lw=2)
    plt.plot(time_s[:batch_size], R_strogatz[:batch_size], label='Strogatz R(t) (Small-world)', lw=2)
    plt.plot(time_ml, pinn_outputs[:, 0], label='SHODes PINN Osc1', lw=2, ls='--')
    plt.plot(time_ml, pinn_outputs[:, 1], label='SHODes PINN Osc2', lw=2, ls='--')
    plt.xlabel("Time")
    plt.ylabel("Synchronization / Output")
    plt.title("Kuramoto vs Strogatz vs SHODes ML Benchmark")
    plt.legend()
    plt.grid(True)
    plt.savefig('benchmark3.png')
    plt.show()

if __name__ == "__main__":
    main()
