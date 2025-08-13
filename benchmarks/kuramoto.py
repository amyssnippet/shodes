# benchmarks/kuramoto.py
import numpy as np
from typing import Tuple, Dict

def kuramoto_rhs(theta: np.ndarray, omega: np.ndarray, K: float, A: np.ndarray) -> np.ndarray:
    # A: adjacency/coupling matrix (N,N)
    # pairwise sin(theta_j - theta_i)
    diff = theta[None, :] - theta[:, None]  # (N,N)
    coupling = (A * np.sin(diff)).sum(axis=1)
    return omega + (K / max(1, A.shape[0])) * coupling

def simulate_kuramoto(N: int,
                      T: float,
                      dt: float,
                      omega: np.ndarray,
                      K: float,
                      A: np.ndarray,
                      theta0: np.ndarray = None,
                      method: str = "rk4") -> Tuple[np.ndarray, np.ndarray]:
    steps = int(np.round(T / dt))
    theta = np.zeros((steps + 1, N), dtype=float)
    if theta0 is None:
        theta[0] = np.random.uniform(-np.pi, np.pi, size=N)
    else:
        theta[0] = theta0

    def step(x):
        if method == "euler":
            return x + dt * kuramoto_rhs(x, omega, K, A)
        elif method == "rk4":
            k1 = kuramoto_rhs(x, omega, K, A)
            k2 = kuramoto_rhs(x + 0.5 * dt * k1, omega, K, A)
            k3 = kuramoto_rhs(x + 0.5 * dt * k2, omega, K, A)
            k4 = kuramoto_rhs(x + dt * k3, omega, K, A)
            return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError("Unknown method")

    for t in range(steps):
        theta[t + 1] = step(theta[t])

    time = np.linspace(0.0, T, steps + 1)
    return time, theta

def sample_omegas(N: int, mu: float = 0.0, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=N)
