# benchmarks/metrics.py
import numpy as np

def kuramoto_order_parameter(theta: np.ndarray) -> float:
    # theta: (N,) phases at a time t
    z = np.exp(1j * theta).mean()
    return np.abs(z)

def kuramoto_order_parameter_over_time(theta_ts: np.ndarray) -> np.ndarray:
    # theta_ts: (T, N)
    return np.apply_along_axis(kuramoto_order_parameter, 1, theta_ts)

def mean_frequency_sync(theta_ts: np.ndarray, dt: float) -> float:
    # proxy for phase-locking: variance of instantaneous freq across oscillators
    dtheta = np.diff(theta_ts, axis=0) / dt
    return float(np.mean(np.var(dtheta, axis=1)))

def energy_from_sho(x: np.ndarray, v: np.ndarray, m: float, k: float) -> float:
    # x, v: (N,)
    return 0.5 * m * np.sum(v**2) + 0.5 * k * np.sum(x**2)
