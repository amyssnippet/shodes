# benchmarks/networks.py
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

def fully_connected_adjacency(n: int) -> np.ndarray:
    A = np.ones((n, n), dtype=float)
    np.fill_diagonal(A, 0.0)
    return A

def watts_strogatz_adjacency(n: int, k: int, p: float, seed: int = 42) -> np.ndarray:
    if nx is None:
        raise ImportError("networkx is required for Watts–Strogatz graphs.")
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    np.fill_diagonal(A, 0.0)
    return A

def erdos_renyi_adjacency(n: int, p: float, seed: int = 42) -> np.ndarray:
    if nx is None:
        raise ImportError("networkx is required for Erdos–Renyi graphs.")
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    np.fill_diagonal(A, 0.0)
    return A

def normalize_adjacency(A: np.ndarray, mode: str = "row-stochastic") -> np.ndarray:
    if mode == "row-stochastic":
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return A / row_sums
    elif mode == "max":
        m = A.max()
        return A / (m if m > 0 else 1.0)
    else:
        return A
