# benchmarks/scenarios.py
import numpy as np
from dataclasses import dataclass
from .networks import fully_connected_adjacency, watts_strogatz_adjacency, normalize_adjacency

@dataclass
class Scenario:
    name: str
    N: int
    T: float
    dt: float
    topology: str
    topo_params: dict
    coupling: float
    omega_mu: float = 0.0
    omega_sigma: float = 1.0

def build_adjacency(s: Scenario) -> np.ndarray:
    if s.topology == "complete":
        A = fully_connected_adjacency(s.N)
    elif s.topology == "watts-strogatz":
        A = watts_strogatz_adjacency(n=s.N, k=s.topo_params.get("k", 6), p=s.topo_params.get("p", 0.1),
                                     seed=s.topo_params.get("seed", 42))
    else:
        raise ValueError(f"Unknown topology {s.topology}")
    return normalize_adjacency(A, mode="row-stochastic")

def default_scenarios():
    return [
        Scenario(name="S1_complete_weak",
                 N=50, T=20.0, dt=0.01,
                 topology="complete",
                 topo_params={},
                 coupling=0.5,
                 omega_mu=0.0, omega_sigma=1.0),
        Scenario(name="S2_ws_medium",
                 N=100, T=20.0, dt=0.01,
                 topology="watts-strogatz",
                 topo_params={"k": 8, "p": 0.2, "seed": 42},
                 coupling=1.5,
                 omega_mu=0.0, omega_sigma=1.0),
        Scenario(name="S3_ws_strong",
                 N=200, T=20.0, dt=0.005,
                 topology="watts-strogatz",
                 topo_params={"k": 10, "p": 0.3, "seed": 7},
                 coupling=3.0,
                 omega_mu=0.0, omega_sigma=0.5),
    ]
