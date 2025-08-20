# benchmarks/runner.py
import os
import numpy as np

from .scenarios import default_scenarios, build_adjacency
from .kuramoto import simulate_kuramoto, sample_omegas
from .metrics import kuramoto_order_parameter_over_time
from .io_utils import save_time_series_csv, ensure_dir
from .plots import plot_kuramoto_results, plot_comparison_R

# Stubs to integrate with existing SHODes codebase
def run_shodes_pipeline(potential: str, outdir: str, config: dict):
    """
    Expected to:
    - Generate data for the potential (mecpot, genpot1, genpot2, genpot3)
    - Train PINN and DNN
    - Save CSV logs: e.g., time series of energy or PINN loss
    - Return dict with paths to metrics for plotting
    Adapt to actual module names/functions in your repo.
    """
    # Example: os.system(f"python -m shodes.scripts.run --potential {potential} --outdir {outdir}")
    # For now, assume logs are saved and return paths.
    return {
        "energy_csv": os.path.join(outdir, f"{potential}_energy.csv"),
        "loss_csv": os.path.join(outdir, f"{potential}_loss.csv"),
    }

def load_single_column_csv(path: str, t_col: int = 0, y_col: int = 1):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, t_col], data[:, y_col]

def run_kuramoto_benchmark(s, outroot: str):
    A = build_adjacency(s)
    omega = sample_omegas(s.N, mu=s.omega_mu, sigma=s.omega_sigma, seed=42)
    time, theta_ts = simulate_kuramoto(N=s.N, T=s.T, dt=s.dt, omega=omega, K=s.coupling, A=A, method="rk4")
    R_t = kuramoto_order_parameter_over_time(theta_ts)

    outdir = os.path.join(outroot, s.name, "kuramoto")
    ensure_dir(outdir)
    save_time_series_csv(os.path.join(outdir, "kuramoto_Rt.csv"), time, {"R": R_t})
    plot_kuramoto_results(outdir, time, theta_ts, R_t)
    return time, R_t, outdir

def run_shodes_benchmark(s, outroot: str, potentials=("mecpot", "genpot1", "genpot2", "genpot3")):
    results = {}
    for pot in potentials:
        outdir = os.path.join(outroot, s.name, f"shodes_{pot}")
        ensure_dir(outdir)
        meta = run_shodes_pipeline(pot, outdir, config={"N": s.N, "T": s.T, "dt": s.dt})
        # If energy_csv available, load and return for comparison
        if os.path.exists(meta["energy_csv"]):
            t, e = load_single_column_csv(meta["energy_csv"])
            results[pot] = (t, e)
    return results

def main(outroot: str = "benchmarks_out"):
    scenarios = default_scenarios()
    for s in scenarios:
        # Kuramoto
        t_k, R_t, k_dir = run_kuramoto_benchmark(s, outroot)

        # SHODes runs per potential
        shodes_curves = run_shodes_benchmark(s, outroot)
        # Plot comparison: Kuramoto R(t) vs. any SHO metric (example compares R(t) to normalized energy if desired)
        curves = {"Kuramoto R(t)": (t_k, R_t)}
        curves.update({f"SHO {k}": v for k, v in shodes_curves.items()})
        plot_comparison_R(os.path.join(outroot, s.name), curves, xlabel="t", ylabel="metric", title=f"Scenario {s.name} Comparison")

if __name__ == "__main__":
    main()
