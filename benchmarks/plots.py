# benchmarks/plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
from .io_utils import ensure_dir

def plot_kuramoto_results(outdir: str, time: np.ndarray, theta_ts: np.ndarray, R_t: np.ndarray):
    ensure_dir(outdir)
    # Order parameter
    plt.figure(figsize=(6,4))
    plt.plot(time, R_t, lw=2)
    plt.xlabel("t")
    plt.ylabel("R(t)")
    plt.title("Kuramoto order parameter")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "kuramoto_Rt.png"), dpi=150)
    plt.close()

    # Phase raster
    plt.figure(figsize=(6,4))
    plt.imshow(np.mod(theta_ts.T + np.pi, 2*np.pi) - np.pi, aspect="auto", cmap="twilight", extent=[time[0], time[-1], 0, theta_ts.shape[1]])
    plt.colorbar(label="phase (rad)")
    plt.xlabel("t")
    plt.ylabel("oscillator index")
    plt.title("Kuramoto phases")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "kuramoto_phases.png"), dpi=150)
    plt.close()

def plot_comparison_R(outdir: str, curves: dict, xlabel: str = "t", ylabel: str = "metric", title: str = "Comparison"):
    ensure_dir(outdir)
    plt.figure(figsize=(6,4))
    for label, (x, y) in curves.items():
        plt.plot(x, y, lw=2, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison.png"), dpi=150)
    plt.close()
