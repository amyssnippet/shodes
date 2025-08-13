# benchmarks/io_utils.py
import os
import csv
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_time_series_csv(path: str, time: np.ndarray, series: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        headers = ["t"] + list(series.keys())
        writer.writerow(headers)
        for i, t in enumerate(time):
            row = [t] + [float(series[k][i]) for k in series]
            writer.writerow(row)
