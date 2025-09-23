"""resource_logger.py
Simple utilities to log wall‑clock runtime, CPU/GPU usage, and disk usage to CSV.
Usage:

from resource_logger import ResourceLogger
with ResourceLogger(tag="clinicalbert_lstm"):
    train()

The logger appends one row per run to <BASE>/resource_usage.csv.
"""
import csv, os, time, psutil, shutil, torch
from contextlib import contextmanager

BASE = "./"
LOG_PATH = os.path.join(BASE, "resource_usage.csv")
os.makedirs(BASE, exist_ok=True)

@contextmanager
def ResourceLogger(tag: str):
    start = time.time()
    yield  # code block executes here
    end = time.time()

    # wall time (hours)
    elapsed_hr = (end - start) / 3600

    # CPU percent snapshot
    cpu_pct = psutil.cpu_percent(interval=None)

    # GPU hours (simple estimate: 1 GPU * elapsed_hr if CUDA present)
    gpu_hrs = elapsed_hr if torch.cuda.is_available() else 0.0

    # Disk used (GB) after run
    _, used, _ = shutil.disk_usage("/")
    used_gb = used / 1e9

    # Append to CSV
    header = ["tag", "elapsed_hr", "gpu_hrs", "cpu_pct", "disk_used_gb"]
    row    = [tag, f"{elapsed_hr:.3f}", f"{gpu_hrs:.3f}", f"{cpu_pct:.1f}", f"{used_gb:.1f}"]

    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"[ResourceLogger] logged resources for {tag} → {LOG_PATH}")