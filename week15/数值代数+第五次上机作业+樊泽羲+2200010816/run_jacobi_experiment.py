"""Run experiments for the Threshold Jacobi method on the tridiagonal test matrix.

Saves a CSV with columns:
  n, time_s, sweeps, rotations, converged, max_eig_err, residual, ortho_err

And optionally saves simple plots if matplotlib is available.
"""
from __future__ import annotations

import os
import time
import csv
from typing import List

import numpy as np

from jacobi_method import tridiagonal_matrix, threshold_jacobi


def run_experiments(sizes: List[int], out_csv: str = "results_jacobi.csv", eps: float = 1e-10, max_sweeps: int = 500):
    results = []

    for n in sizes:
        print(f"Running n={n}")
        A = tridiagonal_matrix(n)
        t0 = time.perf_counter()
        w_j, V_j, info = threshold_jacobi(A, eps=eps, max_sweeps=max_sweeps, compute_eigenvectors=True, verbose=False)
        elapsed = info.get("time", time.perf_counter() - t0)

        # Reference eigenvalues
        w_ref, _ = np.linalg.eigh(A)

        max_eig_err = float(np.max(np.abs(w_j - w_ref)))

        # Residual: ||A V - V diag(w)||_F / ||A||_F
        residual = 0.0
        ortho_err = 0.0
        if V_j is not None:
            residual = float(np.linalg.norm(A.dot(V_j) - V_j * w_j.reshape((1, -1))) / np.linalg.norm(A))
            ortho_err = float(np.linalg.norm(V_j.T.dot(V_j) - np.eye(n)))

        results.append(
            {
                "n": n,
                "time_s": elapsed,
                "sweeps": int(info.get("sweeps", -1)),
                "rotations": int(info.get("rotations", -1)),
                "converged": bool(info.get("converged", False)),
                "max_eig_err": max_eig_err,
                "residual": residual,
                "ortho_err": ortho_err,
            }
        )

        print(
            f"  time={elapsed:.3f}s, sweeps={info.get('sweeps')}, rotations={info.get('rotations')}, converged={info.get('converged')}, max_eig_err={max_eig_err:.2e}, residual={residual:.2e}, ortho_err={ortho_err:.2e}"
        )

    # Prepare directories for results and plots
    base_dir = os.path.dirname(out_csv) or "."
    # If out_csv already lies in a 'results' directory, use it directly; otherwise
    # create a 'results' subdirectory under base_dir.
    if os.path.basename(base_dir) == "results":
        results_dir = base_dir
        out_csv_path = out_csv
    else:
        results_dir = os.path.join(base_dir, "results")
        out_csv_path = os.path.join(results_dir, os.path.basename(out_csv))

    # Place plots next to the results folder
    plots_dir = os.path.join(os.path.dirname(results_dir), "plots")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    fieldnames = ["n", "time_s", "sweeps", "rotations", "converged", "max_eig_err", "residual", "ortho_err"]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Try to make a couple of plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        ns = [r["n"] for r in results]
        times = [r["time_s"] for r in results]
        errs = [r["max_eig_err"] for r in results]
        rots = [r["rotations"] for r in results]

        plt.figure()
        plt.plot(ns, times, "-o")
        plt.xlabel("n")
        plt.ylabel("time (s)")
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "time_vs_n.png"))
        plt.close()

        plt.figure()
        plt.semilogy(ns, errs, "-o")
        plt.xlabel("n")
        plt.ylabel("max eigenvalue abs error")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "error_vs_n.png"))
        plt.close()

        plt.figure()
        plt.plot(ns, rots, "-o")
        plt.xlabel("n")
        plt.ylabel("rotations")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "rotations_vs_n.png"))
        plt.close()

        print(f"Plots saved to {out_dir}")
    except Exception:
        print("matplotlib not available or failed; skipping plots")

    print(f"CSV saved to {out_csv}")
    return results


if __name__ == "__main__":
    # Default sizes: 50..100 step 10
    sizes = list(range(50, 101, 10))
    out_csv = os.path.join(os.path.dirname(__file__), "results", "results_jacobi.csv")
    run_experiments(sizes, out_csv=out_csv)
