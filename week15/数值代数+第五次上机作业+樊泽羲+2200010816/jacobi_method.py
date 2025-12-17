"""Threshold Jacobi method and experiment helpers

Implements a Threshold Jacobi algorithm for symmetric matrices and helpers
for generating the tridiagonal test matrix.

Functions
- threshold_jacobi(A, eps=1e-10, max_sweeps=200, compute_eigenvectors=True, verbose=False)
- tridiagonal_matrix(n, diag=4.0, offdiag=1.0)

The implementation is written to be clear and robust (rather than hyper-optimized).
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np


def tridiagonal_matrix(n: int, diag: float = 4.0, offdiag: float = 1.0) -> np.ndarray:
    """Return the n-by-n symmetric tridiagonal matrix with `diag` on the main
    diagonal and `offdiag` on the super- and sub-diagonals.
    """
    A = np.zeros((n, n), dtype=float)
    if n <= 0:
        return A
    A.flat[:: n + 1] = diag  # main diagonal
    if n > 1:
        idx = np.arange(n - 1)
        A[idx, idx + 1] = offdiag
        A[idx + 1, idx] = offdiag
    return A


def _off_diagonal_frobenius_norm(A: np.ndarray) -> float:
    """Frobenius norm of the off-diagonal entries of symmetric matrix A."""
    # Use strictly upper triangle (i<j) and multiply by sqrt(2) when needed
    triu = np.triu(A, k=1)
    return math.sqrt(2.0 * np.sum(triu * triu))


def threshold_jacobi(
    A: np.ndarray,
    eps: float = 1e-10,
    max_sweeps: int = 200,
    compute_eigenvectors: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Compute all eigenvalues and (optionally) eigenvectors of a real
    symmetric matrix using the Threshold Jacobi method.

    Parameters
    - A : symmetric (n,n) array_like
    - eps : stopping tolerance on off-diagonal Frobenius norm
    - max_sweeps : maximum number of Jacobi sweeps
    - compute_eigenvectors : whether to accumulate the eigenvector matrix
    - verbose : print progress per sweep

    Returns
    - w : array of eigenvalues (sorted ascending)
    - V : eigenvector matrix (columns are eigenvectors) or None
    - info : dict with keys 'sweeps', 'rotations', 'time', 'off_norm', 'converged'
    """
    A0 = np.array(A, dtype=float, copy=True)
    if A0.shape[0] != A0.shape[1]:
        raise ValueError("A must be square")

    n = A0.shape[0]
    if n == 0:
        return np.array([]), np.empty((0, 0)), {"sweeps": 0, "rotations": 0, "time": 0.0, "converged": True}

    # Work on a copy
    A_work = A0.copy()
    V = np.eye(n, dtype=float) if compute_eigenvectors else None

    off_norm = _off_diagonal_frobenius_norm(A_work)

    # Initial threshold (simple practical choice)
    threshold = off_norm / max(1.0, float(n))

    sweeps = 0
    rotations = 0
    t0 = time.perf_counter()
    converged = False

    for sweep in range(1, max_sweeps + 1):
        sweeps = sweep
        rotated_this_sweep = 0

        # Reduce threshold gradually; after some sweeps we use zero threshold
        if sweep > 1:
            threshold *= 0.75
        if sweep > max(5, int(math.log(max(1, n), 2)) + 5):
            threshold = 0.0

        # Sweep over all p < q
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A_work[p, q]
                if abs(apq) <= threshold:
                    continue

                app = A_work[p, p]
                aqq = A_work[q, q]

                # Compute Jacobi rotation parameters (stable formula)
                # Avoid division by zero
                if apq == 0.0:
                    continue
                tau = (aqq - app) / (2.0 * apq)
                # sign convention ensures |t| <= 1/|tau| when |tau| large
                t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                # Update entries A[k,p] and A[k,q] for k != p,q
                for k in range(n):
                    if k == p or k == q:
                        continue
                    akp = A_work[k, p]
                    akq = A_work[k, q]
                    A_work[k, p] = c * akp - s * akq
                    A_work[p, k] = A_work[k, p]
                    A_work[k, q] = s * akp + c * akq
                    A_work[q, k] = A_work[k, q]

                # Update diagonal entries and zero the (p,q) and (q,p)
                A_work[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
                A_work[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
                A_work[p, q] = 0.0
                A_work[q, p] = 0.0

                # Update eigenvectors
                if compute_eigenvectors:
                    for k in range(n):
                        vkp = V[k, p]
                        vkq = V[k, q]
                        V[k, p] = c * vkp - s * vkq
                        V[k, q] = s * vkp + c * vkq

                rotated_this_sweep += 1
                rotations += 1

        off_norm = _off_diagonal_frobenius_norm(A_work)
        if verbose:
            print(f"sweep {sweep:3d}: rotations={rotated_this_sweep:6d}, off_norm={off_norm:.3e}, threshold={threshold:.3e}")

        if off_norm <= eps:
            converged = True
            break

        # If nothing was rotated in this sweep and threshold is effectively zero, stop
        if rotated_this_sweep == 0 and threshold <= eps:
            break

    total_time = time.perf_counter() - t0

    # Extract eigenvalues (diagonal) and optionally eigenvectors
    w = np.diag(A_work)
    if compute_eigenvectors:
        V_out = V
    else:
        V_out = None

    # Sort eigenvalues and eigenvectors (ascending)
    idx = np.argsort(w)
    w = w[idx]
    if V_out is not None:
        V_out = V_out[:, idx]

    info = {
        "sweeps": sweeps,
        "rotations": rotations,
        "time": total_time,
        "off_norm": float(off_norm),
        "converged": bool(converged),
    }

    return w, V_out, info


if __name__ == "__main__":
    # Simple demo if the file is run directly
    import argparse

    parser = argparse.ArgumentParser(description="Threshold Jacobi demo for tridiagonal matrices")
    parser.add_argument("-n", type=int, default=50, help="matrix size (default: 50)")
    parser.add_argument("--tol", type=float, default=1e-10, help="tolerance for off-diagonal norm")
    parser.add_argument("--max-sweeps", type=int, default=200, help="maximum number of Jacobi sweeps")
    parser.add_argument("--no-evec", dest="evec", action="store_false", help="do not compute eigenvectors")
    parser.add_argument("--verbose", action="store_true", help="print sweep progress")
    args = parser.parse_args()

    A = tridiagonal_matrix(args.n)
    w, V, info = threshold_jacobi(A, eps=args.tol, max_sweeps=args.max_sweeps, compute_eigenvectors=args.evec, verbose=args.verbose)

    print("n=", args.n)
    print("converged:", info["converged"], "sweeps:", info["sweeps"], "rotations:", info["rotations"], "time(s):", info["time"])
    print("first 10 eigenvalues:")
    print(w[: min(10, len(w))])
