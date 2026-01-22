# Stokes multigrid (Python)

This folder provides an accelerated Python implementation using **JAX** with a reorganized code layout, suitable for GPU execution.

## Setup

Install dependencies:

- `jax`
- `tqdm` (optional, for progress bars)

For NVIDIA GPU usage, install a CUDA-enabled `jaxlib` build that matches your local CUDA/CuDNN versions.

## Usage

Run a problem with default settings:

- Problem 1 (V-cycle DGS): `python main.py --problem 1`
- Problem 2 (Uzawa): `python main.py --problem 2`
- Problem 3 (Inexact Uzawa + V-cycle PCG): `python main.py --problem 3`

Override parameters (examples):

- `python main.py --problem 1 --v1 6 --v2 6 --bottom 2 --ns 64 128 256 --progress --warmup`
- `python main.py --problem 2 --alpha 1 --ns 64,128,256,512`
- `python main.py --problem 3 --alpha 1 --tau 1e-3 --v1 4 --v2 4 --bottom 4 --pcg-error 1e-3`
- Save full results to JSON: `python main.py --problem 1 --json results_p1.json`

Large `N` values can be expensive. JAX will JIT-compile kernels on first use, so the first run may be slower.
Problem 3 prints per-outer-step PCG iteration counts (comma-separated) and the JSON output includes them as arrays.
