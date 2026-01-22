"""CLI runner for Stokes multigrid experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from stokes_fast.fields import rhs_terms as build_rhs
from stokes_fast.fields import velocity_error
from stokes_fast.solvers import solve_inexact_uzawa, solve_uzawa, solve_vcycle


def parse_int_list(values: list[str]) -> list[int]:
    result = []
    for item in values:
        for part in item.split(","):
            part = part.strip()
            if part:
                result.append(int(part))
    return result


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = []
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def block_until_ready(*arrays) -> None:
    import jax

    for arr in arrays:
        if arr is not None:
            jax.block_until_ready(arr)


def compute_error_orders(ns: list[int], errors: list[float]) -> list[float]:
    orders = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i - 1] > 0 and ns[i] != ns[i - 1]:
            orders.append(math.log(errors[i - 1] / errors[i]) / math.log(ns[i] / ns[i - 1]))
        else:
            orders.append(float("nan"))
    return orders


def save_json(path: str | Path, payload: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_problem1(ns: Iterable[int], v1: int, v2: int, bottom: int) -> tuple[list[dict], list[float]]:
    headers = ["N", "Wall time (s)", "V-cycle count", "error"]
    rows: list[list[str]] = []
    results: list[dict] = []
    errors: list[float] = []
    for n in ns:
        F_U, F_V = build_rhs(n)
        block_until_ready(F_U, F_V)
        start = time.perf_counter()
        U, V, P, cycles = solve_vcycle(F_U, F_V, n, v1, v2, bottom)
        block_until_ready(U, V, P)
        elapsed = time.perf_counter() - start
        err = velocity_error(U, V, n)
        errors.append(err)
        results.append({"n": int(n), "wall_time": float(elapsed), "vcycle_count": int(cycles), "error": float(err)})
        rows.append([str(n), f"{elapsed:.6f}", str(cycles), f"{err:.6e}"])
    print(format_table(headers, rows))
    return results, errors


def run_problem2(ns: Iterable[int], alpha: float, epsilon: float, k_max: int) -> tuple[list[dict], list[float]]:
    headers = ["N", "Wall time (s)", "Uzawa iters", "error"]
    rows: list[list[str]] = []
    results: list[dict] = []
    errors: list[float] = []
    for n in ns:
        F_U, F_V = build_rhs(n)
        block_until_ready(F_U, F_V)
        start = time.perf_counter()
        U, V, P, iters = solve_uzawa(F_U, F_V, n, alpha=alpha, tol=epsilon, max_iter=k_max)
        block_until_ready(U, V, P)
        elapsed = time.perf_counter() - start
        err = velocity_error(U, V, n)
        errors.append(err)
        results.append({"n": int(n), "wall_time": float(elapsed), "uzawa_iters": int(iters), "error": float(err)})
        rows.append([str(n), f"{elapsed:.6f}", str(iters), f"{err:.6e}"])
    print(format_table(headers, rows))
    return results, errors


def run_problem3(
    ns: Iterable[int],
    alpha: float,
    tau: float,
    v1: int,
    v2: int,
    bottom: int,
    epsilon: float,
    k_max: int,
    error: float,
) -> tuple[list[dict], list[float]]:
    headers = ["N", "Wall time (s)", "Outer iters", "PCG iters", "error"]
    rows: list[list[str]] = []
    results: list[dict] = []
    errors: list[float] = []
    for n in ns:
        F_U, F_V = build_rhs(n)
        block_until_ready(F_U, F_V)
        start = time.perf_counter()
        U, V, P, iters, pcg_iters = solve_inexact_uzawa(
            F_U,
            F_V,
            n,
            alpha=alpha,
            tau=tau,
            nu_pre=v1,
            nu_post=v2,
            coarsest_n=bottom,
            max_iter=k_max,
            tol=epsilon,
            vcycle_tol=error,
        )
        block_until_ready(U, V, P)
        elapsed = time.perf_counter() - start
        err = velocity_error(U, V, n)
        errors.append(err)
        pcg_iters_str = ",".join(str(x) for x in pcg_iters)
        results.append(
            {
                "n": int(n),
                "wall_time": float(elapsed),
                "outer_iters": int(iters),
                "pcg_iters": [int(x) for x in pcg_iters],
                "error": float(err),
            }
        )
        rows.append([str(n), f"{elapsed:.6f}", str(iters), pcg_iters_str, f"{err:.6e}"])
    print(format_table(headers, rows))
    return results, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Stokes multigrid experiments (Python translation).")
    parser.add_argument("--problem", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--ns", nargs="+", default=None, help="List of N values (comma or space separated).")
    parser.add_argument("--v1", type=int, default=4)
    parser.add_argument("--v2", type=int, default=4)
    parser.add_argument("--bottom", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0e-3)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--pcg-error", type=float, default=1.0e-3, help="Tolerance for V-cycle preconditioner in PCG.")
    parser.add_argument("--progress", action="store_true", help="Show a progress bar over N values (requires tqdm).")
    parser.add_argument("--warmup", action="store_true", help="Run a tiny case to trigger JAX JIT before timing.")
    parser.add_argument("--json", type=str, default=None, help="Write full results to a JSON file.")
    args = parser.parse_args()

    if args.ns is None:
        if args.problem == 2:
            ns = [64, 128, 256, 512]
        else:
            ns = [64, 128, 256, 512, 1024, 2048]
    else:
        ns = parse_int_list(args.ns)

    ns_list = list(ns)
    if args.progress and tqdm is not None:
        ns = tqdm(ns_list, desc="N")
    else:
        ns = ns_list

    if args.warmup:
        warm_n = 8
        F_U, F_V = build_rhs(warm_n)
        if args.problem == 1:
            warm = solve_vcycle(F_U, F_V, warm_n, max(1, args.v1), max(1, args.v2), min(args.bottom, warm_n // 2))
            block_until_ready(*warm[:3])
        elif args.problem == 2:
            warm = solve_uzawa(F_U, F_V, warm_n, alpha=args.alpha, tol=args.epsilon, max_iter=10)
            block_until_ready(*warm[:3])
        else:
            warm = solve_inexact_uzawa(
                F_U,
                F_V,
                warm_n,
                alpha=args.alpha,
                tau=args.tau,
                nu_pre=max(1, args.v1),
                nu_post=max(1, args.v2),
                coarsest_n=min(args.bottom, warm_n // 2),
                max_iter=10,
                tol=args.epsilon,
                vcycle_tol=args.pcg_error,
            )
            block_until_ready(*warm[:3])

    if args.problem == 1:
        results, errors = run_problem1(ns, args.v1, args.v2, args.bottom)
        orders = compute_error_orders(ns_list, errors)
        if orders:
            print("Observed error orders:", ", ".join(f"{o:.4f}" for o in orders))
        payload = {
            "problem": 1,
            "parameters": {"v1": args.v1, "v2": args.v2, "bottom": args.bottom},
            "ns": ns_list,
            "results": results,
            "error_orders": orders,
        }
    elif args.problem == 2:
        results, errors = run_problem2(ns, args.alpha, args.epsilon, args.k_max)
        orders = compute_error_orders(ns_list, errors)
        if orders:
            print("Observed error orders:", ", ".join(f"{o:.4f}" for o in orders))
        payload = {
            "problem": 2,
            "parameters": {"alpha": args.alpha, "epsilon": args.epsilon, "k_max": args.k_max},
            "ns": ns_list,
            "results": results,
            "error_orders": orders,
        }
    else:
        results, errors = run_problem3(ns, args.alpha, args.tau, args.v1, args.v2, args.bottom, args.epsilon, args.k_max, args.pcg_error)
        orders = compute_error_orders(ns_list, errors)
        if orders:
            print("Observed error orders:", ", ".join(f"{o:.4f}" for o in orders))
        payload = {
            "problem": 3,
            "parameters": {
                "alpha": args.alpha,
                "tau": args.tau,
                "v1": args.v1,
                "v2": args.v2,
                "bottom": args.bottom,
                "epsilon": args.epsilon,
                "k_max": args.k_max,
                "pcg_error": args.pcg_error,
            },
            "ns": ns_list,
            "results": results,
            "error_orders": orders,
        }

    if args.json:
        save_json(args.json, payload)


if __name__ == "__main__":
    main()
