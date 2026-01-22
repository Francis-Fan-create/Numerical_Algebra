"""Top-level solvers (JAX)."""

from __future__ import annotations

import jax.numpy as jnp

from .multigrid import pcg_with_vcycle, vcycle_stokes
from .stencils import apply_block, apply_gradient, apply_viscous, apply_divergence


def solve_vcycle(rhs_u: jnp.ndarray, rhs_v: jnp.ndarray, n: int, nu_pre: int, nu_post: int, coarsest_n: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    return vcycle_stokes(rhs_u, rhs_v, nu_pre, nu_post, n, coarsest_n)


def cg_velocity(rhs_u: jnp.ndarray, rhs_v: jnp.ndarray, n: int, tol: float, max_iter: int) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    u = jnp.zeros_like(rhs_u)
    v = jnp.zeros_like(rhs_v)
    au, av = apply_viscous(u, v, n)
    r_u = rhs_u - au
    r_v = rhs_v - av
    p_u = r_u
    p_v = r_v
    rho = float(jnp.sum(r_u * r_u) + jnp.sum(r_v * r_v))
    rhs_norm = float(jnp.sqrt(jnp.sum(rhs_u**2) + jnp.sum(rhs_v**2)))
    k = 0

    while jnp.sqrt(rho) > tol * rhs_norm and k < max_iter:
        ap_u, ap_v = apply_viscous(p_u, p_v, n)
        alpha = rho / float(jnp.sum(p_u * ap_u) + jnp.sum(p_v * ap_v))
        u = u + alpha * p_u
        v = v + alpha * p_v
        r_u = r_u - alpha * ap_u
        r_v = r_v - alpha * ap_v
        rho_new = float(jnp.sum(r_u * r_u) + jnp.sum(r_v * r_v))
        beta = rho_new / rho
        p_u = r_u + beta * p_u
        p_v = r_v + beta * p_v
        rho = rho_new
        k += 1

    return u, v, k


def solve_uzawa(rhs_u: jnp.ndarray, rhs_v: jnp.ndarray, n: int, alpha: float, tol: float, max_iter: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    p = jnp.zeros((n, n), dtype=jnp.float64)
    u = jnp.zeros((n + 1, n), dtype=jnp.float64)
    v = jnp.zeros((n, n + 1), dtype=jnp.float64)
    ref_norm = float(jnp.sqrt(jnp.sum(rhs_u**2) + jnp.sum(rhs_v**2)))
    iters = 0

    while True:
        iters += 1
        gx, gy = apply_gradient(p, n)
        u, v, _ = cg_velocity(rhs_u - gx, rhs_v - gy, n, tol, max_iter)
        p = p + alpha * apply_divergence(u, v, n)
        au, av, div = apply_block(u, v, p, n)
        res = float(jnp.sqrt(jnp.sum((rhs_u - au) ** 2) + jnp.sum((rhs_v - av) ** 2) + jnp.sum(div**2)))
        if res / ref_norm < 1.0e-8:
            break

    return u, v, p, iters


def solve_inexact_uzawa(
    rhs_u: jnp.ndarray,
    rhs_v: jnp.ndarray,
    n: int,
    alpha: float,
    tau: float,
    nu_pre: int,
    nu_post: int,
    coarsest_n: int,
    max_iter: int,
    tol: float,
    vcycle_tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, list[int]]:
    p = jnp.zeros((n, n), dtype=jnp.float64)
    u = jnp.zeros((n + 1, n), dtype=jnp.float64)
    v = jnp.zeros((n, n + 1), dtype=jnp.float64)
    ref_norm = float(jnp.sqrt(jnp.sum(rhs_u**2) + jnp.sum(rhs_v**2)))
    iters = 0
    pcg_iters_log: list[int] = []

    while True:
        iters += 1
        divergence_bound = tau * float(jnp.linalg.norm(apply_divergence(u, v, n)))
        gx, gy = apply_gradient(p, n)
        u, v, pcg_iters, _pcg_hist = pcg_with_vcycle(
            rhs_u - gx,
            rhs_v - gy,
            n,
            max_iter,
            tol,
            vcycle_tol,
            nu_pre,
            nu_post,
            coarsest_n,
            divergence_bound,
        )
        pcg_iters_log.append(int(pcg_iters))
        p = p + alpha * apply_divergence(u, v, n)
        au, av, div = apply_block(u, v, p, n)
        res = float(jnp.sqrt(jnp.sum((rhs_u - au) ** 2) + jnp.sum((rhs_v - av) ** 2) + jnp.sum(div**2)))
        if res / ref_norm < 1.0e-8:
            break

    return u, v, p, iters, pcg_iters_log
