"""Multigrid cycles and PCG preconditioning (reorganized)."""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import jax.numpy as jnp

from .coarse import coarse_stokes_solve, coarse_velocity_solve
from .coarsen import prolong_all, prolong_velocity, restrict_all, restrict_velocity
from .relax import distributive_relax, smooth_velocity
from .stencils import apply_block, apply_viscous


def vcycle_stokes(
    rhs_u: jnp.ndarray,
    rhs_v: jnp.ndarray,
    nu_pre: int,
    nu_post: int,
    finest_n: int,
    coarsest_n: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    levels = int(round(math.log(finest_n / coarsest_n, 2)))
    n = finest_n

    u_levels: List[jnp.ndarray] = [None] * (levels + 1)
    v_levels: List[jnp.ndarray] = [None] * (levels + 1)
    p_levels: List[jnp.ndarray] = [None] * (levels + 1)
    rhs_u_levels: List[jnp.ndarray] = [None] * levels
    rhs_v_levels: List[jnp.ndarray] = [None] * levels
    rhs_div_levels: List[jnp.ndarray] = [None] * levels

    u_levels[0] = jnp.zeros((n + 1, n), dtype=jnp.float64)
    v_levels[0] = jnp.zeros((n, n + 1), dtype=jnp.float64)
    p_levels[0] = jnp.zeros((n, n), dtype=jnp.float64)
    rhs_u_levels[0] = rhs_u
    rhs_v_levels[0] = rhs_v
    rhs_div_levels[0] = jnp.zeros((n, n), dtype=jnp.float64)

    ref_norm = float(jnp.sqrt(jnp.sum(rhs_u**2) + jnp.sum(rhs_v**2)))
    cycles = 0

    while True:
        cycles += 1
        for k in range(levels + 1):
            if k != 0:
                u = jnp.zeros((n + 1, n), dtype=jnp.float64)
                v = jnp.zeros((n, n + 1), dtype=jnp.float64)
                p = jnp.zeros((n, n), dtype=jnp.float64)
            else:
                u = u_levels[0]
                v = v_levels[0]
                p = p_levels[0]
                rhs_u = rhs_u_levels[0]
                rhs_v = rhs_v_levels[0]
                rhs_div = rhs_div_levels[0]

            if k != levels:
                for _ in range(nu_pre):
                    u, v, p = distributive_relax(u, v, p, rhs_u, rhs_v, rhs_div, n)
            else:
                u, v, p = coarse_stokes_solve(rhs_u, rhs_v, n)

            u_levels[k] = u
            v_levels[k] = v
            p_levels[k] = p
            if k != 0 and k != levels:
                rhs_u_levels[k] = rhs_u
                rhs_v_levels[k] = rhs_v
                rhs_div_levels[k] = rhs_div

            if k != levels:
                au, av, div = apply_block(u, v, p, n)
                res_u = rhs_u - au
                res_v = rhs_v - av
                res_div = rhs_div - div
                rhs_u, rhs_v, rhs_div = restrict_all(res_u, res_v, res_div, n)
                n //= 2

        for up in range(levels):
            k = levels - 1 - up
            u, v, p = prolong_all(u_levels[k + 1], v_levels[k + 1], p_levels[k + 1], n)
            n *= 2
            u = u_levels[k] + u
            v = v_levels[k] + v
            p = p_levels[k] + p
            for _ in range(nu_post):
                u, v, p = distributive_relax(u, v, p, rhs_u_levels[k], rhs_v_levels[k], rhs_div_levels[k], n)
            u_levels[k] = u
            v_levels[k] = v
            p_levels[k] = p

        au, av, div = apply_block(u, v, p, n)
        res = float(jnp.sqrt(jnp.sum((au - rhs_u_levels[0]) ** 2) + jnp.sum((av - rhs_v_levels[0]) ** 2) + jnp.sum(div**2)))
        if res / ref_norm < 1.0e-8:
            break

    return u, v, p, cycles


def vcycle_velocity(
    rhs_u: jnp.ndarray,
    rhs_v: jnp.ndarray,
    nu_pre: int,
    nu_post: int,
    tol: float,
    finest_n: int,
    coarsest_n: int,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    levels = int(round(math.log(finest_n / coarsest_n, 2)))
    n = finest_n

    u_levels: List[jnp.ndarray] = [None] * (levels + 1)
    v_levels: List[jnp.ndarray] = [None] * (levels + 1)
    rhs_u_levels: List[jnp.ndarray] = [None] * levels
    rhs_v_levels: List[jnp.ndarray] = [None] * levels

    u_levels[0] = jnp.zeros((n + 1, n), dtype=jnp.float64)
    v_levels[0] = jnp.zeros((n, n + 1), dtype=jnp.float64)
    rhs_u_levels[0] = rhs_u
    rhs_v_levels[0] = rhs_v

    ref_norm = float(jnp.sqrt(jnp.sum(rhs_u**2) + jnp.sum(rhs_v**2)))
    cycles = 0

    while True:
        cycles += 1
        for k in range(levels + 1):
            if k != 0:
                u = jnp.zeros((n + 1, n), dtype=jnp.float64)
                v = jnp.zeros((n, n + 1), dtype=jnp.float64)
            else:
                u = u_levels[0]
                v = v_levels[0]
                rhs_u = rhs_u_levels[0]
                rhs_v = rhs_v_levels[0]

            if k != levels:
                for _ in range(nu_pre):
                    u, v = smooth_velocity(u, v, rhs_u, rhs_v, n)
            else:
                u, v = coarse_velocity_solve(rhs_u, rhs_v, n)

            u_levels[k] = u
            v_levels[k] = v
            if k != 0 and k != levels:
                rhs_u_levels[k] = rhs_u
                rhs_v_levels[k] = rhs_v

            if k != levels:
                au, av = apply_viscous(u, v, n)
                res_u = rhs_u - au
                res_v = rhs_v - av
                rhs_u, rhs_v = restrict_velocity(res_u, res_v, n)
                n //= 2

        for up in range(levels):
            k = levels - 1 - up
            u, v = prolong_velocity(u_levels[k + 1], v_levels[k + 1], n)
            n *= 2
            u = u_levels[k] + u
            v = v_levels[k] + v
            for _ in range(nu_post):
                u, v = smooth_velocity(u, v, rhs_u_levels[k], rhs_v_levels[k], n)
            u_levels[k] = u
            v_levels[k] = v

        au, av = apply_viscous(u, v, n)
        res = float(jnp.sqrt(jnp.sum((au - rhs_u_levels[0]) ** 2) + jnp.sum((av - rhs_v_levels[0]) ** 2)))
        if res / ref_norm < tol:
            break

    return u, v, cycles


def pcg_with_vcycle(
    rhs_u: jnp.ndarray,
    rhs_v: jnp.ndarray,
    n: int,
    max_iter: int,
    eps: float,
    vcycle_tol: float,
    nu_pre: int,
    nu_post: int,
    coarsest_n: int,
    divergence_tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray, int, jnp.ndarray]:
    u = jnp.zeros_like(rhs_u)
    v = jnp.zeros_like(rhs_v)
    r_u = rhs_u
    r_v = rhs_v
    rhs_norm = float(jnp.sqrt(jnp.sum(rhs_u**2) + jnp.sum(rhs_v**2)))

    p_u = None
    p_v = None
    rho = None
    hist = []
    iters = 0

    while float(jnp.sqrt(jnp.sum(r_u**2) + jnp.sum(r_v**2))) > max(eps * rhs_norm, divergence_tol) and iters < max_iter:
        t0 = time.perf_counter()
        z_u, z_v, vcycles = vcycle_velocity(r_u, r_v, nu_pre, nu_post, vcycle_tol, n, coarsest_n)
        t1 = time.perf_counter()
        hist.append((vcycles, t1 - t0))

        if iters == 0:
            p_u, p_v = z_u, z_v
            rho = float(jnp.sum(r_u * z_u) + jnp.sum(r_v * z_v))
        else:
            rho_prev = rho
            rho = float(jnp.sum(r_u * z_u) + jnp.sum(r_v * z_v))
            beta = rho / rho_prev
            p_u = z_u + beta * p_u
            p_v = z_v + beta * p_v

        ap_u, ap_v = apply_viscous(p_u, p_v, n)
        denom = float(jnp.sum(p_u * ap_u) + jnp.sum(p_v * ap_v))
        alpha = rho / denom
        u = u + alpha * p_u
        v = v + alpha * p_v
        r_u = r_u - alpha * ap_u
        r_v = r_v - alpha * ap_v
        iters += 1

    hist_arr = jnp.array(hist, dtype=jnp.float64).T if hist else jnp.zeros((2, 0), dtype=jnp.float64)
    return u, v, iters, hist_arr
