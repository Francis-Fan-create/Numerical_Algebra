"""Relaxation schemes for multigrid (vectorized, JAX-friendly)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .stencils import apply_viscous, apply_gradient, apply_divergence


def _jacobi_velocity(ux: jnp.ndarray, vy: jnp.ndarray, rhs_u: jnp.ndarray, rhs_v: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    h = 1.0 / n
    out_u = ux
    out_v = vy

    out_u = out_u.at[1:n, 1:n - 1].set(
        (ux[1:n, 2:n] + ux[1:n, 0:n - 2] + ux[0:n - 1, 1:n - 1] + ux[2:n + 1, 1:n - 1] + h**2 * rhs_u[1:n, 1:n - 1])
        / 4.0
    )
    out_u = out_u.at[1:n, 0].set(
        (ux[0:n - 1, 0] + ux[2:n + 1, 0] + ux[1:n, 1] + h**2 * rhs_u[1:n, 0]) / 3.0
    )
    out_u = out_u.at[1:n, n - 1].set(
        (ux[0:n - 1, n - 1] + ux[2:n + 1, n - 1] + ux[1:n, n - 2] + h**2 * rhs_u[1:n, n - 1]) / 3.0
    )

    out_v = out_v.at[1:n - 1, 1:n].set(
        (vy[1:n - 1, 2:n + 1] + vy[1:n - 1, 0:n - 1] + vy[0:n - 2, 1:n] + vy[2:n, 1:n] + h**2 * rhs_v[1:n - 1, 1:n])
        / 4.0
    )
    out_v = out_v.at[0, 1:n].set(
        (vy[1, 1:n] + vy[0, 2:n + 1] + vy[0, 0:n - 1] + h**2 * rhs_v[0, 1:n]) / 3.0
    )
    out_v = out_v.at[n - 1, 1:n].set(
        (vy[n - 2, 1:n] + vy[n - 1, 2:n + 1] + vy[n - 1, 0:n - 1] + h**2 * rhs_v[n - 1, 1:n]) / 3.0
    )

    return out_u, out_v


@partial(jax.jit, static_argnums=(6,))
def distributive_relax(
    ux: jnp.ndarray,
    vy: jnp.ndarray,
    p: jnp.ndarray,
    rhs_u: jnp.ndarray,
    rhs_v: jnp.ndarray,
    rhs_div: jnp.ndarray,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h = 1.0 / n
    au, av = apply_viscous(ux, vy, n)
    gx, gy = apply_gradient(p, n)
    ru = rhs_u - au - gx
    rv = rhs_v - av - gy

    ux, vy = _jacobi_velocity(ux, vy, ru, rv, n)

    res_div = apply_divergence(ux, vy, n) - rhs_div

    # interior cells
    r_int = res_div[1:n - 1, 1:n - 1]
    delta = r_int * h / 4.0
    ux = ux.at[1:n - 1, 1:n - 1].add(-delta)
    ux = ux.at[2:n, 1:n - 1].add(delta)
    vy = vy.at[1:n - 1, 1:n - 1].add(-delta)
    vy = vy.at[1:n - 1, 2:n].add(delta)
    p = p.at[1:n - 1, 1:n - 1].add(r_int)
    p = p.at[2:n, 1:n - 1].add(-0.25 * r_int)
    p = p.at[0:n - 2, 1:n - 1].add(-0.25 * r_int)
    p = p.at[1:n - 1, 2:n].add(-0.25 * r_int)
    p = p.at[1:n - 1, 0:n - 2].add(-0.25 * r_int)

    # top and bottom edges (j = n-1 and j = 0)
    r_top = res_div[1:n - 1, n - 1]
    delta_top = r_top * h / 3.0
    ux = ux.at[1:n - 1, n - 1].add(-delta_top)
    ux = ux.at[2:n, n - 1].add(delta_top)
    vy = vy.at[1:n - 1, n - 1].add(-delta_top)
    p = p.at[1:n - 1, n - 1].add(r_top)
    p = p.at[2:n, n - 1].add(-r_top / 3.0)
    p = p.at[0:n - 2, n - 1].add(-r_top / 3.0)
    p = p.at[1:n - 1, n - 2].add(-r_top / 3.0)

    r_bottom = res_div[1:n - 1, 0]
    delta_bottom = r_bottom * h / 3.0
    ux = ux.at[1:n - 1, 0].add(-delta_bottom)
    ux = ux.at[2:n, 0].add(delta_bottom)
    vy = vy.at[1:n - 1, 1].add(delta_bottom)
    p = p.at[1:n - 1, 0].add(r_bottom)
    p = p.at[2:n, 0].add(-r_bottom / 3.0)
    p = p.at[0:n - 2, 0].add(-r_bottom / 3.0)
    p = p.at[1:n - 1, 1].add(-r_bottom / 3.0)

    # left and right edges (i = 0 and i = n-1)
    r_right = res_div[n - 1, 1:n - 1]
    delta_right = r_right * h / 3.0
    vy = vy.at[n - 1, 1:n - 1].add(-delta_right)
    vy = vy.at[n - 1, 2:n].add(delta_right)
    ux = ux.at[n - 1, 1:n - 1].add(-delta_right)
    p = p.at[n - 1, 1:n - 1].add(r_right)
    p = p.at[n - 1, 2:n].add(-r_right / 3.0)
    p = p.at[n - 2, 1:n - 1].add(-r_right / 3.0)
    p = p.at[n - 1, 0:n - 2].add(-r_right / 3.0)

    r_left = res_div[0, 1:n - 1]
    delta_left = r_left * h / 3.0
    vy = vy.at[0, 1:n - 1].add(-delta_left)
    vy = vy.at[0, 2:n].add(delta_left)
    ux = ux.at[1, 1:n - 1].add(delta_left)
    p = p.at[0, 1:n - 1].add(r_left)
    p = p.at[0, 2:n].add(-r_left / 3.0)
    p = p.at[1, 1:n - 1].add(-r_left / 3.0)
    p = p.at[0, 0:n - 2].add(-r_left / 3.0)

    # corners
    r00 = res_div[0, 0]
    delta = r00 * h / 2.0
    vy = vy.at[0, 1].add(delta)
    ux = ux.at[1, 0].add(delta)
    p = p.at[0, 0].add(r00)
    p = p.at[0, 1].add(-0.5 * r00)
    p = p.at[1, 0].add(-0.5 * r00)

    rN0 = res_div[n - 1, 0]
    delta = rN0 * h / 2.0
    vy = vy.at[n - 1, 1].add(delta)
    ux = ux.at[n - 1, 0].add(-delta)
    p = p.at[n - 1, 0].add(rN0)
    p = p.at[n - 2, 0].add(-0.5 * rN0)
    p = p.at[n - 1, 1].add(-0.5 * rN0)

    r0N = res_div[0, n - 1]
    delta = r0N * h / 2.0
    vy = vy.at[0, n - 1].add(-delta)
    ux = ux.at[1, n - 1].add(delta)
    p = p.at[0, n - 1].add(r0N)
    p = p.at[1, n - 1].add(-0.5 * r0N)
    p = p.at[0, n - 2].add(-0.5 * r0N)

    rNN = res_div[n - 1, n - 1]
    delta = rNN * h / 2.0
    vy = vy.at[n - 1, n - 1].add(-delta)
    ux = ux.at[n - 1, n - 1].add(-delta)
    p = p.at[n - 1, n - 1].add(rNN)
    p = p.at[n - 2, n - 1].add(-0.5 * rNN)
    p = p.at[n - 1, n - 2].add(-0.5 * rNN)

    return ux, vy, p


@partial(jax.jit, static_argnums=(4,))
def smooth_velocity(ux: jnp.ndarray, vy: jnp.ndarray, rhs_u: jnp.ndarray, rhs_v: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Smoother for A-only problems (Jacobi-style)."""
    return _jacobi_velocity(ux, vy, rhs_u, rhs_v, n)
