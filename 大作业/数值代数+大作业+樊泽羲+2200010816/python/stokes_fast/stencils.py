"""Vectorized stencil operations on the MAC grid (JAX)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(2,))
def apply_viscous(ux: jnp.ndarray, vy: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    h = 1.0 / n
    au = jnp.zeros((n + 1, n), dtype=ux.dtype)
    av = jnp.zeros((n, n + 1), dtype=vy.dtype)

    au = au.at[1:n, 1:n - 1].set(
        -(1.0 / h**2)
        * (
            ux[1:n, 2:n]
            + ux[1:n, 0:n - 2]
            + ux[0:n - 1, 1:n - 1]
            + ux[2:n + 1, 1:n - 1]
            - 4.0 * ux[1:n, 1:n - 1]
        )
    )
    au = au.at[1:n, 0].set(
        -(1.0 / h**2) * (
            ux[0:n - 1, 0] + ux[2:n + 1, 0] - 2.0 * ux[1:n, 0] + ux[1:n, 1] - ux[1:n, 0]
        )
    )
    au = au.at[1:n, n - 1].set(
        -(1.0 / h**2)
        * (
            ux[0:n - 1, n - 1]
            + ux[2:n + 1, n - 1]
            - 2.0 * ux[1:n, n - 1]
            - ux[1:n, n - 1]
            + ux[1:n, n - 2]
        )
    )
    au = au.at[0, :].set(ux[0, :])
    au = au.at[n, :].set(ux[n, :])

    av = av.at[1:n - 1, 1:n].set(
        -(1.0 / h**2)
        * (
            vy[1:n - 1, 2:n + 1]
            + vy[1:n - 1, 0:n - 1]
            + vy[0:n - 2, 1:n]
            + vy[2:n, 1:n]
            - 4.0 * vy[1:n - 1, 1:n]
        )
    )
    av = av.at[0, 1:n].set(
        -(1.0 / h**2) * (
            vy[1, 1:n] - vy[0, 1:n] + vy[0, 2:n + 1] + vy[0, 0:n - 1] - 2.0 * vy[0, 1:n]
        )
    )
    av = av.at[n - 1, 1:n].set(
        -(1.0 / h**2)
        * (
            vy[n - 2, 1:n]
            - vy[n - 1, 1:n]
            + vy[n - 1, 2:n + 1]
            + vy[n - 1, 0:n - 1]
            - 2.0 * vy[n - 1, 1:n]
        )
    )
    av = av.at[:, 0].set(vy[:, 0])
    av = av.at[:, n].set(vy[:, n])

    return au, av


@partial(jax.jit, static_argnums=(1,))
def apply_gradient(p: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    h = 1.0 / n
    gx = jnp.zeros((n + 1, n), dtype=p.dtype)
    gy = jnp.zeros((n, n + 1), dtype=p.dtype)

    gx = gx.at[1:n, :].set((p[1:n, :] - p[0:n - 1, :]) / h)
    gy = gy.at[:, 1:n].set((p[:, 1:n] - p[:, 0:n - 1]) / h)
    return gx, gy


@partial(jax.jit, static_argnums=(2,))
def apply_divergence(ux: jnp.ndarray, vy: jnp.ndarray, n: int) -> jnp.ndarray:
    h = 1.0 / n
    return -(ux[1:n + 1, :] - ux[0:n, :] + vy[:, 1:n + 1] - vy[:, 0:n]) / h


@partial(jax.jit, static_argnums=(3,))
def apply_block(ux: jnp.ndarray, vy: jnp.ndarray, p: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    au, av = apply_viscous(ux, vy, n)
    gx, gy = apply_gradient(p, n)
    div = apply_divergence(ux, vy, n)
    return au + gx, av + gy, div
