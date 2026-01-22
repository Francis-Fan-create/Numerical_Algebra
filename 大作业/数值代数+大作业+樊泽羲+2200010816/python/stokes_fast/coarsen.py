"""Restriction and prolongation (distinct naming from original)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(2,))
def restrict_velocity(ux: jnp.ndarray, vy: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    half = n // 2
    coarse_u = jnp.zeros((half + 1, half), dtype=ux.dtype)
    coarse_v = jnp.zeros((half, half + 1), dtype=vy.dtype)

    coarse_u = coarse_u.at[0, :].set(0.5 * (ux[0, 0:n:2] + ux[0, 1:n:2]))
    coarse_u = coarse_u.at[half, :].set(0.5 * (ux[n, 0:n:2] + ux[n, 1:n:2]))

    i = jnp.arange(1, half)
    j = jnp.arange(0, half)
    ii = (2 * i)[:, None]
    jj = (2 * j)[None, :]
    coarse_u = coarse_u.at[i[:, None], j[None, :]].set(
        0.25 * (ux[ii, jj + 1] + ux[ii, jj])
        + 0.125 * (ux[ii - 1, jj + 1] + ux[ii + 1, jj + 1] + ux[ii - 1, jj] + ux[ii + 1, jj])
    )

    coarse_v = coarse_v.at[:, 0].set(0.5 * (vy[0:n:2, 0] + vy[1:n:2, 0]))
    coarse_v = coarse_v.at[:, half].set(0.5 * (vy[0:n:2, n] + vy[1:n:2, n]))

    i = jnp.arange(0, half)
    j = jnp.arange(1, half)
    ii = (2 * i)[:, None]
    jj = (2 * j)[None, :]
    coarse_v = coarse_v.at[i[:, None], j[None, :]].set(
        0.25 * (vy[ii + 1, jj] + vy[ii, jj])
        + 0.125 * (vy[ii + 1, jj - 1] + vy[ii, jj - 1] + vy[ii + 1, jj + 1] + vy[ii, jj + 1])
    )

    return coarse_u, coarse_v


@partial(jax.jit, static_argnums=(3,))
def restrict_all(ux: jnp.ndarray, vy: jnp.ndarray, p: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    half = n // 2
    coarse_u, coarse_v = restrict_velocity(ux, vy, n)
    coarse_p = 0.25 * (p[0:n:2, 0:n:2] + p[1:n:2, 0:n:2] + p[0:n:2, 1:n:2] + p[1:n:2, 1:n:2])
    return coarse_u, coarse_v, coarse_p


@partial(jax.jit, static_argnums=(2,))
def prolong_velocity(ux: jnp.ndarray, vy: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    fine_u = jnp.zeros((2 * n + 1, 2 * n), dtype=ux.dtype)
    fine_v = jnp.zeros((2 * n, 2 * n + 1), dtype=vy.dtype)

    fine_u = fine_u.at[0::2, 0::2].set(ux)
    fine_u = fine_u.at[0::2, 1::2].set(ux)
    fine_u = fine_u.at[1::2, 0::2].set(0.5 * (ux[:-1, :] + ux[1:, :]))
    fine_u = fine_u.at[1::2, 1::2].set(0.5 * (ux[:-1, :] + ux[1:, :]))

    fine_v = fine_v.at[1::2, 0::2].set(vy)
    fine_v = fine_v.at[0::2, 0::2].set(vy)
    fine_v = fine_v.at[1::2, 1::2].set(0.5 * (vy[:, :-1] + vy[:, 1:]))
    fine_v = fine_v.at[0::2, 1::2].set(0.5 * (vy[:, :-1] + vy[:, 1:]))

    return fine_u, fine_v


@partial(jax.jit, static_argnums=(3,))
def prolong_all(ux: jnp.ndarray, vy: jnp.ndarray, p: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    fine_u, fine_v = prolong_velocity(ux, vy, n)
    fine_p = jnp.repeat(jnp.repeat(p, 2, axis=0), 2, axis=1)
    return fine_u, fine_v, fine_p
