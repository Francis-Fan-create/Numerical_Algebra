"""Right-hand side construction and error metrics."""

from __future__ import annotations

import jax.numpy as jnp


def rhs_terms(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    h = 1.0 / n
    fx = jnp.zeros((n + 1, n), dtype=jnp.float64)
    fy = jnp.zeros((n, n + 1), dtype=jnp.float64)

    i = jnp.arange(1, n)
    j = jnp.arange(1, n - 1)
    x = i[:, None] * h
    y = (j + 0.5) * h
    fx = fx.at[1:n, 1:n - 1].set(
        -4.0 * jnp.pi**2 * (2.0 * jnp.cos(2.0 * jnp.pi * x) - 1.0) * jnp.sin(2.0 * jnp.pi * y)
        + x**2
    )

    x = i * h
    y_bottom = 0.5 * h
    fx = fx.at[1:n, 0].set(
        -4.0 * jnp.pi**2 * (2.0 * jnp.cos(2.0 * jnp.pi * x) - 1.0) * jnp.sin(2.0 * jnp.pi * y_bottom)
        + x**2
        + (1.0 / h) * (2.0 * jnp.pi * (jnp.cos(2.0 * jnp.pi * x) - 1.0))
    )
    y_top = (n - 0.5) * h
    fx = fx.at[1:n, n - 1].set(
        -4.0 * jnp.pi**2 * (2.0 * jnp.cos(2.0 * jnp.pi * x) - 1.0) * jnp.sin(2.0 * jnp.pi * y_top)
        + x**2
        + (1.0 / h) * (-2.0 * jnp.pi * (jnp.cos(2.0 * jnp.pi * x) - 1.0))
    )

    i = jnp.arange(1, n - 1)
    j = jnp.arange(1, n)
    x = (i + 0.5)[:, None] * h
    y = j * h
    fy = fy.at[1:n - 1, 1:n].set(
        4.0 * jnp.pi**2 * (2.0 * jnp.cos(2.0 * jnp.pi * y) - 1.0) * jnp.sin(2.0 * jnp.pi * x)
    )

    j = jnp.arange(1, n)
    x_left = 0.5 * h
    y = j * h
    fy = fy.at[0, 1:n].set(
        4.0 * jnp.pi**2 * (2.0 * jnp.cos(2.0 * jnp.pi * y) - 1.0) * jnp.sin(2.0 * jnp.pi * x_left)
        + (1.0 / h) * (-2.0 * jnp.pi * (jnp.cos(2.0 * jnp.pi * y) - 1.0))
    )

    x_right = (n - 0.5) * h
    fy = fy.at[n - 1, 1:n].set(
        4.0 * jnp.pi**2 * (2.0 * jnp.cos(2.0 * jnp.pi * y) - 1.0) * jnp.sin(2.0 * jnp.pi * x_right)
        + (1.0 / h) * (2.0 * jnp.pi * (jnp.cos(2.0 * jnp.pi * y) - 1.0))
    )

    return fx, fy


def velocity_error(ux: jnp.ndarray, vy: jnp.ndarray, n: int) -> float:
    h = 1.0 / n
    i = jnp.arange(1, n)
    j = jnp.arange(1, n + 1)
    x = i[:, None] * h
    y = (j - 0.5) * h
    u_exact = (1.0 - jnp.cos(2.0 * jnp.pi * x)) * jnp.sin(2.0 * jnp.pi * y)
    u_num = ux[1:n, :]

    i = jnp.arange(1, n + 1)
    j = jnp.arange(1, n)
    x = (i - 0.5) * h
    y = j * h
    v_exact = -(1.0 - jnp.cos(2.0 * jnp.pi * y)) * jnp.sin(2.0 * jnp.pi * x[:, None])
    v_num = vy[:, 1:n]

    err = h * jnp.sqrt(jnp.sum((u_num - u_exact) ** 2) + jnp.sum((v_num - v_exact) ** 2))
    return float(err)
