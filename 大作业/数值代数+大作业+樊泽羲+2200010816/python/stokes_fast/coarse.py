"""Coarsest-grid direct solvers (JAX)."""

from __future__ import annotations

import jax.numpy as jnp


def coarse_velocity_solve(rhs_u: jnp.ndarray, rhs_v: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    size = 2 * n * (n - 1)
    h = 1.0 / n
    A = jnp.zeros((size, size), dtype=jnp.float64)

    for j in range(1, n - 1):
        for i in range(1, n):
            row = j * (n - 1) + (i - 1)
            A = A.at[row, (j + 1) * (n - 1) + (i - 1)].set(-1.0 / h**2)
            A = A.at[row, (j - 1) * (n - 1) + (i - 1)].set(-1.0 / h**2)
            if i != n - 1:
                A = A.at[row, j * (n - 1) + i].set(-1.0 / h**2)
            if i != 1:
                A = A.at[row, j * (n - 1) + (i - 2)].set(-1.0 / h**2)
            A = A.at[row, row].set(4.0 / h**2)

    for i in range(1, n):
        j = 0
        row = j * (n - 1) + (i - 1)
        A = A.at[row, (j + 1) * (n - 1) + (i - 1)].set(-1.0 / h**2)
        if i != n - 1:
            A = A.at[row, j * (n - 1) + i].set(-1.0 / h**2)
        if i != 1:
            A = A.at[row, j * (n - 1) + (i - 2)].set(-1.0 / h**2)
        A = A.at[row, row].set(3.0 / h**2)

        j = n - 1
        row = j * (n - 1) + (i - 1)
        A = A.at[row, (j - 1) * (n - 1) + (i - 1)].set(-1.0 / h**2)
        if i != n - 1:
            A = A.at[row, j * (n - 1) + i].set(-1.0 / h**2)
        if i != 1:
            A = A.at[row, j * (n - 1) + (i - 2)].set(-1.0 / h**2)
        A = A.at[row, row].set(3.0 / h**2)

    offset = n * (n - 1)
    for i in range(1, n - 1):
        for j in range(1, n):
            row = offset + i * (n - 1) + (j - 1)
            A = A.at[row, offset + (i + 1) * (n - 1) + (j - 1)].set(-1.0 / h**2)
            A = A.at[row, offset + (i - 1) * (n - 1) + (j - 1)].set(-1.0 / h**2)
            if j != n - 1:
                A = A.at[row, offset + i * (n - 1) + j].set(-1.0 / h**2)
            if j != 1:
                A = A.at[row, offset + i * (n - 1) + (j - 2)].set(-1.0 / h**2)
            A = A.at[row, row].set(4.0 / h**2)

    for j in range(1, n):
        i = 0
        row = offset + i * (n - 1) + (j - 1)
        A = A.at[row, row].set(3.0 / h**2)
        A = A.at[row, offset + (i + 1) * (n - 1) + (j - 1)].set(-1.0 / h**2)
        if j != n - 1:
            A = A.at[row, offset + i * (n - 1) + j].set(-1.0 / h**2)
        if j != 1:
            A = A.at[row, offset + i * (n - 1) + (j - 2)].set(-1.0 / h**2)

        i = n - 1
        row = offset + i * (n - 1) + (j - 1)
        A = A.at[row, row].set(3.0 / h**2)
        A = A.at[row, offset + (i - 1) * (n - 1) + (j - 1)].set(-1.0 / h**2)
        if j != n - 1:
            A = A.at[row, offset + i * (n - 1) + j].set(-1.0 / h**2)
        if j != 1:
            A = A.at[row, offset + i * (n - 1) + (j - 2)].set(-1.0 / h**2)

    b = jnp.zeros((size,), dtype=jnp.float64)
    for i in range(1, n):
        for j in range(n):
            b = b.at[j * (n - 1) + (i - 1)].set(rhs_u[i, j])
    for j in range(1, n):
        for i in range(n):
            b = b.at[offset + i * (n - 1) + (j - 1)].set(rhs_v[i, j])

    x = jnp.linalg.solve(A, b)
    out_u = jnp.zeros_like(rhs_u)
    out_v = jnp.zeros_like(rhs_v)
    for i in range(1, n):
        for j in range(n):
            out_u = out_u.at[i, j].set(x[j * (n - 1) + (i - 1)])
    for j in range(1, n):
        for i in range(n):
            out_v = out_v.at[i, j].set(x[offset + i * (n - 1) + (j - 1)])

    return out_u, out_v


def coarse_stokes_solve(rhs_u: jnp.ndarray, rhs_v: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_vel = 2 * n * (n - 1)
    n_p = n * n
    h = 1.0 / n
    A = jnp.zeros((n_vel, n_vel), dtype=jnp.float64)
    B = jnp.zeros((n_vel, n_p), dtype=jnp.float64)

    for j in range(1, n - 1):
        for i in range(1, n):
            row = j * (n - 1) + (i - 1)
            A = A.at[row, (j + 1) * (n - 1) + (i - 1)].set(-1.0 / h**2)
            A = A.at[row, (j - 1) * (n - 1) + (i - 1)].set(-1.0 / h**2)
            if i != n - 1:
                A = A.at[row, j * (n - 1) + i].set(-1.0 / h**2)
            if i != 1:
                A = A.at[row, j * (n - 1) + (i - 2)].set(-1.0 / h**2)
            A = A.at[row, row].set(4.0 / h**2)
            B = B.at[row, j * n + i].set(1.0 / h)
            B = B.at[row, j * n + i - 1].set(-1.0 / h)

    for i in range(1, n):
        j = 0
        row = j * (n - 1) + (i - 1)
        A = A.at[row, (j + 1) * (n - 1) + (i - 1)].set(-1.0 / h**2)
        if i != n - 1:
            A = A.at[row, j * (n - 1) + i].set(-1.0 / h**2)
        if i != 1:
            A = A.at[row, j * (n - 1) + (i - 2)].set(-1.0 / h**2)
        A = A.at[row, row].set(3.0 / h**2)
        B = B.at[row, j * n + i].set(1.0 / h)
        B = B.at[row, j * n + i - 1].set(-1.0 / h)

        j = n - 1
        row = j * (n - 1) + (i - 1)
        A = A.at[row, (j - 1) * (n - 1) + (i - 1)].set(-1.0 / h**2)
        if i != n - 1:
            A = A.at[row, j * (n - 1) + i].set(-1.0 / h**2)
        if i != 1:
            A = A.at[row, j * (n - 1) + (i - 2)].set(-1.0 / h**2)
        A = A.at[row, row].set(3.0 / h**2)
        B = B.at[row, j * n + i].set(1.0 / h)
        B = B.at[row, j * n + i - 1].set(-1.0 / h)

    offset = n * (n - 1)
    for i in range(1, n - 1):
        for j in range(1, n):
            row = offset + i * (n - 1) + (j - 1)
            A = A.at[row, offset + (i + 1) * (n - 1) + (j - 1)].set(-1.0 / h**2)
            A = A.at[row, offset + (i - 1) * (n - 1) + (j - 1)].set(-1.0 / h**2)
            if j != n - 1:
                A = A.at[row, offset + i * (n - 1) + j].set(-1.0 / h**2)
            if j != 1:
                A = A.at[row, offset + i * (n - 1) + (j - 2)].set(-1.0 / h**2)
            A = A.at[row, row].set(4.0 / h**2)
            B = B.at[row, j * n + i].set(1.0 / h)
            B = B.at[row, (j - 1) * n + i].set(-1.0 / h)

    for j in range(1, n):
        i = 0
        row = offset + i * (n - 1) + (j - 1)
        A = A.at[row, row].set(3.0 / h**2)
        A = A.at[row, offset + (i + 1) * (n - 1) + (j - 1)].set(-1.0 / h**2)
        if j != n - 1:
            A = A.at[row, offset + i * (n - 1) + j].set(-1.0 / h**2)
        if j != 1:
            A = A.at[row, offset + i * (n - 1) + (j - 2)].set(-1.0 / h**2)
        B = B.at[row, j * n + i].set(1.0 / h)
        B = B.at[row, (j - 1) * n + i].set(-1.0 / h)

        i = n - 1
        row = offset + i * (n - 1) + (j - 1)
        A = A.at[row, row].set(3.0 / h**2)
        A = A.at[row, offset + (i - 1) * (n - 1) + (j - 1)].set(-1.0 / h**2)
        if j != n - 1:
            A = A.at[row, offset + i * (n - 1) + j].set(-1.0 / h**2)
        if j != 1:
            A = A.at[row, offset + i * (n - 1) + (j - 2)].set(-1.0 / h**2)
        B = B.at[row, j * n + i].set(1.0 / h)
        B = B.at[row, (j - 1) * n + i].set(-1.0 / h)

    H = jnp.zeros((n_vel + n_p + 1, n_vel + n_p), dtype=jnp.float64)
    H = H.at[:n_vel, :n_vel].set(A)
    H = H.at[:n_vel, n_vel:].set(B)
    H = H.at[n_vel:n_vel + n_p, :n_vel].set(B.T)
    H = H.at[n_vel + n_p, -1].set(1.0)

    b = jnp.zeros((n_vel + n_p + 1,), dtype=jnp.float64)
    b = b.at[n_vel + n_p].set(1.0)
    for i in range(1, n):
        for j in range(n):
            b = b.at[j * (n - 1) + (i - 1)].set(rhs_u[i, j])
    for j in range(1, n):
        for i in range(n):
            b = b.at[offset + i * (n - 1) + (j - 1)].set(rhs_v[i, j])

    x = jnp.linalg.solve(H.T @ H, H.T @ b)
    out_u = jnp.zeros_like(rhs_u)
    out_v = jnp.zeros_like(rhs_v)
    out_p = jnp.zeros((n, n), dtype=jnp.float64)
    for i in range(1, n):
        for j in range(n):
            out_u = out_u.at[i, j].set(x[j * (n - 1) + (i - 1)])
    for j in range(1, n):
        for i in range(n):
            out_v = out_v.at[i, j].set(x[offset + i * (n - 1) + (j - 1)])
    for i in range(n):
        for j in range(n):
            out_p = out_p.at[i, j].set(x[n_vel + j * n + i])

    return out_u, out_v, out_p
