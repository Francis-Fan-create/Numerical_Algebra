import numpy as np
import time
import os
import csv
import logging

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

USE_NUMBA = NUMBA_AVAILABLE and os.environ.get("USE_NUMBA", "1") != "0"

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _dgs_smoother_numba(u, v, p, f, g, h, d):
        h2 = h * h
        N = p.shape[0]

        # Enforce Dirichlet boundaries
        for j in range(N):
            u[0, j] = 0.0
            u[N, j] = 0.0
        for i in range(N):
            v[i, 0] = 0.0
            v[i, N] = 0.0

        omega = 1.0

        # Momentum relaxation for u (red-black Gauss-Seidel)
        for color in (0, 1):
            if N > 2:
                for i in range(1, N):
                    for j in range(1, N - 1):
                        if (i + j) % 2 == color:
                            px = (p[i, j] - p[i - 1, j]) / h
                            rhs = f[i, j] - px
                            u[i, j] = (1.0 - omega) * u[i, j] + omega * 0.25 * (
                                u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] + h2 * rhs
                            )

            for i in range(1, N):
                if (i % 2) == color:
                    px = (p[i, 0] - p[i - 1, 0]) / h
                    rhs = f[i, 0] - px
                    u[i, 0] = (1.0 - omega) * u[i, 0] + omega * (1.0 / 3.0) * (
                        u[i - 1, 0] + u[i + 1, 0] + u[i, 1] + h2 * rhs
                    )

                if ((i + (N - 1)) % 2) == color:
                    px = (p[i, N - 1] - p[i - 1, N - 1]) / h
                    rhs = f[i, N - 1] - px
                    u[i, N - 1] = (1.0 - omega) * u[i, N - 1] + omega * (1.0 / 3.0) * (
                        u[i - 1, N - 1] + u[i + 1, N - 1] + u[i, N - 2] + h2 * rhs
                    )

        # Momentum relaxation for v (red-black Gauss-Seidel)
        for color in (0, 1):
            if N > 2:
                for i in range(1, N - 1):
                    for j in range(1, N):
                        if (i + j) % 2 == color:
                            py = (p[i, j] - p[i, j - 1]) / h
                            rhs = g[i, j] - py
                            v[i, j] = (1.0 - omega) * v[i, j] + omega * 0.25 * (
                                v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1] + h2 * rhs
                            )

            for j in range(1, N):
                if (j % 2) == color:
                    py = (p[0, j] - p[0, j - 1]) / h
                    rhs = g[0, j] - py
                    v[0, j] = (1.0 - omega) * v[0, j] + omega * (1.0 / 3.0) * (
                        v[1, j] + v[0, j - 1] + v[0, j + 1] + h2 * rhs
                    )

                if (((N - 1) + j) % 2) == color:
                    py = (p[N - 1, j] - p[N - 1, j - 1]) / h
                    rhs = g[N - 1, j] - py
                    v[N - 1, j] = (1.0 - omega) * v[N - 1, j] + omega * (1.0 / 3.0) * (
                        v[N - 2, j] + v[N - 1, j - 1] + v[N - 1, j + 1] + h2 * rhs
                    )

        # Divergence correction with boundary-aware scaling (red-black DGS)
        for color in (0, 1):
            for i in range(N):
                for j in range(N):
                    if (i + j) % 2 == color:
                        div = (u[i + 1, j] - u[i, j]) / h + (v[i, j + 1] - v[i, j]) / h
                        r = -div - d[i, j]
                        n = 4.0
                        if i == 0:
                            n -= 1.0
                        if i == N - 1:
                            n -= 1.0
                        if j == 0:
                            n -= 1.0
                        if j == N - 1:
                            n -= 1.0
                        delta = r * h / n
                        if i > 0:
                            u[i, j] -= delta
                        if i < N - 1:
                            u[i + 1, j] += delta
                        if j > 0:
                            v[i, j] -= delta
                        if j < N - 1:
                            v[i, j + 1] += delta
                        w = r / n
                        p[i, j] += r
                        if i > 0:
                            p[i - 1, j] -= w
                        if i < N - 1:
                            p[i + 1, j] -= w
                        if j > 0:
                            p[i, j - 1] -= w
                        if j < N - 1:
                            p[i, j + 1] -= w

        # Ensure zero-mean pressure
        mean_p = 0.0
        for i in range(N):
            for j in range(N):
                mean_p += p[i, j]
        mean_p /= (N * N)
        for i in range(N):
            for j in range(N):
                p[i, j] -= mean_p

        # Re-enforce Dirichlet boundaries after correction
        for j in range(N):
            u[0, j] = 0.0
            u[N, j] = 0.0
        for i in range(N):
            v[i, 0] = 0.0
            v[i, N] = 0.0

    @njit(cache=True)
    def _restrict_uvp_numba(r_u, r_v, r_p):
        N = r_p.shape[0]
        Nc = N // 2
        rc_u = np.zeros((Nc + 1, Nc))
        rc_v = np.zeros((Nc, Nc + 1))
        rc_p = np.zeros((Nc, Nc))

        for i in range(Nc + 1):
            for j in range(Nc):
                i_f = 2 * i
                j_f = 2 * j
                if i_f < N + 1 and j_f + 1 < N:
                    rc_u[i, j] = 0.5 * (r_u[i_f, j_f] + r_u[i_f, j_f + 1])

        for i in range(Nc):
            for j in range(Nc + 1):
                i_f = 2 * i
                j_f = 2 * j
                if i_f + 1 < N and j_f < N + 1:
                    rc_v[i, j] = 0.5 * (r_v[i_f, j_f] + r_v[i_f + 1, j_f])

        for i in range(Nc):
            for j in range(Nc):
                i_f = 2 * i
                j_f = 2 * j
                rc_p[i, j] = 0.25 * (
                    r_p[i_f, j_f] + r_p[i_f + 1, j_f] + r_p[i_f, j_f + 1] + r_p[i_f + 1, j_f + 1]
                )

        return rc_u, rc_v, rc_p

    @njit(cache=True)
    def _prolongate_rect_numba(ec, nr, nc):
        ef = np.zeros((nr, nc))
        nr_c, nc_c = ec.shape
        for I in range(nr):
            i = I // 2
            if i >= nr_c:
                i = nr_c - 1
            i1 = i + 1
            if i1 >= nr_c:
                i1 = nr_c - 1
            wi1 = 0.0 if (I % 2) == 0 else 0.5
            wi0 = 1.0 - wi1
            for J in range(nc):
                j = J // 2
                if j >= nc_c:
                    j = nc_c - 1
                j1 = j + 1
                if j1 >= nc_c:
                    j1 = nc_c - 1
                wj1 = 0.0 if (J % 2) == 0 else 0.5
                wj0 = 1.0 - wj1
                ef[I, J] = (
                    wi0 * wj0 * ec[i, j]
                    + wi1 * wj0 * ec[i1, j]
                    + wi0 * wj1 * ec[i, j1]
                    + wi1 * wj1 * ec[i1, j1]
                )
        return ef

class StokesMultigrid:
    def __init__(self, N, nu1=3, nu2=3, L=None):
        """
        Initialize Staggered Grid for Stokes Equations.
        N: Grid resolution (must be power of 2)
        h: Mesh size 1/N
        """
        self.N = N
        self.h = 1.0 / N
        # Pre-smoothing and post-smoothing iteration counts
        self.nu1 = int(nu1)
        self.nu2 = int(nu2)
        # Number of multigrid levels (coarsest resolution will be N_coarse=2**(L-1))
        if L is None:
            # default: maximum allowed by grid resolution
            self.L = int(np.log2(N))
        else:
            self.L = int(L)
        
        # --- Grid Storage ---
        # u: Vertical edges, size (N+1) x N. 
        # v: Horizontal edges, size N x (N+1).
        # p: Cell centers, size N x N.
        self.u = np.zeros((N + 1, N))
        self.v = np.zeros((N, N + 1))
        self.p = np.zeros((N, N))
        
        # Right Hand Side (Force Terms)
        self.f = np.zeros((N + 1, N))
        self.g = np.zeros((N, N + 1))
        
        self.init_problem()

    def init_problem(self):
        """Initialize Force terms and Boundary Conditions"""
        # --- Initialize f on u-grid ---
        I_u, J_u = np.meshgrid(np.arange(self.N + 1), np.arange(self.N), indexing='ij')
        X_u = I_u * self.h
        Y_u = (J_u + 0.5) * self.h
        
        self.f = -4 * np.pi**2 * (2 * np.cos(2*np.pi*X_u) - 1) * np.sin(2*np.pi*Y_u) + X_u**2
        # Neumann boundary contributions for u in y-direction
        x_u = np.arange(1, self.N) * self.h
        u_y = 2 * np.pi * (1 - np.cos(2 * np.pi * x_u))
        self.f[1:-1, 0] -= u_y / self.h
        self.f[1:-1, -1] += u_y / self.h
        # Dirichlet boundaries in x
        self.f[0, :] = 0
        self.f[-1, :] = 0

        # --- Initialize g on v-grid ---
        I_v, J_v = np.meshgrid(np.arange(self.N), np.arange(self.N + 1), indexing='ij')
        X_v = (I_v + 0.5) * self.h
        Y_v = J_v * self.h
        
        self.g = 4 * np.pi**2 * (2 * np.cos(2*np.pi*Y_v) - 1) * np.sin(2*np.pi*X_v)
        # Neumann boundary contributions for v in x-direction
        y_v = np.arange(1, self.N) * self.h
        v_x = -2 * np.pi * (1 - np.cos(2 * np.pi * y_v))
        self.g[0, 1:-1] -= v_x / self.h
        self.g[-1, 1:-1] += v_x / self.h
        # Dirichlet boundaries in y
        self.g[:, 0] = 0
        self.g[:, -1] = 0


    def get_exact_solution(self):
        """Calculate exact solution on discrete grid"""
        I_u, J_u = np.meshgrid(np.arange(self.N + 1), np.arange(self.N), indexing='ij')
        X_u = I_u * self.h
        Y_u = (J_u + 0.5) * self.h
        u_exact = (1 - np.cos(2*np.pi*X_u)) * np.sin(2*np.pi*Y_u)
        
        I_v, J_v = np.meshgrid(np.arange(self.N), np.arange(self.N + 1), indexing='ij')
        X_v = (I_v + 0.5) * self.h
        Y_v = J_v * self.h
        v_exact = -(1 - np.cos(2*np.pi*Y_v)) * np.sin(2*np.pi*X_v)
        
        return u_exact, v_exact

    def dgs_smoother(self, u, v, p, f, g, h, d=None):
        """DGS Smoother: Momentum relaxation + Divergence correction"""
        h2 = h * h
        N = p.shape[0]
        if d is None:
            d = np.zeros_like(p)

        if USE_NUMBA:
            _dgs_smoother_numba(u, v, p, f, g, h, d)
            return

        # Enforce Dirichlet boundaries
        u[0, :] = 0.0
        u[-1, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        
        # GS relaxation parameter for momentum equations
        omega = 1.0
        
        # Momentum relaxation for u (red-black Gauss-Seidel)
        px = np.zeros((N+1, N))
        px[1:-1, :] = (p[1:, :] - p[:-1, :]) / h

        for color in [0, 1]:
            # interior rows (j=1..N-2)
            if N > 2:
                I, J = np.meshgrid(np.arange(1, N), np.arange(1, N-1), indexing='ij')
                mask = (I + J) % 2 == color
                u_inner = u[1:-1, 1:-1]
                u_left = u[0:-2, 1:-1]
                u_right = u[2:, 1:-1]
                u_down = u[1:-1, 0:-2]
                u_up = u[1:-1, 2:]
                rhs_u = f[1:-1, 1:-1] - px[1:-1, 1:-1]
                u_inner[mask] = (1-omega) * u_inner[mask] + omega * 0.25 * (
                    u_left[mask] + u_right[mask] + u_down[mask] + u_up[mask] + h2 * rhs_u[mask]
                )

            # bottom boundary row (j=0)
            i_idx = np.arange(1, N)
            mask_row = ((i_idx + 0) % 2) == color
            u_row = u[1:-1, 0]
            rhs_row = f[1:-1, 0] - px[1:-1, 0]
            u_row[mask_row] = (1-omega) * u_row[mask_row] + omega * (1.0 / 3.0) * (
                u[0:-2, 0][mask_row] + u[2:, 0][mask_row] + u[1:-1, 1][mask_row] +
                h2 * rhs_row[mask_row]
            )

            # top boundary row (j=N-1)
            mask_row = ((i_idx + (N-1)) % 2) == color
            u_row = u[1:-1, -1]
            rhs_row = f[1:-1, -1] - px[1:-1, -1]
            u_row[mask_row] = (1-omega) * u_row[mask_row] + omega * (1.0 / 3.0) * (
                u[0:-2, -1][mask_row] + u[2:, -1][mask_row] + u[1:-1, -2][mask_row] +
                h2 * rhs_row[mask_row]
            )
        
        # Momentum relaxation for v (red-black Gauss-Seidel)
        py = np.zeros((N, N+1))
        py[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / h

        for color in [0, 1]:
            # interior columns (i=1..N-2)
            if N > 2:
                I, J = np.meshgrid(np.arange(1, N-1), np.arange(1, N), indexing='ij')
                mask = (I + J) % 2 == color
                v_inner = v[1:-1, 1:-1]
                v_left = v[0:-2, 1:-1]
                v_right = v[2:, 1:-1]
                v_down = v[1:-1, 0:-2]
                v_up = v[1:-1, 2:]
                rhs_v = g[1:-1, 1:-1] - py[1:-1, 1:-1]
                v_inner[mask] = (1-omega) * v_inner[mask] + omega * 0.25 * (
                    v_left[mask] + v_right[mask] + v_down[mask] + v_up[mask] + h2 * rhs_v[mask]
                )

            # left boundary column (i=0)
            j_idx = np.arange(1, N)
            mask_col = ((0 + j_idx) % 2) == color
            v_col = v[0, 1:-1]
            rhs_col = g[0, 1:-1] - py[0, 1:-1]
            v_col[mask_col] = (1-omega) * v_col[mask_col] + omega * (1.0 / 3.0) * (
                v[1, 1:-1][mask_col] + v[0, 0:-2][mask_col] + v[0, 2:][mask_col] +
                h2 * rhs_col[mask_col]
            )

            # right boundary column (i=N-1)
            mask_col = (((N-1) + j_idx) % 2) == color
            v_col = v[-1, 1:-1]
            rhs_col = g[-1, 1:-1] - py[-1, 1:-1]
            v_col[mask_col] = (1-omega) * v_col[mask_col] + omega * (1.0 / 3.0) * (
                v[-2, 1:-1][mask_col] + v[-1, 0:-2][mask_col] + v[-1, 2:][mask_col] +
                h2 * rhs_col[mask_col]
            )

        # Divergence correction with boundary-aware scaling (red-black DGS)
        I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        n = 4.0 * np.ones((N, N))
        n[0, :] -= 1
        n[-1, :] -= 1
        n[:, 0] -= 1
        n[:, -1] -= 1

        for color in [0, 1]:
            div = (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h
            r = -div - d
            delta = r * h / n
            mask = ((I + J) % 2) == color

            delta_c = np.zeros_like(delta)
            delta_c[mask] = delta[mask]

            # Apply corrections, excluding Dirichlet boundary edges
            delta_left = delta_c.copy()
            delta_left[0, :] = 0.0
            delta_right = delta_c.copy()
            delta_right[-1, :] = 0.0
            u[:-1, :] -= delta_left
            u[1:, :] += delta_right

            delta_bottom = delta_c.copy()
            delta_bottom[:, 0] = 0.0
            delta_top = delta_c.copy()
            delta_top[:, -1] = 0.0
            v[:, :-1] -= delta_bottom
            v[:, 1:] += delta_top

            # Distributive pressure update with boundary-aware weights (masked)
            r_c = np.zeros_like(r)
            r_c[mask] = r[mask]
            w = np.zeros_like(r)
            w[mask] = r[mask] / n[mask]
            p += r_c
            p[1:, :] -= w[:-1, :]
            p[:-1, :] -= w[1:, :]
            p[:, 1:] -= w[:, :-1]
            p[:, :-1] -= w[:, 1:]
        # Ensure zero-mean pressure to keep pressure uniquely defined
        p -= np.mean(p)

        # Re-enforce Dirichlet boundaries after correction
        u[0, :] = 0.0
        u[-1, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0

    def compute_residual_norm(self, u, v, p, f, g, h):
        """Calculate L2 Norm of Residuals"""
        # Momentum u residual
        px = (p[1:, :] - p[:-1, :]) / h
        u_pad = np.pad(u, ((0,0),(1,1)), mode='edge')
        lap_u = (u[0:-2,:] + u[2:,:] + u_pad[1:-1,0:-2] + u_pad[1:-1,2:] - 4*u[1:-1,:]) / h**2
        res_u = f[1:-1, :] - (-lap_u + px)
        
        # Momentum v residual
        py = (p[:, 1:] - p[:, :-1]) / h
        v_pad = np.pad(v, ((1,1),(0,0)), mode='edge')
        lap_v = (v_pad[0:-2,1:-1] + v_pad[2:,1:-1] + v[:,0:-2] + v[:,2:] - 4*v[:,1:-1]) / h**2
        res_v = g[:, 1:-1] - (-lap_v + py)
        
        # Continuity residual
        div = (u[1:, :] - u[:-1, :])/h + (v[:, 1:] - v[:, :-1])/h
        
        return np.sqrt(np.sum(res_u**2) + np.sum(res_v**2) + np.sum(div**2))

    def restrict(self, r_u, r_v, r_p):
        """Restriction Operator (Fine -> Coarse)"""
        N = r_p.shape[0]
        Nc = N // 2

        if USE_NUMBA:
            return _restrict_uvp_numba(r_u, r_v, r_p)
        
        # Restrict u
        rc_u = np.zeros((Nc + 1, Nc))
        for i in range(Nc + 1):
            for j in range(Nc):
                i_f, j_f = 2*i, 2*j
                if i_f < N+1 and j_f+1 < N:
                    rc_u[i, j] = 0.5 * (r_u[i_f, j_f] + r_u[i_f, j_f+1])

        # Restrict v
        rc_v = np.zeros((Nc, Nc + 1))
        for i in range(Nc):
            for j in range(Nc + 1):
                i_f, j_f = 2*i, 2*j
                if i_f+1 < N and j_f < N+1:
                    rc_v[i, j] = 0.5 * (r_v[i_f, j_f] + r_v[i_f+1, j_f])

        # Restrict p
        rc_p = np.zeros((Nc, Nc))
        for i in range(Nc):
            for j in range(Nc):
                i_f, j_f = 2*i, 2*j
                rc_p[i, j] = 0.25 * (r_p[i_f, j_f] + r_p[i_f+1, j_f] +
                                      r_p[i_f, j_f+1] + r_p[i_f+1, j_f+1])
        
        return rc_u, rc_v, rc_p

    def prolongate(self, du_c, dv_c, dp_c):
        """Prolongation Operator (Coarse -> Fine) using bilinear interpolation."""
        Nc = dp_c.shape[0]
        Nf = Nc * 2

        if USE_NUMBA:
            dp = _prolongate_rect_numba(dp_c, Nf, Nf)
            du = _prolongate_rect_numba(du_c, Nf + 1, Nf)
            dv = _prolongate_rect_numba(dv_c, Nf, Nf + 1)
            return du, dv, dp

        def prolongate_rect(ec, target_shape):
            ef = np.zeros(target_shape)
            ef[0::2, 0::2] = ec

            # odd rows, even cols
            ec_pad_v = np.pad(ec, ((0, 1), (0, 0)), mode='edge')
            vv = 0.5 * (ec_pad_v[:-1, :] + ec_pad_v[1:, :])
            tr_v = ef[1::2, 0::2]
            ef[1::2, 0::2][:tr_v.shape[0], :tr_v.shape[1]] = vv[:tr_v.shape[0], :tr_v.shape[1]]

            # even rows, odd cols
            ec_pad_h = np.pad(ec, ((0, 0), (0, 1)), mode='edge')
            hh = 0.5 * (ec_pad_h[:, :-1] + ec_pad_h[:, 1:])
            tr_h = ef[0::2, 1::2]
            ef[0::2, 1::2][:tr_h.shape[0], :tr_h.shape[1]] = hh[:tr_h.shape[0], :tr_h.shape[1]]

            # odd rows, odd cols
            ec_pad = np.pad(ec, ((0, 1), (0, 1)), mode='edge')
            cc = 0.25 * (ec_pad[:-1, :-1] + ec_pad[1:, :-1] + ec_pad[:-1, 1:] + ec_pad[1:, 1:])
            tr_c = ef[1::2, 1::2]
            ef[1::2, 1::2][:tr_c.shape[0], :tr_c.shape[1]] = cc[:tr_c.shape[0], :tr_c.shape[1]]
            return ef

        dp = prolongate_rect(dp_c, (Nf, Nf))
        du = prolongate_rect(du_c, (Nf + 1, Nf))
        dv = prolongate_rect(dv_c, (Nf, Nf + 1))

        return du, dv, dp

    def v_cycle(self, u, v, p, f, g, h, level=0, d=None):
        """V-Cycle"""
        N = p.shape[0]
        if d is None:
            d = np.zeros_like(p)
        
        # Damping for coarse grid correction
        beta = 1.0
        
        # Pre-smoothing
        for _ in range(self.nu1):
            self.dgs_smoother(u, v, p, f, g, h, d=d)
            
        # Base case
        # Base case: use coarse grid when the number of cells is small or when levels are exhausted
        if N <= 4 or level >= (self.L - 1):
            for _ in range(20):
                self.dgs_smoother(u, v, p, f, g, h, d=d)
            return

        # Compute residuals
        px = (p[1:, :] - p[:-1, :]) / h
        u_pad = np.pad(u, ((0,0),(1,1)), mode='edge')
        lap_u = (u[0:-2,:] + u[2:,:] + u_pad[1:-1,0:-2] + u_pad[1:-1,2:] - 4*u[1:-1,:]) / h**2
        r_u = np.zeros_like(u); r_u[1:-1, :] = f[1:-1, :] - (-lap_u + px)
        
        py = (p[:, 1:] - p[:, :-1]) / h
        v_pad = np.pad(v, ((1,1),(0,0)), mode='edge')
        lap_v = (v_pad[0:-2,1:-1] + v_pad[2:,1:-1] + v[:,0:-2] + v[:,2:] - 4*v[:,1:-1]) / h**2
        r_v = np.zeros_like(v); r_v[:, 1:-1] = g[:, 1:-1] - (-lap_v + py)
        
        div = (u[1:, :] - u[:-1, :])/h + (v[:, 1:] - v[:, :-1])/h
        r_p = -div - d

        # Restrict
        rc_u, rc_v, rc_p = self.restrict(r_u, r_v, r_p)
        
        # Solve coarse - damped correction
        ec_u = np.zeros_like(rc_u)
        ec_v = np.zeros_like(rc_v)
        ec_p = np.zeros_like(rc_p)
        self.v_cycle(ec_u, ec_v, ec_p, rc_u, rc_v, 2*h, level+1, d=-rc_p)
        
        # Prolongate and correct with adaptive damping
        du, dv, dp = self.prolongate(ec_u, ec_v, ec_p)
        u += beta * du
        v += beta * dv
        p += beta * dp
        
        # Post-smoothing
        for _ in range(self.nu2):
            self.dgs_smoother(u, v, p, f, g, h, d=d)

    def solve(self, logger=None):
        if logger is not None:
            logger.info(f"--- Solving Stokes (DGS-MG) for N={self.N} ---")
        else:
            print(f"--- Solving Stokes (DGS-MG) for N={self.N} ---")
        t0 = time.time()
        
        r0 = self.compute_residual_norm(self.u, self.v, self.p, self.f, self.g, self.h)
        if r0 < 1e-15: r0 = 1.0
        if logger is not None:
            logger.info(f"Init Residual: {r0:.4e}")
        else:
            print(f"Init Residual: {r0:.4e}")
        
        tol = 1e-8
        rel_res = 1.0
        res_history = [1.0]
        iter_k = 0
        
        while rel_res > tol and iter_k < 50:
            self.v_cycle(self.u, self.v, self.p, self.f, self.g, self.h)
            
            rk = self.compute_residual_norm(self.u, self.v, self.p, self.f, self.g, self.h)
            rel_res = rk / r0
            res_history.append(rel_res)
            iter_k += 1
            msg = f"Iter {iter_k}: Rel Res = {rel_res:.4e}"
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)
            
        cpu_time = time.time() - t0
        
        u_ex, v_ex = self.get_exact_solution()
        err_u_sq = np.sum((self.u[1:-1, :] - u_ex[1:-1, :])**2)
        err_v_sq = np.sum((self.v[:, 1:-1] - v_ex[:, 1:-1])**2)
        e_N = self.h * np.sqrt(err_u_sq + err_v_sq)
        
        if logger is not None:
            logger.info(f"Finished: Iters={iter_k}, CPU(s)={cpu_time:.6f}, Error={e_N:.6e}")
        return iter_k, cpu_time, e_N, res_history

# --- Main Execution Block ---
if __name__ == "__main__":
    max_n = int(os.environ.get("MAX_N", "2048"))
    N_list = [n for n in [64, 128, 256, 512, 1024, 2048] if n <= max_n]

    def levels_for_coarse(N, N_coarse):
        return int(np.log2(N / N_coarse)) + 1

    configs = [
        {"name": "nu6_L2", "nu1": 6, "nu2": 6, "N_coarse": 2},
        {"name": "nu6_L4", "nu1": 6, "nu2": 6, "N_coarse": 4},
        {"name": "nu4_L2", "nu1": 4, "nu2": 4, "N_coarse": 2},
        {"name": "nu4_L4", "nu1": 4, "nu2": 4, "N_coarse": 4},
        {"name": "nu2_L2", "nu1": 2, "nu2": 2, "N_coarse": 2},
        {"name": "nu2_L4", "nu1": 2, "nu2": 2, "N_coarse": 4},
    ]

    only_cfg = os.environ.get("ONLY_CONFIG")
    if only_cfg:
        configs = [c for c in configs if c["name"] == only_cfg]

    results_base = os.path.join(os.path.dirname(__file__), "results")
    script_name = "v_cycle"
    results_dir = os.path.join(results_base, script_name)
    os.makedirs(results_dir, exist_ok=True)
    force_rerun = os.environ.get("FORCE_RERUN") == "1"

    for cfg in configs:
        cfg_dir = os.path.join(results_dir, cfg["name"])
        os.makedirs(cfg_dir, exist_ok=True)
        summary_file = os.path.join(cfg_dir, f"results_{cfg['name']}.csv")

        print(f"\n=== V-cycle DGS-MG: {cfg['name']} ===")
        print(f"{'N':<6} | {'V-Cycles':<8} | {'CPU Time(s)':<12} | {'Error e_N':<12}")
        print("-" * 46)

        existing_ns = set()
        if not force_rerun and os.path.exists(summary_file):
            with open(summary_file, "r", newline='') as sf:
                reader = csv.reader(sf)
                next(reader, None)
                for row in reader:
                    if row and row[0].isdigit():
                        existing_ns.add(int(row[0]))

        write_header = True if force_rerun else (not os.path.exists(summary_file))
        open_mode = "w" if force_rerun else "a"
        with open(summary_file, open_mode, newline='') as sf:
            writer = csv.writer(sf)
            if write_header:
                writer.writerow(["N", "V-Cycles", "CPU Time(s)", "Error e_N", "nu1", "nu2", "L", "InitialRelRes", "FinalRelRes", "ResidualFile"])

            for N in N_list:
                if N in existing_ns:
                    continue
                L = levels_for_coarse(N, cfg["N_coarse"])
                solver = StokesMultigrid(N, nu1=cfg["nu1"], nu2=cfg["nu2"], L=L)

                run_dir = os.path.join(cfg_dir, f"N{N}")
                os.makedirs(run_dir, exist_ok=True)
                log_file = os.path.join(run_dir, f"v_cycle_{cfg['name']}_N{N}.log")
                logger = logging.getLogger(f"v_cycle_{cfg['name']}_{N}")
                for h in list(logger.handlers):
                    logger.removeHandler(h)
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler(log_file)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                fh.setFormatter(formatter)
                logger.addHandler(fh)

                logger.info(f"Run V-cycle: N={N}, h={solver.h}, cfg={cfg['name']}")
                it, t, err, res_hist = solver.solve(logger=logger)

                residuals_file = os.path.join(run_dir, f"residuals_v_cycle_{cfg['name']}_N{N}.csv")
                with open(residuals_file, "w", newline='') as rf:
                    w2 = csv.writer(rf)
                    w2.writerow(["iter", "rel_res"])
                    for idx, val in enumerate(res_hist):
                        w2.writerow([idx, val])

                summary_row = (
                    N,
                    it,
                    t,
                    err,
                    solver.nu1,
                    solver.nu2,
                    solver.L,
                    res_hist[0],
                    res_hist[-1],
                    os.path.basename(residuals_file),
                )
                writer.writerow(summary_row)
                sf.flush()
                print(f"{N:<6} | {it:<8} | {t:<12.4f} | {err:<12.4e}")
                logger.info(
                    f"Summary: N={N}, V-Cycles={it}, CPU(s)={t:.6f}, Error={err:.6e}, ResidualFile={os.path.basename(residuals_file)}"
                )
                logger.removeHandler(fh)
                fh.close()

        print(f"Summary results saved to: {summary_file}")

    print(f"Per-run files saved in: {results_dir}")
