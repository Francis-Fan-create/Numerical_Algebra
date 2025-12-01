import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import os
import csv
import logging

class PoissonMGSolver:
    """
    Matrix-Free Geometric Multigrid Solver for Poisson: -Delta u = f
    Operates on full vertex grid (N+1) x (N+1) to ensure standard 
    coarsening behavior (N -> N/2).
    """
    def __init__(self, N):
        self.N = N

    # NOTE: This Poisson MG solver targets a standard vertex-centered (N+1 x N+1)
    # grid Poisson problem (i.e., -Delta u = f with Dirichlet BCs). The Inexact
    # Uzawa implementation uses staggered, edge-centered velocities (u,v), so
    # directly re-using this MG solver on u/v arrays without careful mapping
    # between edge-centered and vertex-centered grids can cause inconsistent
    # interior indexing resulting in incorrect velocity approximations ->
    # significantly increased residuals. For now, the Inexact Uzawa solver uses
    # a sparse-matrix CG inner solver on the correctly-indexed interior u/v
    # unknowns (N-1 x N and N x N-1 blocks). If you want to use MG, write a
    # rectangular/edge-grid MG solver which properly restricts and prolongates
    # fields on the staggered grid.

    def relax(self, u, f, h):
        """
        Red-Black Gauss-Seidel Smoother.
        Solves 4u_ij - neighbors = h^2 * f_ij
        """
        h2 = h**2
        rows, cols = u.shape
        
        # We perform updates on internal nodes [1:-1, 1:-1]
        # Dirichlet BCs (u=0 on boundary) are preserved by not writing to edges.
        
        for color in [0, 1]:
            # Vectorized Checkerboard Mask
            # I, J correspond to indices in the u array
            I, J = np.meshgrid(np.arange(1, rows-1), np.arange(1, cols-1), indexing='ij')
            mask = (I + J) % 2 == color
            
            # Extract slices for neighbors
            # u center is u[1:-1, 1:-1]
            u_up    = u[1:-1, 2:]
            u_down  = u[1:-1, 0:-2]
            u_left  = u[0:-2, 1:-1]
            u_right = u[2:,   1:-1]
            
            f_inner = f[1:-1, 1:-1]
            u_inner = u[1:-1, 1:-1]
            
            # GS Update Equation
            # u = 0.25 * (u_left + u_right + u_up + u_down + h^2*f)
            u_inner[mask] = 0.25 * (u_up[mask] + u_down[mask] + 
                                    u_left[mask] + u_right[mask] + 
                                    h2 * f_inner[mask])

    def restrict(self, r):
        """
        Full Weighting Restriction Operator (9-point stencil).
        Maps Fine grid (N+1) -> Coarse grid (N/2 + 1).
        """
        # Dimensions
        N_fine = r.shape[0] - 1
        N_coarse = N_fine // 2
        rc = np.zeros((N_coarse + 1, N_coarse + 1))
        
        # Weighted Average:
        # Coarse node (i,j) aligns with Fine node (2i, 2j).
        # Center weight: 1/4 (at 2i, 2j)
        # Edge weight: 1/8   (at 2i+/-1, 2j and 2i, 2j+/-1)
        # Corner weight: 1/16 (at 2i+/-1, 2j+/-1)
        
        # We vectorizing by taking strided slices of the fine residual r
        
        # Center
        c = r[2:-2:2, 2:-2:2]
        
        # Edges (Up, Down, Left, Right relative to center)
        e = r[2:-2:2, 1:-3:2] + r[2:-2:2, 3:-1:2] + \
            r[1:-3:2, 2:-2:2] + r[3:-1:2, 2:-2:2]
            
        # Corners (TL, TR, BL, BR)
        k = r[1:-3:2, 1:-3:2] + r[3:-1:2, 1:-3:2] + \
            r[1:-3:2, 3:-1:2] + r[3:-1:2, 3:-1:2]
            
        # Apply weights
        # Note: rc internal range is 1:-1
        rc[1:-1, 1:-1] = 0.25 * c + 0.125 * e + 0.0625 * k
        
        return rc

    def prolongate_add(self, u, ec):
        """
        Bilinear Interpolation and Addition.
        u_fine += Prolong(e_coarse)
        """
        # Interpolate ec (Coarse) onto u (Fine)
        # 1. Coincident points: u[2i, 2j] += ec[i,j]
        u[2:-2:2, 2:-2:2] += ec[1:-1, 1:-1]
        
        # 2. Horizontal Edges (Average of horizontal coarse neighbors)
        # Fine u[2i+1, 2j] is between Coarse [i,j] and [i+1,j]
        # ec slice 1:-1 is i. ec slice 2: is i+1.
        
        # Horizontal avg (mapped to odd rows 3, 5...) 
        # indices 1..N_coarse-1 in coarse map to 2..N_fine-2 in fine.
        # Specifically: Fine index 2i+1 corresponds to 0.5*(ec[i] + ec[i+1])
        # Note: ec indices 0..Nc. 
        
        # Efficient expansion using Kronecker-like logic or direct slicing
        # Let's use a temporary expanded array
        ec_ex = np.zeros_like(u)
        
        # Fill even rows/cols (Coincident)
        ec_ex[0::2, 0::2] = ec
        
        # Fill odd rows, even cols (Vertical average of coarse)
        ec_ex[1::2, 0::2] = 0.5 * (ec[:-1, :] + ec[1:, :])
        
        # Fill even rows, odd cols (Horizontal average of coarse)
        ec_ex[0::2, 1::2] = 0.5 * (ec[:, :-1] + ec[:, 1:])
        
        # Fill odd rows, odd cols (Center average of 4 corners)
        ec_ex[1::2, 1::2] = 0.25 * (ec[:-1, :-1] + ec[1:, :-1] + ec[:-1, 1:] + ec[1:, 1:])
        
        u += ec_ex

    def v_cycle(self, u, f, h):
        """Recursive V-Cycle [cite: 146]"""
        # 1. Pre-Smoothing
        for _ in range(2): 
            self.relax(u, f, h)


class RectangularPoissonMGSolver:
    """Multigrid solver for rectangular grids (generalized Poisson).
    Grid arrays include boundary lines: shape=(nrows, ncols) where interior
    unknowns are [1:-1, 1:-1] or other boundary patterns depending on staggered
    layout. For our use in Inexact Uzawa preconditioning, we will expect arrays
    shaped like u-grid (N+1,N) or v-grid (N,N+1).
    """
    def __init__(self, nrows, ncols, nu1=2, nu2=2, L=None):
        self.nrows = nrows
        self.ncols = ncols
        self.nu1 = int(nu1)
        self.nu2 = int(nu2)
        if L is None:
            # levels approximated by min(log2(nrows-1), log2(ncols-1))
            self.L = int(min(np.log2(max(2, nrows-1)), np.log2(max(2, ncols-1))))
        else:
            self.L = int(L)

    def relax(self, u, f, h):
        h2 = h**2
        rows, cols = u.shape
        # Use symmetric sweeps: forward then backward color ordering to help
        # keep preconditioner approximately symmetric.
        for color in [0, 1, 1, 0]:
            I, J = np.meshgrid(np.arange(1, rows-1), np.arange(1, cols-1), indexing='ij')
            mask = (I + J) % 2 == color
            u_up = u[1:-1, 2:]
            u_down = u[1:-1, 0:-2]
            u_left = u[0:-2, 1:-1]
            u_right = u[2:, 1:-1]
            f_inner = f[1:-1, 1:-1]
            u_inner = u[1:-1, 1:-1]
            u_inner[mask] = 0.25 * (u_up[mask] + u_down[mask] + u_left[mask] + u_right[mask] + h2 * f_inner[mask])

    def restrict(self, r):
        nr, nc = r.shape
        Nf_r = nr - 1
        Nf_c = nc - 1
        Nc_r = Nf_r // 2
        Nc_c = Nf_c // 2
        rc = np.zeros((Nc_r + 1, Nc_c + 1))
        for i in range(Nc_r + 1):
            for j in range(Nc_c + 1):
                i_f, j_f = 2*i, 2*j
                val = 0.25 * r[i_f, j_f]
                # edges
                if i_f - 1 >= 0:
                    val += 0.125 * r[i_f - 1, j_f]
                if i_f + 1 < nr:
                    val += 0.125 * r[i_f + 1, j_f]
                if j_f - 1 >= 0:
                    val += 0.125 * r[i_f, j_f - 1]
                if j_f + 1 < nc:
                    val += 0.125 * r[i_f, j_f + 1]
                # corners
                if i_f - 1 >= 0 and j_f - 1 >= 0:
                    val += 0.0625 * r[i_f - 1, j_f - 1]
                if i_f - 1 >= 0 and j_f + 1 < nc:
                    val += 0.0625 * r[i_f - 1, j_f + 1]
                if i_f + 1 < nr and j_f - 1 >= 0:
                    val += 0.0625 * r[i_f + 1, j_f - 1]
                if i_f + 1 < nr and j_f + 1 < nc:
                    val += 0.0625 * r[i_f + 1, j_f + 1]
                rc[i, j] = val
        return rc

    def prolongate_add(self, u, ec):
        # Bilinear interpolation/prolongation with edge-aware padding for
        # rectangular coarse arrays.
        ec_ex = np.zeros_like(u)
        # Place coincident coarse values
        ec_ex[0::2, 0::2] = ec

        # Vertical averages for odd rows, even cols
        if ec.shape[0] > 1:
            # pad last row to ensure shapes match for rectangular arrays
            ec_pad_v = np.pad(ec, ((0,1), (0,0)), mode='edge')
            vv = 0.5 * (ec_pad_v[:-1, :] + ec_pad_v[1:, :])
            tr_v = ec_ex[1::2, 0::2]
            ec_ex[1::2, 0::2][:vv.shape[0], :vv.shape[1]] = vv[:tr_v.shape[0], :tr_v.shape[1]]

        # Horizontal averages for even rows, odd cols
        if ec.shape[1] > 1:
            # pad right edge to avoid shape mismatch when coarse columns < fine odd columns
            ec_pad = np.pad(ec, ((0, 0), (0, 1)), mode='edge')
            hh = 0.5 * (ec_pad[:, :-1] + ec_pad[:, 1:])
            target = ec_ex[0::2, 1::2]
            # ensure shape match by trimming
            ec_ex[0::2, 1::2][:target.shape[0], :target.shape[1]] = hh[:target.shape[0], :target.shape[1]]

        # Centers (odd rows, odd cols): compute via padded interior average
        if ec.shape[0] > 1 and ec.shape[1] > 1:
            ec_ppad = np.pad(ec, ((0, 1), (0, 1)), mode='edge')
            cc = 0.25 * (ec_ppad[:-1, :-1] + ec_ppad[1:, :-1] + ec_ppad[:-1, 1:] + ec_ppad[1:, 1:])
            # target area
            tr = ec_ex[1::2, 1::2]
            ec_ex[1::2, 1::2][:tr.shape[0], :tr.shape[1]] = cc[:tr.shape[0], :tr.shape[1]]

        u += ec_ex

    def v_cycle(self, u, f, h, level=0):
        # Pre-smoothing
        for _ in range(self.nu1):
            self.relax(u, f, h)
        nr, nc = u.shape
        if nr <= 4 or nc <= 4 or level >= (self.L - 1):
            # direct relaxation with a bunch of smoothing steps
            for _ in range(10):
                self.relax(u, f, h)
            return
        # residual - compute Laplacian using neighbor slices (safe for rectangular shapes)
        u_mid = u[1:-1, 1:-1]
        u_up = u[1:-1, 2:]
        u_down = u[1:-1, 0:-2]
        u_left = u[0:-2, 1:-1]
        u_right = u[2:, 1:-1]
        lap_u = (u_up + u_down + u_left + u_right - 4*u_mid) / h**2
        r = np.zeros_like(u)
        r[1:-1, 1:-1] = f[1:-1, 1:-1] - (-lap_u)
        # restrict
        rc = self.restrict(r)
        # Solve coarse grid Ae = rc (with rc used as f on coarse grid)
        ec = np.zeros_like(rc)
        self.v_cycle(ec, rc, 2*h, level+1)
        # prolongate-add
        self.prolongate_add(u, ec)
        # post-smoothing
        for _ in range(self.nu2):
            self.relax(u, f, h)
        
        # Base Case: Coarsest grid (e.g., 2x2 or 4x4)
        if u.shape[0] <= 4:
            return

        # 2. Residual Calculation: r = f - (-Lap(u))
        u_mid = u[1:-1, 1:-1]
        lap_u = (u[0:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, 0:-2] + u[1:-1, 2:] - 4*u_mid) / h**2
        r = np.zeros_like(u)
        r[1:-1, 1:-1] = f[1:-1, 1:-1] - (-lap_u)
        
        # 3. Restriction
        rc = self.restrict(r)
        
        # 4. Recursion (Solve Ae = r on coarse grid)
        ec = np.zeros_like(rc)
        self.v_cycle(ec, rc, 2*h)
        
        # 5. Prolongation and Correction
        self.prolongate_add(u, ec)
        
        # 6. Post-Smoothing
        for _ in range(2): 
            self.relax(u, f, h)


class InexactUzawaSolver:
    def __init__(self, N, alpha=None, tau=1e-6, inner_iters=2, use_mg=False, nu1=2, nu2=2, L=None, debug=False):
        """
        Inexact Uzawa Solver .
        N: Grid resolution.
        """
        self.N = N
        self.h = 1.0 / N
        # Choose alpha proportional to h^2 by default (stable step for pressure updates)
        if alpha is None:
            self.alpha = 0.125 * self.h**2
        else:
            self.alpha = alpha
        # Number of inner solver iterations (CG or MG) to perform per outer step
        self.inner_iters = int(inner_iters)
        # Toggle: use MG inner solvers (True) or sparse matrix-based CG (False)
        self.use_mg = bool(use_mg)
        # Debug mode: print additional diagnostics
        self.debug = bool(debug)
        self.tau = float(tau)
        self.nu1 = int(nu1)
        self.nu2 = int(nu2)
        self.L = None if L is None else int(L)
        
        # Data Structures:
        # Use staggered MAC storage consistent with other modules
        # u: vertical edges, size (N+1) x N
        # v: horizontal edges, size N x (N+1)
        self.u = np.zeros((N+1, N))
        self.v = np.zeros((N, N+1))
        self.p = np.zeros((N, N))
        
        # Inner solvers for preconditioning
        self.mg_solver = PoissonMGSolver(N)
        # Rectangular MG for u and v inner solves if using MG preconditioning
        self.mg_u = RectangularPoissonMGSolver(N+1, N, nu1=self.nu1, nu2=self.nu2, L=self.L)
        self.mg_v = RectangularPoissonMGSolver(N, N+1, nu1=self.nu1, nu2=self.nu2, L=self.L)
        # Build sparse operators (A_u, A_v) for inner CG solves (ensures consistent discretization)
        self.build_operators()
        
        self.init_rhs()
        # Build cheap diagonal preconditioner for CG inner solvers (fast)
        self.Mu_diag = None
        self.Mv_diag = None
        if self.A_u is not None:
            diag = self.A_u.diagonal()
            inv_diag = 1.0 / (diag + 1e-16)
            self.Mu_diag = spla.LinearOperator(self.A_u.shape, matvec=lambda x, inv=inv_diag: inv * x)
        if self.A_v is not None:
            diag = self.A_v.diagonal()
            inv_diag = 1.0 / (diag + 1e-16)
            self.Mv_diag = spla.LinearOperator(self.A_v.shape, matvec=lambda x, inv=inv_diag: inv * x)

    def init_rhs(self):
        """Initialize exact force terms [cite: 133-134]"""
        # u-grid physical coordinates (for f) - full (N+1)xN
        I, J = np.meshgrid(np.arange(self.N+1), np.arange(self.N), indexing='ij')
        X_u = I * self.h
        Y_u = (J + 0.5) * self.h
        self.f = -4 * np.pi**2 * (2 * np.cos(2*np.pi*X_u) - 1) * np.sin(2*np.pi*Y_u) + X_u**2

        # v-grid physical coordinates (for g) - full N x (N+1)
        I_v, J_v = np.meshgrid(np.arange(self.N), np.arange(self.N+1), indexing='ij')
        X_v = (I_v + 0.5) * self.h
        Y_v = J_v * self.h
        self.g = 4 * np.pi**2 * (2 * np.cos(2*np.pi*Y_v) - 1) * np.sin(2*np.pi*X_v)

    def build_operators(self):
        """
        Construct sparse Laplacian operators for u and v inner solves.
        A_u for u interior unknowns (N-1 x N), A_v for v interior unknowns (N x N-1).
        """
        h = self.h
        N = self.N
        h2 = h**2

        def laplace_1d(n_pts):
            # tri-diagonal Laplacian operator for 1D
            if n_pts <= 0:
                return sp.csr_matrix((0, 0))
            main = 2 * np.ones(n_pts)
            off = -1 * np.ones(n_pts - 1)
            return sp.diags([off, main, off], [-1, 0, 1], shape=(n_pts, n_pts)) / h2

        # A_u: (N-1) x N grid => kron(Dxx, I_N) + kron(I_(N-1), Dyy)
        Dxx_u = laplace_1d(N - 1)
        Dyy_u = laplace_1d(N)
        if Dxx_u.shape[0] > 0 and Dyy_u.shape[0] > 0:
            self.A_u = sp.kron(Dxx_u, sp.eye(N)) + sp.kron(sp.eye(N - 1), Dyy_u)
        else:
            self.A_u = None

        # A_v: N x (N-1) grid
        Dxx_v = laplace_1d(N)
        Dyy_v = laplace_1d(N - 1)
        if Dxx_v.shape[0] > 0 and Dyy_v.shape[0] > 0:
            self.A_v = sp.kron(Dxx_v, sp.eye(N - 1)) + sp.kron(sp.eye(N), Dyy_v)
        else:
            self.A_v = None

    def compute_residual_norm(self, p_override=None):
        """Compute full Stokes residual ||r_h||_2 / ||r_0||_2 logic."""
        h = self.h
        
        # 1. Momentum Residuals (Interior Points)
        # res_u = f - (-Lap(u) + p_x)
        # u is stored in (N+1, N)
        u_ext = self.u
        # Laplacian at interior points [1:-1, :] which is (N-1, N)
        u_pad = np.pad(u_ext, ((0,0), (1,1)), 'constant')
        lap_u = (u_ext[0:-2, :] + u_ext[2:, :] + u_pad[1:-1, 0:-2] + u_pad[1:-1, 2:] - 4*u_ext[1:-1, :]) / h**2
        
        # p_x: pressure gradient (N-1, N)
        p_arr = self.p if p_override is None else p_override
        px = (p_arr[1:, :] - p_arr[:-1, :]) / h
        
        # Residual on interior points (N-1, N)
        r_u = self.f[1:-1, :] - (-lap_u + px)
        
        # res_v = g - (-Lap(v) + p_y)
        # v is stored in (N, N+1)
        v_ext = self.v
        v_pad = np.pad(v_ext, ((1,1), (0,0)), 'constant')
        lap_v = (v_pad[0:-2, 1:-1] + v_pad[2:, 1:-1] + v_ext[:, 0:-2] + v_ext[:, 2:] - 4*v_ext[:, 1:-1]) / h**2
        
        # p_y: pressure gradient (N, N-1)
        py = (p_arr[:, 1:] - p_arr[:, :-1]) / h
        
        # Residual on interior points (N, N-1)
        r_v = self.g[:, 1:-1] - (-lap_v + py)
        
        # 2. Divergence Residual (Constraint)
        # div = (u_x + v_y) on pressure grid
        div = (self.u[1:self.N+1, :self.N] - self.u[:self.N, :self.N])/h + (self.v[:self.N, 1:self.N+1] - self.v[:self.N, :self.N])/h
        
        total_sq = np.sum(r_u**2) + np.sum(r_v**2) + np.sum(div**2)
        return np.sqrt(total_sq)

    def solve(self, logger=None):
        if logger is not None:
            logger.info(f"--- Solving Inexact Uzawa (N={self.N}) ---")
        else:
            print(f"--- Solving Inexact Uzawa (N={self.N}) ---")
        t0 = time.time()
        
        # Initial Residual (Approximated by Force norm)
        r0 = self.compute_residual_norm()
        if r0 < 1e-12: r0 = 1.0
        if logger is not None:
            logger.info(f"Init Residual: {r0:.4e}")
        
        tol = 1e-8
        # Report which inner solver is used
        if logger is not None:
            solver_name = 'MG-GMRES' if self.use_mg else 'CG'
            logger.info(f"Inner solver: {solver_name}, inner_iters={self.inner_iters}, alpha={self.alpha:.6e}")
        rel_res = 1.0
        res_history = [rel_res]
        k = 0
        
        # Build preconditioners if requested
        M_u = None
        M_v = None
        if self.use_mg and self.A_u is not None and self.A_v is not None:
            # MG preconditioner for u
            def pre_u(x):
                # map x -> fine grid extended RHS
                b_ext = np.zeros((self.N + 1, self.N))
                b_ext[1:-1, :] = x.reshape((self.N - 1, self.N))
                u_ext = np.zeros_like(b_ext)
                self.mg_u.v_cycle(u_ext, b_ext, self.h)
                return u_ext[1:-1, :].flatten()
            M_u = spla.LinearOperator(self.A_u.shape, matvec=pre_u)
            # MG preconditioner for v
            def pre_v(x):
                b_ext = np.zeros((self.N, self.N + 1))
                b_ext[:, 1:-1] = x.reshape((self.N, self.N - 1))
                v_ext = np.zeros_like(b_ext)
                self.mg_v.v_cycle(v_ext, b_ext, self.h)
                return v_ext[:, 1:-1].flatten()
            M_v = spla.LinearOperator(self.A_v.shape, matvec=pre_v)

        # Iteration Loop [cite: 108-111]
        # Iteration loop
        logger.info(f"Starting solve with initial alpha={self.alpha:.6e}") if logger is not None else print(f"Starting solve with initial alpha={self.alpha:.6e}")
        while rel_res > tol and k < 200:
            # 1. RHS for Poisson Steps
            # rhs_u = f - p_x
            px = (self.p[1:, :] - self.p[:-1, :]) / self.h
            rhs_u = self.f.copy()
            rhs_u[1:-1, :] -= px
            
            # rhs_v = g - p_y
            py = (self.p[:, 1:] - self.p[:, :-1]) / self.h
            rhs_v = self.g.copy()
            rhs_v[:self.N, 1:-1] -= py
            
            # 2. Approximate Velocity Solve (CG or MG)
            # (Optional diagnostic: compute momentum/divergence norms before inner solve)
            if self.debug and logger is not None:
                r_u_vec = None
                if self.A_u is not None:
                    u_interior = self.u[1:-1, :self.N]
                    u_vec = u_interior.flatten()
                    b_u = rhs_u[1:-1, :self.N].flatten()
                    r_u_vec = b_u - self.A_u.dot(u_vec)
                r_v_vec = None
                if self.A_v is not None:
                    v_interior = self.v[:self.N, 1:-1]
                    v_vec = v_interior.flatten()
                    b_v = rhs_v[:self.N, 1:-1].flatten()
                    r_v_vec = b_v - self.A_v.dot(v_vec)
                div_arr = (self.u[1:self.N+1, :self.N] - self.u[:self.N, :self.N]) / self.h + (self.v[:self.N, 1:self.N+1] - self.v[:self.N, :self.N]) / self.h
                div_vec = div_arr.flatten()
                if r_u_vec is not None and r_v_vec is not None:
                    logger.info(f"Before inner: ||r_u||={np.linalg.norm(r_u_vec):.3e}, ||r_v||={np.linalg.norm(r_v_vec):.3e}, ||div||={np.linalg.norm(div_vec):.3e}")
            # Use sparse CG inner solves on properly-mapped interior unknowns
            if self.A_u is not None and self.A_v is not None and not self.use_mg:
                for _ in range(max(1, self.inner_iters)):
                    # Solve for u (interior only) using preconditioned CG
                    u_interior = self.u[1:-1, :self.N]
                    u_vec = u_interior.flatten()
                    b_u = rhs_u[1:-1, :self.N].flatten()
                    if self.debug and self.N <= 64:
                        # For debugging small N: direct SPARSE solve to check exact behavior
                        sol_u = spla.spsolve(self.A_u.tocsr(), b_u)
                        info_u = 0
                    else:
                        sol_u, info_u = spla.cg(self.A_u, b_u, x0=u_vec, rtol=self.tau, M=self.Mu_diag)
                    if info_u != 0:
                        logging.warning(f"CG inner solve for A_u did not converge: info={info_u}")
                    u_interior[:, :] = sol_u.reshape(u_interior.shape)

                    # Solve for v (interior only) using preconditioned CG
                    v_interior = self.v[:self.N, 1:-1]
                    v_vec = v_interior.flatten()
                    b_v = rhs_v[:self.N, 1:-1].flatten()
                    if self.debug and self.N <= 64:
                        sol_v = spla.spsolve(self.A_v.tocsr(), b_v)
                        info_v = 0
                    else:
                        sol_v, info_v = spla.cg(self.A_v, b_v, x0=v_vec, rtol=self.tau, M=self.Mv_diag)
                    if info_v != 0:
                        logging.warning(f"CG inner solve for A_v did not converge: info={info_v}")
                    v_interior[:, :] = sol_v.reshape(v_interior.shape)
            else:
                # Use preconditioned GMRES with MG preconditioner if requested, otherwise fallback to MG v_cycle
                if self.A_u is not None and self.A_v is not None and self.use_mg:
                    for _ in range(max(1, self.inner_iters)):
                        u_interior = self.u[1:-1, :self.N]
                        u_vec = u_interior.flatten()
                        b_u = rhs_u[1:-1, :self.N].flatten()
                        # Try GMRES with the MG preconditioner
                        sol_u, info_u = spla.gmres(self.A_u, b_u, x0=u_vec, rtol=self.tau, M=M_u)
                        if info_u != 0:
                            logging.warning(f"PCG inner solve for A_u did not converge: info={info_u}")
                        u_interior[:, :] = sol_u.reshape(u_interior.shape)

                        v_interior = self.v[:self.N, 1:-1]
                        v_vec = v_interior.flatten()
                        b_v = rhs_v[:self.N, 1:-1].flatten()
                        sol_v, info_v = spla.gmres(self.A_v, b_v, x0=v_vec, rtol=self.tau, M=M_v)
                        if info_v != 0:
                            logging.warning(f"PCG inner solve for A_v did not converge: info={info_v}")
                        v_interior[:, :] = sol_v.reshape(v_interior.shape)
                else:
                    # fallback to simple rectangular MG on u and v grids
                    for _ in range(max(1, self.inner_iters)):
                        self.mg_u.v_cycle(self.u, rhs_u, self.h)
                        self.mg_v.v_cycle(self.v, rhs_v, self.h)
            # optional: compute residual AFTER inner solves (debug only)
            if self.debug and logger is not None:
                r_after_mg = self.compute_residual_norm()
                # Recompute r_u, r_v, div after inner solves
                u_interior = self.u[1:-1, :self.N]
                u_vec = u_interior.flatten()
                b_u = rhs_u[1:-1, :self.N].flatten()
                r_u_vec_after = b_u - (self.A_u.dot(u_vec) if self.A_u is not None else 0)
                v_interior = self.v[:self.N, 1:-1]
                v_vec = v_interior.flatten()
                b_v = rhs_v[:self.N, 1:-1].flatten()
                r_v_vec_after = b_v - (self.A_v.dot(v_vec) if self.A_v is not None else 0)
                div_arr_after = (self.u[1:self.N+1, :self.N] - self.u[:self.N, :self.N]) / self.h + (self.v[:self.N, 1:self.N+1] - self.v[:self.N, :self.N]) / self.h
                div_vec_after = div_arr_after.flatten()
                logger.info(f"After inner: ||r_u||={np.linalg.norm(r_u_vec_after):.3e}, ||r_v||={np.linalg.norm(r_v_vec_after):.3e}, ||div||={np.linalg.norm(div_vec_after):.3e}")
            
            # 3. Update Pressure. Note: Use the same sign convention as B^T (divergence)
            # In the Kronecker operator implementation (Bx/By) we have div = Bx^T u + By^T v.
            # The finite-difference divergence computed below is the negative of Bx^T u,
            # so add a minus sign to get the same sign as 'Bx.T.dot(u)' used elsewhere.
            div = -((self.u[1:self.N+1, :self.N] - self.u[:self.N, :self.N]) / self.h + (self.v[:self.N, 1:self.N+1] - self.v[:self.N, :self.N]) / self.h)

            # Apply the pressure update with a fixed alpha (no adaptive halving).
            # Adaptive halving based on the full residual can cause alpha to shrink
            # to machine precision (and stagnation) because r_candidate is
            # computed without re-solving velocity. Using a fixed alpha is
            # consistent with classical Uzawa/Inexact Uzawa methods.
            self.p = self.p + self.alpha * div
            # Ensure zero-mean pressure after update
            self.p -= np.mean(self.p)
            
            # 4. Check Convergence
            r_curr = self.compute_residual_norm()
            rel_res = r_curr / r0
            res_history.append(rel_res)
            
            k += 1
            if k % 10 == 0:
                msg = f"Iter {k}: Rel Res = {rel_res:.4e}"
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)
            if self.debug and logger is not None:
                # log final norms
                u_interior = self.u[1:-1, :self.N]
                u_vec = u_interior.flatten()
                b_u = rhs_u[1:-1, :self.N].flatten()
                r_u_vec = b_u - (self.A_u.dot(u_vec) if self.A_u is not None else 0)
                v_interior = self.v[:self.N, 1:-1]
                v_vec = v_interior.flatten()
                b_v = rhs_v[:self.N, 1:-1].flatten()
                r_v_vec = b_v - (self.A_v.dot(v_vec) if self.A_v is not None else 0)
                div_arr = (self.u[1:self.N+1, :self.N] - self.u[:self.N, :self.N]) / self.h + (self.v[:self.N, 1:self.N+1] - self.v[:self.N, :self.N]) / self.h
                div_vec = div_arr.flatten()
                logger.info(f"End iter {k}: ||r_u||={np.linalg.norm(r_u_vec):.3e}, ||r_v||={np.linalg.norm(r_v_vec):.3e}, ||div||={np.linalg.norm(div_vec):.3e}, rel_res={rel_res:.3e}")

        cpu_time = time.time() - t0
        
        # Error Calculation [cite: 149]
        # Exact solution
        I, J = np.meshgrid(np.arange(self.N+1), np.arange(self.N), indexing='ij')
        X_u = I * self.h; Y_u = (J+0.5) * self.h
        u_ex = (1 - np.cos(2*np.pi*X_u)) * np.sin(2*np.pi*Y_u)
        
        I, J = np.meshgrid(np.arange(self.N), np.arange(self.N+1), indexing='ij')
        X_v = (I+0.5) * self.h; Y_v = J * self.h
        v_ex = -(1 - np.cos(2*np.pi*Y_v)) * np.sin(2*np.pi*X_v)
        
        u_err = np.sum((self.u[1:-1, :self.N] - u_ex[1:-1, :])**2)
        v_err = np.sum((self.v[:self.N, 1:-1] - v_ex[:, 1:-1])**2)
        e_N = self.h * np.sqrt(u_err + v_err)
        if logger is not None:
            logger.info(f"Finished: Iters={k}, CPU(s)={cpu_time:.6f}, Error={e_N:.6e}")
        
        return k, cpu_time, e_N, res_history

if __name__ == "__main__":
    # Task 3: Run for N = 64, 128, 256, 512, 1024, 2048
    # 2048 fits in memory due to Matrix-Free implementation.
    # We demonstrate a subset here.
    N_list = [64, 128, 256, 512, 1024, 2048]
    
    print(f"{'N':<6} | {'Iters':<8} | {'CPU(s)':<10} | {'Error':<12}")
    print("-" * 42)
    
    # Create base results directory and script-specific folder
    results_base = os.path.join(os.path.dirname(__file__), "results")
    script_name = "inexact_uzawa"
    results_dir = os.path.join(results_base, script_name)
    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []
    summary_file = os.path.join(results_dir, "results_inexact_uzawa.csv")
    with open(summary_file, "w", newline='') as sf:
        writer = csv.writer(sf)
        writer.writerow(["N", "Iters", "CPU(s)", "Error", "alpha", "tau", "nu1", "nu2", "InitialRelRes", "FinalRelRes", "ResidualFile"])
        for N in N_list:
            solver = InexactUzawaSolver(N)
            # per-run subfolder
            run_dir = os.path.join(results_dir, f"N{N}")
            os.makedirs(run_dir, exist_ok=True)
            # set up a per-run logger
            log_file = os.path.join(run_dir, f"inexact_uzawa_N{N}.log")
            logger = logging.getLogger(f"inexact_uzawa_{N}")
            # remove old handlers
            for h in list(logger.handlers):
                logger.removeHandler(h)
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info(f"Run Inexact Uzawa: N={N}, h={solver.h}")
            k, t, e, res_hist = solver.solve(logger=logger)
            # Write residual history to CSV
            residuals_file = os.path.join(run_dir, f"residuals_inexact_uzawa_N{N}.csv")
            with open(residuals_file, "w", newline='') as rf:
                w2 = csv.writer(rf)
                w2.writerow(["iter", "rel_res"])
                for idx, val in enumerate(res_hist):
                    w2.writerow([idx, val])
                    # incrementally log residuals too
                    logger.info(f"Iter {idx}: Rel Res = {val:.6e}")

            summary_row = (N, k, t, e, solver.alpha, solver.tau, solver.nu1, solver.nu2, res_hist[0], res_hist[-1], os.path.basename(residuals_file))
            writer.writerow(summary_row)
            sf.flush()
            summary_rows.append(summary_row)
            print(f"{N:<6} | {k:<8} | {t:<10.2f} | {e:<12.4e}")
            # log summary for this run
            logger.info(f"Summary: N={N}, Iters={k}, CPU(s)={t:.6f}, Error={e:.6e}, ResidualFile={os.path.basename(residuals_file)}")
            # close file handler
            logger.removeHandler(fh)
            fh.close()
    # File already written incrementally inside 'with' context
        print(f"Summary results saved to: {summary_file}")
    print(f"Per-run files saved in: {results_dir}/*/N<value>")