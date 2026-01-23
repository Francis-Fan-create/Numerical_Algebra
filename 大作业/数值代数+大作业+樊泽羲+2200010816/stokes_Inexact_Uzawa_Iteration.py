import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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
    def _rect_relax_numba(u, f, h, bc_x_dir, bc_x_neu, bc_y_dir, bc_y_neu):
        h2 = h * h
        rows, cols = u.shape

        if bc_x_dir:
            for j in range(cols):
                u[0, j] = 0.0
                u[rows - 1, j] = 0.0
        if bc_y_dir:
            for i in range(rows):
                u[i, 0] = 0.0
                u[i, cols - 1] = 0.0

        colors = (0, 1, 1, 0)
        for color in colors:
            if rows > 2 and cols > 2:
                for i in range(1, rows - 1):
                    for j in range(1, cols - 1):
                        if ((i + j) & 1) == color:
                            u[i, j] = 0.25 * (
                                u[i, j + 1]
                                + u[i, j - 1]
                                + u[i - 1, j]
                                + u[i + 1, j]
                                + h2 * f[i, j]
                            )

            if bc_y_neu and cols >= 2 and rows > 2:
                for i in range(1, rows - 1):
                    if ((i + 0) & 1) == color:
                        u[i, 0] = (u[i - 1, 0] + u[i + 1, 0] + u[i, 1] + h2 * f[i, 0]) / 3.0
                    if ((i + (cols - 1)) & 1) == color:
                        u[i, cols - 1] = (u[i - 1, cols - 1] + u[i + 1, cols - 1] + u[i, cols - 2] + h2 * f[i, cols - 1]) / 3.0

            if bc_x_neu and rows >= 2 and cols > 2:
                for j in range(1, cols - 1):
                    if ((0 + j) & 1) == color:
                        u[0, j] = (u[1, j] + u[0, j - 1] + u[0, j + 1] + h2 * f[0, j]) / 3.0
                    if (((rows - 1) + j) & 1) == color:
                        u[rows - 1, j] = (u[rows - 2, j] + u[rows - 1, j - 1] + u[rows - 1, j + 1] + h2 * f[rows - 1, j]) / 3.0

    @njit(cache=True)
    def _rect_compute_residual_inplace_numba(r, u, f, h, bc_x_neu, bc_y_neu):
        rows, cols = u.shape
        h2 = h * h
        for i in range(rows):
            for j in range(cols):
                r[i, j] = 0.0

        if rows > 2 and cols > 2:
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    lap = (
                        u[i - 1, j]
                        + u[i + 1, j]
                        + u[i, j - 1]
                        + u[i, j + 1]
                        - 4.0 * u[i, j]
                    ) / h2
                    r[i, j] = f[i, j] - (-lap)

        if bc_y_neu and rows > 2:
            for i in range(1, rows - 1):
                lap = (u[i - 1, 0] + u[i + 1, 0] + u[i, 1] + u[i, 0] - 4.0 * u[i, 0]) / h2
                r[i, 0] = f[i, 0] - (-lap)
                lap = (u[i - 1, cols - 1] + u[i + 1, cols - 1] + u[i, cols - 2] + u[i, cols - 1] - 4.0 * u[i, cols - 1]) / h2
                r[i, cols - 1] = f[i, cols - 1] - (-lap)

        if bc_x_neu and cols > 2:
            for j in range(1, cols - 1):
                lap = (u[1, j] + u[0, j - 1] + u[0, j + 1] + u[0, j] - 4.0 * u[0, j]) / h2
                r[0, j] = f[0, j] - (-lap)
                lap = (u[rows - 2, j] + u[rows - 1, j - 1] + u[rows - 1, j + 1] + u[rows - 1, j] - 4.0 * u[rows - 1, j]) / h2
                r[rows - 1, j] = f[rows - 1, j] - (-lap)

    @njit(cache=True)
    def _rect_restrict_numba(r, bc_x_neu, bc_y_neu):
        nr, nc = r.shape
        Nf_r = nr - 1
        Nf_c = nc - 1
        Nc_r = Nf_r // 2
        Nc_c = Nf_c // 2
        rc = np.zeros((Nc_r + 1, Nc_c + 1))

        if nr >= 5 and nc >= 5 and Nc_r >= 1 and Nc_c >= 1:
            for i in range(1, Nc_r):
                i_f = 2 * i
                for j in range(1, Nc_c):
                    j_f = 2 * j
                    c = r[i_f, j_f]
                    e = (
                        r[i_f, j_f - 1]
                        + r[i_f, j_f + 1]
                        + r[i_f - 1, j_f]
                        + r[i_f + 1, j_f]
                    )
                    k = (
                        r[i_f - 1, j_f - 1]
                        + r[i_f + 1, j_f - 1]
                        + r[i_f - 1, j_f + 1]
                        + r[i_f + 1, j_f + 1]
                    )
                    rc[i, j] = 0.25 * c + 0.125 * e + 0.0625 * k

        if bc_y_neu and Nc_r >= 1:
            for i in range(1, Nc_r):
                i_f = 2 * i
                rc[i, 0] = 0.5 * r[i_f, 0] + 0.25 * (r[i_f - 1, 0] + r[i_f + 1, 0])
                rc[i, Nc_c] = 0.5 * r[i_f, nc - 1] + 0.25 * (r[i_f - 1, nc - 1] + r[i_f + 1, nc - 1])

        if bc_x_neu and Nc_c >= 1:
            for j in range(1, Nc_c):
                j_f = 2 * j
                rc[0, j] = 0.5 * r[0, j_f] + 0.25 * (r[0, j_f - 1] + r[0, j_f + 1])
                rc[Nc_r, j] = 0.5 * r[nr - 1, j_f] + 0.25 * (r[nr - 1, j_f - 1] + r[nr - 1, j_f + 1])

        if nr >= 1 and nc >= 1:
            rc[0, 0] = r[0, 0]
            rc[0, Nc_c] = r[0, nc - 1]
            rc[Nc_r, 0] = r[nr - 1, 0]
            rc[Nc_r, Nc_c] = r[nr - 1, nc - 1]
        return rc

    @njit(cache=True)
    def _rect_prolongate_add_numba(u, ec):
        nr, nc = u.shape
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
                u[I, J] += (
                    wi0 * wj0 * ec[i, j]
                    + wi1 * wj0 * ec[i1, j]
                    + wi0 * wj1 * ec[i, j1]
                    + wi1 * wj1 * ec[i1, j1]
                )

class PoissonMGSolver:
    """
    Matrix-Free Geometric Multigrid Solver for Poisson: -Delta u = f
    Operates on full vertex grid (N+1) x (N+1) to ensure standard 
    coarsening behavior (N -> N/2).
    """
    def __init__(self, N):
        self.N = N

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
    layout. Supports mixed BCs via bc_x/bc_y ('dirichlet' or 'neumann').
    """
    def __init__(self, nrows, ncols, nu1=2, nu2=2, L=None, bc_x="dirichlet", bc_y="dirichlet"):
        self.nrows = nrows
        self.ncols = ncols
        self.nu1 = int(nu1)
        self.nu2 = int(nu2)
        self.bc_x = bc_x
        self.bc_y = bc_y
        self._bc_x_dir = self.bc_x == "dirichlet"
        self._bc_x_neu = self.bc_x == "neumann"
        self._bc_y_dir = self.bc_y == "dirichlet"
        self._bc_y_neu = self.bc_y == "neumann"
        if L is None:
            # levels approximated by min(log2(nrows-1), log2(ncols-1))
            self.L = int(min(np.log2(max(2, nrows-1)), np.log2(max(2, ncols-1))))
        else:
            self.L = int(L)

        self._mask0 = None
        self._mask1 = None
        if nrows > 2 and ncols > 2:
            I, J = np.meshgrid(np.arange(1, nrows-1), np.arange(1, ncols-1), indexing='ij')
            self._mask0 = (I + J) % 2 == 0
            self._mask1 = ~self._mask0
        self._i_idx = np.arange(1, nrows-1) if nrows > 2 else None
        self._j_idx = np.arange(1, ncols-1) if ncols > 2 else None
        self._r_buf = None

    def relax(self, u, f, h):
        if USE_NUMBA:
            _rect_relax_numba(u, f, h, self._bc_x_dir, self._bc_x_neu, self._bc_y_dir, self._bc_y_neu)
            return
        h2 = h**2
        rows, cols = u.shape

        # Enforce Dirichlet boundaries
        if self.bc_x == "dirichlet":
            u[0, :] = 0.0
            u[-1, :] = 0.0
        if self.bc_y == "dirichlet":
            u[:, 0] = 0.0
            u[:, -1] = 0.0

        # Use symmetric sweeps: forward then backward color ordering
        for color in [0, 1, 1, 0]:
            # Interior updates (exclude boundary lines)
            if rows > 2 and cols > 2:
                mask = self._mask0 if color == 0 else self._mask1
                u_up = u[1:-1, 2:]
                u_down = u[1:-1, 0:-2]
                u_left = u[0:-2, 1:-1]
                u_right = u[2:, 1:-1]
                f_inner = f[1:-1, 1:-1]
                u_inner = u[1:-1, 1:-1]
                u_inner[mask] = 0.25 * (u_up[mask] + u_down[mask] + u_left[mask] + u_right[mask] + h2 * f_inner[mask])

            # Neumann boundaries: update with 3-point stencil
            if self.bc_y == "neumann" and cols >= 2 and rows > 2:
                i_idx = self._i_idx
                # bottom (j=0)
                mask = ((i_idx + 0) % 2) == color
                u_row = u[1:-1, 0]
                rhs = f[1:-1, 0]
                u_row[mask] = (u[0:-2, 0][mask] + u[2:, 0][mask] + u[1:-1, 1][mask] + h2 * rhs[mask]) / 3.0
                # top (j=cols-1)
                mask = ((i_idx + (cols-1)) % 2) == color
                u_row = u[1:-1, -1]
                rhs = f[1:-1, -1]
                u_row[mask] = (u[0:-2, -1][mask] + u[2:, -1][mask] + u[1:-1, -2][mask] + h2 * rhs[mask]) / 3.0

            if self.bc_x == "neumann" and rows >= 2 and cols > 2:
                j_idx = self._j_idx
                # left (i=0)
                mask = ((0 + j_idx) % 2) == color
                u_col = u[0, 1:-1]
                rhs = f[0, 1:-1]
                u_col[mask] = (u[1, 1:-1][mask] + u[0, 0:-2][mask] + u[0, 2:][mask] + h2 * rhs[mask]) / 3.0
                # right (i=rows-1)
                mask = (((rows-1) + j_idx) % 2) == color
                u_col = u[-1, 1:-1]
                rhs = f[-1, 1:-1]
                u_col[mask] = (u[-2, 1:-1][mask] + u[-1, 0:-2][mask] + u[-1, 2:][mask] + h2 * rhs[mask]) / 3.0

    def compute_residual(self, u, f, h):
        """Compute residual r = f - (-Laplace(u)) with mixed BCs."""
        if USE_NUMBA:
            if self._r_buf is None or self._r_buf.shape != u.shape:
                self._r_buf = np.zeros_like(u)
            _rect_compute_residual_inplace_numba(self._r_buf, u, f, h, self._bc_x_neu, self._bc_y_neu)
            return self._r_buf
        rows, cols = u.shape
        r = np.zeros_like(u)

        # interior
        if rows > 2 and cols > 2:
            u_mid = u[1:-1, 1:-1]
            lap = (u[0:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, 0:-2] + u[1:-1, 2:] - 4 * u_mid) / h**2
            r[1:-1, 1:-1] = f[1:-1, 1:-1] - (-lap)

        # Neumann boundaries
        if self.bc_y == "neumann" and rows > 2:
            # bottom j=0
            lap = (u[0:-2, 0] + u[2:, 0] + u[1:-1, 1] + u[1:-1, 0] - 4 * u[1:-1, 0]) / h**2
            r[1:-1, 0] = f[1:-1, 0] - (-lap)
            # top j=cols-1
            lap = (u[0:-2, -1] + u[2:, -1] + u[1:-1, -2] + u[1:-1, -1] - 4 * u[1:-1, -1]) / h**2
            r[1:-1, -1] = f[1:-1, -1] - (-lap)

        if self.bc_x == "neumann" and cols > 2:
            # left i=0
            lap = (u[1, 1:-1] + u[0, 0:-2] + u[0, 2:] + u[0, 1:-1] - 4 * u[0, 1:-1]) / h**2
            r[0, 1:-1] = f[0, 1:-1] - (-lap)
            # right i=rows-1
            lap = (u[-2, 1:-1] + u[-1, 0:-2] + u[-1, 2:] + u[-1, 1:-1] - 4 * u[-1, 1:-1]) / h**2
            r[-1, 1:-1] = f[-1, 1:-1] - (-lap)

        # Dirichlet boundaries are fixed; keep residual zero there
        return r

    def restrict(self, r):
        if USE_NUMBA:
            return _rect_restrict_numba(r, self._bc_x_neu, self._bc_y_neu)
        nr, nc = r.shape
        Nf_r = nr - 1
        Nf_c = nc - 1
        Nc_r = Nf_r // 2
        Nc_c = Nf_c // 2
        rc = np.zeros((Nc_r + 1, Nc_c + 1))

        # Vectorized full-weighting for interior coarse points
        if nr >= 5 and nc >= 5 and Nc_r >= 1 and Nc_c >= 1:
            c = r[2:-2:2, 2:-2:2]
            e = r[2:-2:2, 1:-3:2] + r[2:-2:2, 3:-1:2] + r[1:-3:2, 2:-2:2] + r[3:-1:2, 2:-2:2]
            k = r[1:-3:2, 1:-3:2] + r[1:-3:2, 3:-1:2] + r[3:-1:2, 1:-3:2] + r[3:-1:2, 3:-1:2]
            rc[1:-1, 1:-1] = 0.25 * c + 0.125 * e + 0.0625 * k

        # Preserve Neumann-boundary residuals using 1D full-weighting
        # along the boundary lines (prevents boundary error loss).
        if self.bc_y == "neumann" and Nc_r >= 1:
            # bottom (j=0)
            for i in range(1, Nc_r):
                i_f = 2 * i
                rc[i, 0] = 0.5 * r[i_f, 0] + 0.25 * (r[i_f - 1, 0] + r[i_f + 1, 0])
            # top (j=nc-1)
            for i in range(1, Nc_r):
                i_f = 2 * i
                rc[i, -1] = 0.5 * r[i_f, -1] + 0.25 * (r[i_f - 1, -1] + r[i_f + 1, -1])

        if self.bc_x == "neumann" and Nc_c >= 1:
            # left (i=0)
            for j in range(1, Nc_c):
                j_f = 2 * j
                rc[0, j] = 0.5 * r[0, j_f] + 0.25 * (r[0, j_f - 1] + r[0, j_f + 1])
            # right (i=nr-1)
            for j in range(1, Nc_c):
                j_f = 2 * j
                rc[-1, j] = 0.5 * r[-1, j_f] + 0.25 * (r[-1, j_f - 1] + r[-1, j_f + 1])

        # Inject corners (safe for Dirichlet: residual is already zero)
        if nr >= 1 and nc >= 1:
            rc[0, 0] = r[0, 0]
            rc[0, -1] = r[0, -1]
            rc[-1, 0] = r[-1, 0]
            rc[-1, -1] = r[-1, -1]
        return rc

    def prolongate_add(self, u, ec):
        if USE_NUMBA:
            _rect_prolongate_add_numba(u, ec)
            return
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
        # residual (mixed BCs)
        r = self.compute_residual(u, f, h)
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


class CoupledVelocityMGSolver:
    """MATLAB-style coupled V-cycle for (u, v) with coarse direct solve."""
    def __init__(self, N, nu1=2, nu2=2, L=None, bottom=None, coarse_direct=True):
        self.N = int(N)
        self.h = 1.0 / self.N
        self.nu1 = int(nu1)
        self.nu2 = int(nu2)
        if L is None:
            self.L = int(np.log2(self.N))
        else:
            self.L = int(L)
        if bottom is None:
            self.bottom = max(2, int(self.N // (2 ** max(0, self.L - 1))))
        else:
            self.bottom = int(bottom)
        self.coarse_direct = bool(coarse_direct)
        self._direct_cache = {}

        self.u_levels = []
        self.v_levels = []
        for lvl in range(self.L):
            N_lvl = self.N // (2 ** lvl)
            self.u_levels.append(
                RectangularPoissonMGSolver(
                    N_lvl + 1,
                    N_lvl,
                    nu1=self.nu1,
                    nu2=self.nu2,
                    L=1,
                    bc_x="dirichlet",
                    bc_y="neumann",
                )
            )
            self.v_levels.append(
                RectangularPoissonMGSolver(
                    N_lvl,
                    N_lvl + 1,
                    nu1=self.nu1,
                    nu2=self.nu2,
                    L=1,
                    bc_x="neumann",
                    bc_y="dirichlet",
                )
            )

    @staticmethod
    def _build_velocity_operators(N, h):
        h2 = h**2

        def laplace_1d_dirichlet(n_pts):
            if n_pts <= 0:
                return sp.csr_matrix((0, 0))
            main = 2 * np.ones(n_pts)
            off = -1 * np.ones(n_pts - 1)
            return sp.diags([off, main, off], [-1, 0, 1], shape=(n_pts, n_pts)) / h2

        def laplace_1d_neumann(n_pts):
            if n_pts <= 0:
                return sp.csr_matrix((0, 0))
            main = 2 * np.ones(n_pts)
            if n_pts > 1:
                main[0] = 1
                main[-1] = 1
            off = -1 * np.ones(n_pts - 1)
            return sp.diags([off, main, off], [-1, 0, 1], shape=(n_pts, n_pts)) / h2

        Dxx_u = laplace_1d_dirichlet(N - 1)
        Dyy_u = laplace_1d_neumann(N)
        if Dxx_u.shape[0] > 0 and Dyy_u.shape[0] > 0:
            A_u = sp.kron(Dxx_u, sp.eye(N)) + sp.kron(sp.eye(N - 1), Dyy_u)
        else:
            A_u = None

        Dxx_v = laplace_1d_neumann(N)
        Dyy_v = laplace_1d_dirichlet(N - 1)
        if Dxx_v.shape[0] > 0 and Dyy_v.shape[0] > 0:
            A_v = sp.kron(Dxx_v, sp.eye(N - 1)) + sp.kron(sp.eye(N), Dyy_v)
        else:
            A_v = None

        return A_u, A_v

    def _direct_solve_level(self, u, v, f_u, f_v, h):
        N = f_u.shape[1]
        if N <= 1:
            return

        solve_u = None
        solve_v = None
        cache = self._direct_cache.get(N)
        if cache is None:
            A_u, A_v = self._build_velocity_operators(N, h)
            if A_u is not None and A_u.shape[0] > 0:
                solve_u = spla.factorized(A_u.tocsc())
            if A_v is not None and A_v.shape[0] > 0:
                solve_v = spla.factorized(A_v.tocsc())
            self._direct_cache[N] = (solve_u, solve_v)
        else:
            solve_u, solve_v = cache

        if solve_u is not None:
            b_u = f_u[1:-1, :].reshape((N - 1) * N)
            u_int = solve_u(b_u)
            u[1:-1, :] = u_int.reshape((N - 1, N))
        if solve_v is not None:
            b_v = f_v[:, 1:-1].reshape(N * (N - 1))
            v_int = solve_v(b_v)
            v[:, 1:-1] = v_int.reshape((N, N - 1))

        u[0, :] = 0.0
        u[-1, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0

    def v_cycle(self, u, v, f_u, f_v, h, level=0):
        u_solver = self.u_levels[level]
        v_solver = self.v_levels[level]

        for _ in range(self.nu1):
            u_solver.relax(u, f_u, h)
            v_solver.relax(v, f_v, h)

        N = f_u.shape[1]
        if N <= self.bottom or level >= (self.L - 1):
            if self.coarse_direct:
                self._direct_solve_level(u, v, f_u, f_v, h)
            else:
                for _ in range(10):
                    u_solver.relax(u, f_u, h)
                    v_solver.relax(v, f_v, h)
            return

        r_u = u_solver.compute_residual(u, f_u, h)
        r_v = v_solver.compute_residual(v, f_v, h)
        rc_u = u_solver.restrict(r_u)
        rc_v = v_solver.restrict(r_v)

        ec_u = np.zeros_like(rc_u)
        ec_v = np.zeros_like(rc_v)
        self.v_cycle(ec_u, ec_v, rc_u, rc_v, 2 * h, level + 1)

        u_solver.prolongate_add(u, ec_u)
        v_solver.prolongate_add(v, ec_v)

        for _ in range(self.nu2):
            u_solver.relax(u, f_u, h)
            v_solver.relax(v, f_v, h)


class InexactUzawaSolver:
    def __init__(self, N, alpha=None, tau=1e-6, eps=1e-8, inner_iters=1, inner_maxiter=200, use_mg=False, nu1=2, nu2=2, L=None, outer_tol=1e-8, mg_direct=False, debug=False, precond_cycles=1, coupled_vcycle=True):
        """
        Inexact Uzawa Solver .
        N: Grid resolution.
        """
        self.N = N
        self.h = 1.0 / N
        # Default alpha for Uzawa-style pressure updates
        self.alpha = 1.0 if alpha is None else alpha
        # Number of inner solver iterations (CG or MG) to perform per outer step
        self.inner_iters = int(inner_iters)
        # Toggle: use MG inner solvers (True) or sparse matrix-based CG (False)
        self.use_mg = bool(use_mg)
        # Debug mode: print additional diagnostics
        self.debug = bool(debug)
        self.tau = float(tau)
        self.eps = float(eps)
        self.inner_maxiter = int(inner_maxiter)
        self.outer_tol = float(outer_tol)
        self.mg_direct = bool(mg_direct)
        self.nu1 = int(nu1)
        self.nu2 = int(nu2)
        self.L = None if L is None else int(L)
        self.precond_cycles = int(precond_cycles)
        self.coupled_vcycle = bool(coupled_vcycle)
        
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
        self.mg_u = RectangularPoissonMGSolver(
            N + 1,
            N,
            nu1=self.nu1,
            nu2=self.nu2,
            L=self.L,
            bc_x="dirichlet",
            bc_y="neumann",
        )
        self.mg_v = RectangularPoissonMGSolver(
            N,
            N + 1,
            nu1=self.nu1,
            nu2=self.nu2,
            L=self.L,
            bc_x="neumann",
            bc_y="dirichlet",
        )
        self.coupled_mg = None
        if self.use_mg and self.coupled_vcycle:
            levels = int(np.log2(self.N)) if self.L is None else self.L
            bottom = max(2, int(self.N // (2 ** max(0, levels - 1))))
            self.coupled_mg = CoupledVelocityMGSolver(
                self.N,
                nu1=self.nu1,
                nu2=self.nu2,
                L=levels,
                bottom=bottom,
                coarse_direct=True,
            )
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
        # Neumann boundary contributions for u in y-direction
        x_u = np.arange(1, self.N) * self.h
        u_y = 2 * np.pi * (1 - np.cos(2 * np.pi * x_u))
        self.f[1:-1, 0] -= u_y / self.h
        self.f[1:-1, -1] += u_y / self.h

        # v-grid physical coordinates (for g) - full N x (N+1)
        I_v, J_v = np.meshgrid(np.arange(self.N), np.arange(self.N+1), indexing='ij')
        X_v = (I_v + 0.5) * self.h
        Y_v = J_v * self.h
        self.g = 4 * np.pi**2 * (2 * np.cos(2*np.pi*Y_v) - 1) * np.sin(2*np.pi*X_v)
        # Neumann boundary contributions for v in x-direction
        y_v = np.arange(1, self.N) * self.h
        v_x = -2 * np.pi * (1 - np.cos(2 * np.pi * y_v))
        self.g[0, 1:-1] -= v_x / self.h
        self.g[-1, 1:-1] += v_x / self.h

    def build_operators(self):
        """
        Construct sparse Laplacian operators for u and v inner solves.
        A_u for u interior unknowns (N-1 x N), A_v for v interior unknowns (N x N-1).
        """
        h = self.h
        N = self.N
        h2 = h**2

        def laplace_1d_dirichlet(n_pts):
            if n_pts <= 0:
                return sp.csr_matrix((0, 0))
            main = 2 * np.ones(n_pts)
            off = -1 * np.ones(n_pts - 1)
            return sp.diags([off, main, off], [-1, 0, 1], shape=(n_pts, n_pts)) / h2

        def laplace_1d_neumann(n_pts):
            if n_pts <= 0:
                return sp.csr_matrix((0, 0))
            main = 2 * np.ones(n_pts)
            if n_pts > 1:
                main[0] = 1
                main[-1] = 1
            off = -1 * np.ones(n_pts - 1)
            return sp.diags([off, main, off], [-1, 0, 1], shape=(n_pts, n_pts)) / h2

        # A_u: (N-1) x N grid => kron(Dxx, I_N) + kron(I_(N-1), Dyy)
        Dxx_u = laplace_1d_dirichlet(N - 1)
        Dyy_u = laplace_1d_neumann(N)
        if Dxx_u.shape[0] > 0 and Dyy_u.shape[0] > 0:
            self.A_u = sp.kron(Dxx_u, sp.eye(N)) + sp.kron(sp.eye(N - 1), Dyy_u)
        else:
            self.A_u = None

        # A_v: N x (N-1) grid
        Dxx_v = laplace_1d_neumann(N)
        Dyy_v = laplace_1d_dirichlet(N - 1)
        if Dxx_v.shape[0] > 0 and Dyy_v.shape[0] > 0:
            self.A_v = sp.kron(Dxx_v, sp.eye(N - 1)) + sp.kron(sp.eye(N), Dyy_v)
        else:
            self.A_v = None

    def compute_residual_norm(self, p_override=None):
        """Compute full Stokes residual ||r_h||_2 / ||r_0||_2 logic."""
        h = self.h
        
        # 1. Momentum Residuals (Interior Points) using matrix operators
        p_arr = self.p if p_override is None else p_override

        u_interior = self.u[1:-1, :self.N]
        v_interior = self.v[:self.N, 1:-1]

        u_vec = u_interior.flatten()
        v_vec = v_interior.flatten()

        px = (p_arr[1:, :] - p_arr[:-1, :]) / h
        py = (p_arr[:, 1:] - p_arr[:, :-1]) / h

        r_u = self.f[1:-1, :].flatten() - (self.A_u.dot(u_vec) + px.flatten())
        r_v = self.g[:, 1:-1].flatten() - (self.A_v.dot(v_vec) + py.flatten())

        # 2. Divergence Residual (Constraint)
        div = (self.u[1:self.N+1, :self.N] - self.u[:self.N, :self.N])/h + (self.v[:self.N, 1:self.N+1] - self.v[:self.N, :self.N])/h

        total_sq = np.sum(r_u**2) + np.sum(r_v**2) + np.sum(div**2)
        return np.sqrt(total_sq)

    def _pcg_solve(self, A, b, x0, tol_abs, maxiter, precond=None):
        """Preconditioned CG using a callable preconditioner (report Algorithm 6)."""
        x = x0.copy()
        r = b - A.dot(x)
        if np.linalg.norm(r) <= tol_abs:
            return x, 0, True

        z = r.copy() if precond is None else precond(r)
        p = z.copy()
        rz_old = float(np.dot(r, z))
        if rz_old == 0.0:
            return x, 0, False

        it = 0
        while it < maxiter:
            Ap = A.dot(p)
            denom = float(np.dot(p, Ap))
            if denom == 0.0:
                break
            alpha = rz_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            it += 1
            if np.linalg.norm(r) <= tol_abs:
                return x, it, True

            z = r.copy() if precond is None else precond(r)
            rz_new = float(np.dot(r, z))
            if rz_old == 0.0:
                break
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        return x, it, False

    def _pcg_solve_uv(self, b_u, b_v, u0, v0, tol_abs, maxiter, precond=None):
        """PCG for block-diagonal velocity system using coupled V-cycle preconditioner."""
        u = u0.copy()
        v = v0.copy()
        r_u = b_u - self.A_u.dot(u)
        r_v = b_v - self.A_v.dot(v)
        r_norm = np.sqrt(np.dot(r_u, r_u) + np.dot(r_v, r_v))
        if r_norm <= tol_abs:
            return u, v, 0, True

        if precond is None:
            z_u, z_v = r_u.copy(), r_v.copy()
        else:
            z_u, z_v = precond(r_u, r_v)

        p_u = z_u.copy()
        p_v = z_v.copy()
        rz_old = float(np.dot(r_u, z_u) + np.dot(r_v, z_v))
        if rz_old == 0.0:
            return u, v, 0, False

        it = 0
        while it < maxiter:
            w_u = self.A_u.dot(p_u)
            w_v = self.A_v.dot(p_v)
            denom = float(np.dot(p_u, w_u) + np.dot(p_v, w_v))
            if denom == 0.0:
                break
            alpha = rz_old / denom
            u = u + alpha * p_u
            v = v + alpha * p_v
            r_u = r_u - alpha * w_u
            r_v = r_v - alpha * w_v
            it += 1
            r_norm = np.sqrt(np.dot(r_u, r_u) + np.dot(r_v, r_v))
            if r_norm <= tol_abs:
                return u, v, it, True

            if precond is None:
                z_u, z_v = r_u.copy(), r_v.copy()
            else:
                z_u, z_v = precond(r_u, r_v)
            rz_new = float(np.dot(r_u, z_u) + np.dot(r_v, z_v))
            if rz_old == 0.0:
                break
            beta = rz_new / rz_old
            p_u = z_u + beta * p_u
            p_v = z_v + beta * p_v
            rz_old = rz_new

        return u, v, it, False

    def solve(self, logger=None):
        if logger is not None:
            logger.info(f"--- Solving Inexact Uzawa (N={self.N}) ---")
        else:
            print(f"--- Solving Inexact Uzawa (N={self.N}) ---")
        t0 = time.time()
        
        # Initial residual (full Stokes residual)
        r0 = self.compute_residual_norm()
        if r0 < 1e-12:
            r0 = 1.0
        if logger is not None:
            logger.info(f"Init Residual: {r0:.4e}")
        
        tol = self.outer_tol
        # Report which inner solver is used
        if logger is not None:
            if self.use_mg and self.mg_direct:
                solver_name = f"Coupled-MG-Vcycle(x{max(1, self.precond_cycles)})" if self.coupled_mg is not None else f"MG-Vcycle(x{max(1, self.precond_cycles)})"
            elif self.use_mg:
                solver_name = f"PCG(Coupled-Vcycle x{max(1, self.precond_cycles)})" if self.coupled_mg is not None else f"PCG(Vcycle x{max(1, self.precond_cycles)})"
            else:
                solver_name = 'PCG(Diag)'
            logger.info(f"Inner solver: {solver_name}, inner_iters={self.inner_iters}, alpha={self.alpha:.6e}")
        rel_res = 1.0
        res_history = [rel_res]
        k = 0
        pcg_iters = []
        
        # Build MG preconditioner callbacks (used by custom PCG)
        pre_u = None
        pre_v = None
        pre_uv = None
        use_coupled_precond = False
        if self.use_mg and (not self.mg_direct) and self.A_u is not None and self.A_v is not None:
            precond_cycles = max(1, self.precond_cycles)
            u_ext = np.zeros((self.N + 1, self.N))
            b_ext_u = np.zeros_like(u_ext)
            v_ext = np.zeros((self.N, self.N + 1))
            b_ext_v = np.zeros_like(v_ext)

            if self.coupled_mg is not None:
                use_coupled_precond = True

                def pre_uv(r_u, r_v):
                    b_ext_u.fill(0.0)
                    b_ext_u[1:-1, :] = r_u.reshape((self.N - 1, self.N))
                    b_ext_v.fill(0.0)
                    b_ext_v[:, 1:-1] = r_v.reshape((self.N, self.N - 1))
                    u_ext.fill(0.0)
                    v_ext.fill(0.0)
                    for _ in range(precond_cycles):
                        self.coupled_mg.v_cycle(u_ext, v_ext, b_ext_u, b_ext_v, self.h)
                    return u_ext[1:-1, :].flatten(), v_ext[:, 1:-1].flatten()

            else:
                def pre_u(x):
                    b_ext_u.fill(0.0)
                    b_ext_u[1:-1, :] = x.reshape((self.N - 1, self.N))
                    u_ext.fill(0.0)
                    for _ in range(precond_cycles):
                        self.mg_u.v_cycle(u_ext, b_ext_u, self.h)
                    return u_ext[1:-1, :].flatten()

                def pre_v(x):
                    b_ext_v.fill(0.0)
                    b_ext_v[:, 1:-1] = x.reshape((self.N, self.N - 1))
                    v_ext.fill(0.0)
                    for _ in range(precond_cycles):
                        self.mg_v.v_cycle(v_ext, b_ext_v, self.h)
                    return v_ext[:, 1:-1].flatten()

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
            
            # 2. Approximate Velocity Solve (PCG with optional MG preconditioner)
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
            # Divergence norm for inexact stopping criterion
            div_norm = np.linalg.norm(
                (self.u[1:self.N+1, :self.N] - self.u[:self.N, :self.N]) / self.h
                + (self.v[:self.N, 1:self.N+1] - self.v[:self.N, :self.N]) / self.h
            )

            if self.use_mg and self.mg_direct:
                # Use MG V-cycle as a direct inner solver (no CG)
                for _ in range(max(1, self.inner_iters)):
                    if self.coupled_mg is not None:
                        self.coupled_mg.v_cycle(self.u, self.v, rhs_u, rhs_v, self.h)
                    else:
                        self.mg_u.v_cycle(self.u, rhs_u, self.h)
                        self.mg_v.v_cycle(self.v, rhs_v, self.h)
                pcg_iters.append(max(1, self.inner_iters))
            elif self.use_mg and self.A_u is not None and self.A_v is not None:
                pcg_step_max = 0
                for _ in range(max(1, self.inner_iters)):
                    u_interior = self.u[1:-1, :self.N]
                    v_interior = self.v[:self.N, 1:-1]
                    u_vec = u_interior.flatten()
                    v_vec = v_interior.flatten()
                    b_u = rhs_u[1:-1, :self.N].flatten()
                    b_v = rhs_v[:self.N, 1:-1].flatten()

                    if use_coupled_precond:
                        b_norm = np.sqrt(np.dot(b_u, b_u) + np.dot(b_v, b_v))
                        tol_abs = max(self.eps * b_norm, self.tau * div_norm)
                        sol_u, sol_v, it_uv, ok_uv = self._pcg_solve_uv(
                            b_u,
                            b_v,
                            u_vec,
                            v_vec,
                            tol_abs,
                            self.inner_maxiter,
                            precond=pre_uv,
                        )
                        if not ok_uv:
                            logging.warning("PCG(Coupled V-cycle) inner solve for A did not converge")
                        u_interior[:, :] = sol_u.reshape(u_interior.shape)
                        v_interior[:, :] = sol_v.reshape(v_interior.shape)
                        pcg_step_max = max(pcg_step_max, it_uv)
                    else:
                        # Solve for u (interior only) using PCG with V-cycle preconditioner
                        tol_abs_u = max(self.eps * np.linalg.norm(b_u), self.tau * div_norm)
                        sol_u, it_u, ok_u = self._pcg_solve(
                            self.A_u,
                            b_u,
                            u_vec,
                            tol_abs_u,
                            self.inner_maxiter,
                            precond=pre_u,
                        )
                        if not ok_u:
                            logging.warning("PCG(V-cycle) inner solve for A_u did not converge")
                        u_interior[:, :] = sol_u.reshape(u_interior.shape)

                        # Solve for v (interior only) using PCG with V-cycle preconditioner
                        tol_abs_v = max(self.eps * np.linalg.norm(b_v), self.tau * div_norm)
                        sol_v, it_v, ok_v = self._pcg_solve(
                            self.A_v,
                            b_v,
                            v_vec,
                            tol_abs_v,
                            self.inner_maxiter,
                            precond=pre_v,
                        )
                        if not ok_v:
                            logging.warning("PCG(V-cycle) inner solve for A_v did not converge")
                        v_interior[:, :] = sol_v.reshape(v_interior.shape)
                        pcg_step_max = max(pcg_step_max, it_u, it_v)
                pcg_iters.append(pcg_step_max)
            elif self.A_u is not None and self.A_v is not None:
                pcg_step_max = 0
                for _ in range(max(1, self.inner_iters)):
                    # Solve for u (interior only) using (P)CG
                    u_interior = self.u[1:-1, :self.N]
                    u_vec = u_interior.flatten()
                    b_u = rhs_u[1:-1, :self.N].flatten()
                    tol_abs_u = max(self.eps * np.linalg.norm(b_u), self.tau * div_norm)
                    it_u = 0
                    def cb_u(_):
                        nonlocal it_u
                        it_u += 1
                    sol_u, info_u = spla.cg(
                        self.A_u,
                        b_u,
                        x0=u_vec,
                        rtol=0.0,
                        atol=tol_abs_u,
                        maxiter=self.inner_maxiter,
                        M=self.Mu_diag,
                        callback=cb_u,
                    )
                    if info_u != 0:
                        logging.warning(f"PCG inner solve for A_u did not converge: info={info_u}")
                    u_interior[:, :] = sol_u.reshape(u_interior.shape)

                    # Solve for v (interior only) using (P)CG
                    v_interior = self.v[:self.N, 1:-1]
                    v_vec = v_interior.flatten()
                    b_v = rhs_v[:self.N, 1:-1].flatten()
                    tol_abs_v = max(self.eps * np.linalg.norm(b_v), self.tau * div_norm)
                    it_v = 0
                    def cb_v(_):
                        nonlocal it_v
                        it_v += 1
                    sol_v, info_v = spla.cg(
                        self.A_v,
                        b_v,
                        x0=v_vec,
                        rtol=0.0,
                        atol=tol_abs_v,
                        maxiter=self.inner_maxiter,
                        M=self.Mv_diag,
                        callback=cb_v,
                    )
                    if info_v != 0:
                        logging.warning(f"PCG inner solve for A_v did not converge: info={info_v}")
                    v_interior[:, :] = sol_v.reshape(v_interior.shape)
                    pcg_step_max = max(pcg_step_max, it_u, it_v)
                pcg_iters.append(pcg_step_max)
            else:
                # fallback to simple rectangular MG on u and v grids
                for _ in range(max(1, self.inner_iters)):
                    self.mg_u.v_cycle(self.u, rhs_u, self.h)
                    self.mg_v.v_cycle(self.v, rhs_v, self.h)

            # Enforce Dirichlet boundaries after inner solve
            self.u[0, :] = 0.0
            self.u[-1, :] = 0.0
            self.v[:, 0] = 0.0
            self.v[:, -1] = 0.0
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
            
            # 4. Check Convergence (full residual)
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
        
        return k, cpu_time, e_N, res_history, pcg_iters

if __name__ == "__main__":
    # Task 3: Parameter groups from the report
    max_n = int(os.environ.get("MAX_N", "2048"))
    N_list = [n for n in [64, 128, 256, 512, 1024, 2048] if n <= max_n]
    inner_maxiter = int(os.environ.get("INNER_MAXITER", "200"))
    outer_tol = float(os.environ.get("OUTER_TOL", "1e-8"))
    use_mg_env = os.environ.get("USE_MG")
    use_mg_default = True if use_mg_env is None else (use_mg_env != "0")
    mg_direct = os.environ.get("MG_DIRECT") == "1"
    precond_cycles = int(os.environ.get("PRECOND_VCYCLES", "1"))
    coupled_vcycle = os.environ.get("COUPLED_VCYCLE", "1") != "0"
    force_rerun = os.environ.get("FORCE_RERUN") == "1"

    def levels_for_coarse(N, N_coarse):
        return int(np.log2(N / N_coarse)) + 1

    configs = [
        {"name": "tau1e-3_eps1e-8_alpha1_nu2_L2", "tau": 1e-3, "eps": 1e-8, "alpha": 1.0, "nu1": 2, "nu2": 2, "N_coarse": 2},
        {"name": "tau1e-3_eps1e-8_alpha1_nu4_L4", "tau": 1e-3, "eps": 1e-8, "alpha": 1.0, "nu1": 4, "nu2": 4, "N_coarse": 4},
        {"name": "tau1e-3_eps1e-8_alpha1p05_nu2_L2", "tau": 1e-3, "eps": 1e-8, "alpha": 1.05, "nu1": 2, "nu2": 2, "N_coarse": 2},
        {"name": "tau1e-3_eps1e-8_alpha1p05_nu4_L4", "tau": 1e-3, "eps": 1e-8, "alpha": 1.05, "nu1": 4, "nu2": 4, "N_coarse": 4},
        {"name": "tau1e-3_eps1e-8_alpha0p95_nu2_L2", "tau": 1e-3, "eps": 1e-8, "alpha": 0.95, "nu1": 2, "nu2": 2, "N_coarse": 2},
        {"name": "tau1e-3_eps1e-8_alpha0p95_nu4_L4", "tau": 1e-3, "eps": 1e-8, "alpha": 0.95, "nu1": 4, "nu2": 4, "N_coarse": 4},
        {"name": "tau1e-4_eps1e-8_alpha1_nu4_L4", "tau": 1e-4, "eps": 1e-8, "alpha": 1.0, "nu1": 4, "nu2": 4, "N_coarse": 4},
        {"name": "tau1e-5_eps1e-8_alpha1_nu4_L4", "tau": 1e-5, "eps": 1e-8, "alpha": 1.0, "nu1": 4, "nu2": 4, "N_coarse": 4},
    ]

    only_cfg = os.environ.get("ONLY_CONFIG")
    if only_cfg:
        configs = [c for c in configs if c["name"] == only_cfg]

    results_base = os.path.join(os.path.dirname(__file__), "results")
    script_name = "inexact_uzawa"
    results_dir = os.path.join(results_base, script_name)
    os.makedirs(results_dir, exist_ok=True)

    for cfg in configs:
        cfg_dir = os.path.join(results_dir, cfg["name"])
        os.makedirs(cfg_dir, exist_ok=True)
        summary_file = os.path.join(cfg_dir, f"results_{cfg['name']}.csv")

        print(f"\n=== Inexact Uzawa: {cfg['name']} ===")
        print(f"{'N':<6} | {'Iters':<8} | {'CPU(s)':<10} | {'Error':<12}")
        print("-" * 42)

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
                writer.writerow(["N", "Iters", "CPU(s)", "Error", "alpha", "tau", "eps", "nu1", "nu2", "L", "InitialRelRes", "FinalRelRes", "ResidualFile", "PCG_Iters"])

            for N in N_list:
                if N in existing_ns:
                    continue
                L = levels_for_coarse(N, cfg["N_coarse"])
                solver = InexactUzawaSolver(
                    N,
                    alpha=cfg["alpha"],
                    tau=cfg["tau"],
                    eps=cfg["eps"],
                    inner_iters=1,
                    inner_maxiter=inner_maxiter,
                    use_mg=use_mg_default,
                    nu1=cfg["nu1"],
                    nu2=cfg["nu2"],
                    L=L,
                    outer_tol=outer_tol,
                    mg_direct=mg_direct,
                    precond_cycles=precond_cycles,
                    coupled_vcycle=coupled_vcycle,
                )
                run_dir = os.path.join(cfg_dir, f"N{N}")
                os.makedirs(run_dir, exist_ok=True)

                log_file = os.path.join(run_dir, f"inexact_uzawa_{cfg['name']}_N{N}.log")
                logger = logging.getLogger(f"inexact_uzawa_{cfg['name']}_{N}")
                for h in list(logger.handlers):
                    logger.removeHandler(h)
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler(log_file)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                logger.info(f"Run Inexact Uzawa: N={N}, h={solver.h}, cfg={cfg['name']}")

                k, t, e, res_hist, pcg_iters = solver.solve(logger=logger)

                residuals_file = os.path.join(run_dir, f"residuals_inexact_uzawa_{cfg['name']}_N{N}.csv")
                with open(residuals_file, "w", newline='') as rf:
                    w2 = csv.writer(rf)
                    w2.writerow(["iter", "rel_res"])
                    for idx, val in enumerate(res_hist):
                        w2.writerow([idx, val])
                        logger.info(f"Iter {idx}: Rel Res = {val:.6e}")

                summary_row = (
                    N,
                    k,
                    t,
                    e,
                    solver.alpha,
                    solver.tau,
                    solver.eps,
                    solver.nu1,
                    solver.nu2,
                    solver.L,
                    res_hist[0],
                    res_hist[-1],
                    os.path.basename(residuals_file),
                    ",".join(str(v) for v in pcg_iters),
                )
                writer.writerow(summary_row)
                sf.flush()
                print(f"{N:<6} | {k:<8} | {t:<10.2f} | {e:<12.4e}")
                logger.info(
                    f"Summary: N={N}, Iters={k}, CPU(s)={t:.6f}, Error={e:.6e}, ResidualFile={os.path.basename(residuals_file)}, PCG_Iters={','.join(str(v) for v in pcg_iters)}"
                )
                logger.removeHandler(fh)
                fh.close()

        print(f"Summary results saved to: {summary_file}")

    print(f"Per-run files saved in: {results_dir}")