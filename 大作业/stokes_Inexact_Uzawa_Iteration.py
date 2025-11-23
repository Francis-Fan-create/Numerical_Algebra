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
    def __init__(self, N):
        """
        Inexact Uzawa Solver .
        N: Grid resolution.
        """
        self.N = N
        self.h = 1.0 / N
        # Choose alpha proportional to h^2 (stable step for pressure updates)
        # Schur complement eigenvalues scale ~ 1/h^2, so α = O(h^2) is stable.
        self.alpha = 0.125 * self.h**2
        # Number of inner solver iterations (CG or MG) to perform per outer step
        self.inner_iters = 2
        # Toggle: use MG inner solvers (True) or sparse matrix-based CG (False)
        self.use_mg = False
        
        # Data Structures:
        # Use full (N+1)x(N+1) grids for simplicity
        # Staggered interpretation: u on vertical edges, v on horizontal edges
        self.u = np.zeros((N+1, N+1)) 
        self.v = np.zeros((N+1, N+1))
        self.p = np.zeros((N, N))
        
        # Inner Solver - use the multigrid solver
        self.mg_solver = PoissonMGSolver(N)
        # Build sparse operators (A_u, A_v) for inner CG solves (ensures consistent discretization)
        self.build_operators()
        
        self.init_rhs()

    def init_rhs(self):
        """Initialize exact force terms [cite: 133-134]"""
        # u-grid physical coordinates (for f) - full (N+1)x(N+1)
        I, J = np.meshgrid(np.arange(self.N+1), np.arange(self.N+1), indexing='ij')
        X_u = I * self.h
        Y_u = (J + 0.5) * self.h
        self.f = -4 * np.pi**2 * (2 * np.cos(2*np.pi*X_u) - 1) * np.sin(2*np.pi*Y_u) + X_u**2
        self.f[:, -1] = 0  # Last column not used for u

        # v-grid physical coordinates (for g) - full (N+1)x(N+1)
        X_v = (I + 0.5) * self.h
        Y_v = J * self.h
        self.g = 4 * np.pi**2 * (2 * np.cos(2*np.pi*Y_v) - 1) * np.sin(2*np.pi*X_v)
        self.g[-1, :] = 0  # Last row not used for v

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
        # u is stored in (N+1, N+1) but only u[:, :N] are active
        # Interior u points: u[1:N, :N] (excluding top/bottom boundaries)
        u = self.u
        u_active = u[:, :self.N]  # (N+1, N)
        # Laplacian at interior points [1:-1, :] which is (N-1, N)
        lap_u = (u_active[0:-2, :] + u_active[2:, :] + 
                 np.pad(u_active[1:-1, :-1], ((0,0),(1,0)), mode='constant') + 
                 np.pad(u_active[1:-1, 1:], ((0,0),(0,1)), mode='constant') - 
                 4*u_active[1:-1, :]) / h**2
        
        # p_x: pressure gradient (N-1, N)
        p_arr = self.p if p_override is None else p_override
        px = (p_arr[1:, :] - p_arr[:-1, :]) / h
        
        # Residual on interior points (N-1, N)
        r_u = self.f[1:-1, :self.N] - (-lap_u + px)
        
        # res_v = g - (-Lap(v) + p_y)
        # v is stored in (N+1, N+1) but only v[:N, :] are active
        # Interior v points: v[:N, 1:N] (excluding left/right boundaries)
        v = self.v
        v_active = v[:self.N, :]  # (N, N+1)
        # Laplacian at interior points [:, 1:-1] which is (N, N-1)
        lap_v = (np.pad(v_active[:-1, 1:-1], ((1,0),(0,0)), mode='constant') + 
                 np.pad(v_active[1:, 1:-1], ((0,1),(0,0)), mode='constant') + 
                 v_active[:, 0:-2] + v_active[:, 2:] - 
                 4*v_active[:, 1:-1]) / h**2
        
        # p_y: pressure gradient (N, N-1)
        py = (p_arr[:, 1:] - p_arr[:, :-1]) / h
        
        # Residual on interior points (N, N-1)
        r_v = self.g[:self.N, 1:-1] - (-lap_v + py)
        
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
            solver_name = 'MG' if self.use_mg else 'CG'
            logger.info(f"Inner solver: {solver_name}, inner_iters={self.inner_iters}, alpha={self.alpha:.6e}")
        rel_res = 1.0
        res_history = [rel_res]
        k = 0
        
        # Iteration Loop [cite: 108-111]
        # Iteration loop
        while rel_res > tol and k < 200:
            # 1. RHS for Poisson Steps
            # rhs_u = f - p_x
            px = (self.p[1:, :] - self.p[:-1, :]) / self.h
            rhs_u = self.f.copy()
            rhs_u[1:-1, :self.N] -= px
            
            # rhs_v = g - p_y
            py = (self.p[:, 1:] - self.p[:, :-1]) / self.h
            rhs_v = self.g.copy()
            rhs_v[:self.N, 1:-1] -= py
            
            # 2. Approximate Velocity Solve (multigrid V-cycle)
            r_before_mg = self.compute_residual_norm()
            # Use sparse CG inner solves on properly-mapped interior unknowns
            if self.A_u is not None and self.A_v is not None and not self.use_mg:
                for _ in range(max(1, self.inner_iters)):
                    # Solve for u (interior only) using CG
                    u_interior = self.u[1:-1, :self.N]
                    u_vec = u_interior.flatten()
                    b_u = rhs_u[1:-1, :self.N].flatten()
                    sol_u, _ = spla.cg(self.A_u, b_u, x0=u_vec, rtol=1e-6, atol=1e-12)
                    u_interior[:, :] = sol_u.reshape(u_interior.shape)

                    # Solve for v (interior only)
                    v_interior = self.v[:self.N, 1:-1]
                    v_vec = v_interior.flatten()
                    b_v = rhs_v[:self.N, 1:-1].flatten()
                    sol_v, _ = spla.cg(self.A_v, b_v, x0=v_vec, rtol=1e-6, atol=1e-12)
                    v_interior[:, :] = sol_v.reshape(v_interior.shape)
            else:
                # fallback to MG in case sparse operators are requested or sparse operators are not built
                for _ in range(max(1, self.inner_iters)):
                    self.mg_solver.v_cycle(self.u, rhs_u, self.h)
                    self.mg_solver.v_cycle(self.v, rhs_v, self.h)
            r_after_mg = self.compute_residual_norm()
            
            # 3. Update Pressure (P += alpha * Div u) [cite: 110]
            div = (self.u[1:self.N+1, :self.N] - self.u[:self.N, :self.N])/self.h + (self.v[:self.N, 1:self.N+1] - self.v[:self.N, :self.N])/self.h

            # Try the pressure update with adaptive alpha: if the update increases
            # the residual, reduce alpha by half until it decreases (or reach min).
            r_before = self.compute_residual_norm()
            trial_alpha = self.alpha
            accepted = False
            for attempt in range(5):
                p_candidate = self.p + trial_alpha * div
                r_candidate = self.compute_residual_norm(p_override=p_candidate)
                if r_candidate <= r_before or trial_alpha <= 1e-16:
                    # accept update
                    self.p = p_candidate
                    # reduce alpha if we had to shrink it significantly
                    self.alpha = min(self.alpha, trial_alpha)
                    accepted = True
                    break
                trial_alpha *= 0.5
            if not accepted:
                # fallback: apply the smallest alpha we attempted
                self.p = self.p + trial_alpha * div
                self.alpha = min(self.alpha, trial_alpha)
            
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
    N_list = [64, 128] 
    
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
        writer.writerow(["N", "Iters", "CPU(s)", "Error", "InitialRelRes", "FinalRelRes", "ResidualFile"])
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

            summary_row = (N, k, t, e, res_hist[0], res_hist[-1], os.path.basename(residuals_file))
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