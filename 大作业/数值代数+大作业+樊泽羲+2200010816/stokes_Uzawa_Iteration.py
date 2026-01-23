import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import os
import csv
import logging

class StokesUzawaSolver:
    def __init__(self, N):
        """
        Solver for Stokes Equations using Uzawa Iteration[cite: 92].
        Domain: Unit Square (0,1)^2.
        Grid: Staggered MAC Grid.
        """
        self.N = N
        self.h = 1.0 / N
        self.alpha = 1.0  # Relaxation parameter [cite: 104]
        
        # --- Grid Dimensions (Unknowns) ---
        # u: vertical edges (N-1) x N (Dirichlet boundaries removed)
        self.Nu = (N - 1) * N
        # v: horizontal edges N x (N-1) (Dirichlet boundaries removed)
        self.Nv = N * (N - 1)
        # p: cell centers N x N
        self.Np = N * N
        
        print(f"Initializing N={N} (Unknowns: u={self.Nu}, v={self.Nv}, p={self.Np})...")
        self.build_operators()
        self.build_rhs()

    def build_operators(self):
        """
        Construct sparse operators A, B, B.T using Kronecker products.
        A must be SPD for Conjugate Gradient to work.
        """
        h = self.h
        h2 = h**2
        N = self.N

        # 1. Laplacian Stencils (1D)
        # Dirichlet: u=0 at boundaries
        def laplace_1d_dirichlet(n_pts):
            main_diag = 2 * np.ones(n_pts)
            off_diag = -1 * np.ones(n_pts - 1)
            return sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n_pts, n_pts)) / h2

        # Neumann: du/dn prescribed at boundaries (uses modified boundary rows)
        def laplace_1d_neumann(n_pts):
            main_diag = 2 * np.ones(n_pts)
            if n_pts > 1:
                main_diag[0] = 1
                main_diag[-1] = 1
            off_diag = -1 * np.ones(n_pts - 1)
            return sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n_pts, n_pts)) / h2

        # 2. Gradient Stencil (1D): [-1, 1] / h
        # Maps n_in points to n_out edges (n_out = n_in - 1)
        def grad_1d(n_in):
            n_out = n_in - 1
            # Diagonals: main=-1, upper=1
            return sp.diags([-1*np.ones(n_out), np.ones(n_out)], [0, 1], shape=(n_out, n_in)) / h

        # --- Construct A (Momentum Operator) ---
        # A_u (size Nu x Nu)
        # -Lap(u) = -u_xx - u_yy
        # x-direction: size N-1 (internal edges). y-direction: size N (cells).
        Dxx_u = laplace_1d_dirichlet(N - 1)
        Dyy_u = laplace_1d_neumann(N)
        # Kron order: x is outer (slow), y is inner (fast) to match row-major flatten
        # A = Dxx (x) Iy + Ix (x) Dyy
        self.A_u = sp.kron(Dxx_u, sp.eye(N)) + sp.kron(sp.eye(N - 1), Dyy_u)

        # A_v (size Nv x Nv)
        # x-direction: size N. y-direction: size N-1.
        Dxx_v = laplace_1d_neumann(N)
        Dyy_v = laplace_1d_dirichlet(N - 1)
        self.A_v = sp.kron(Dxx_v, sp.eye(N - 1)) + sp.kron(sp.eye(N), Dyy_v)

        # --- Construct B (Gradient Operator) ---
        # Bx: p(N,N) -> u(N-1, N).  Diff in x.
        Gx = grad_1d(N)
        self.Bx = sp.kron(Gx, sp.eye(N))

        # By: p(N,N) -> v(N, N-1). Diff in y.
        Gy = grad_1d(N)
        self.By = sp.kron(sp.eye(N), Gy)

    def build_rhs(self):
        """Compute Force vectors F and G based on [cite: 133-134]."""
        # Grid Coordinates for u (Internal)
        # i=1..N-1, j=0..N-1
        I = np.arange(1, self.N)
        J = np.arange(0, self.N)
        II, JJ = np.meshgrid(I, J, indexing='ij')
        X_u = II * self.h
        Y_u = (JJ + 0.5) * self.h
        
        # F(x,y)
        f_vals = -4 * np.pi**2 * (2 * np.cos(2*np.pi*X_u) - 1) * np.sin(2*np.pi*Y_u) + X_u**2
        # Neumann boundary contributions for u in y-direction
        x_u = I * self.h
        u_y = 2 * np.pi * (1 - np.cos(2 * np.pi * x_u))
        f_vals[:, 0] -= u_y / self.h
        f_vals[:, -1] += u_y / self.h
        self.F = f_vals.flatten() # Row-major match for Kronecker

        # Grid Coordinates for v (Internal)
        # i=0..N-1, j=1..N-1
        I = np.arange(0, self.N)
        J = np.arange(1, self.N)
        II, JJ = np.meshgrid(I, J, indexing='ij')
        X_v = (II + 0.5) * self.h
        Y_v = JJ * self.h
        
        # G(x,y)
        g_vals = 4 * np.pi**2 * (2 * np.cos(2*np.pi*Y_v) - 1) * np.sin(2*np.pi*X_v)
        # Neumann boundary contributions for v in x-direction
        y_v = J * self.h
        v_x = -2 * np.pi * (1 - np.cos(2 * np.pi * y_v))
        g_vals[0, :] -= v_x / self.h
        g_vals[-1, :] += v_x / self.h
        self.G = g_vals.flatten()

    def compute_residual_norm(self, u_vec, v_vec, p_vec):
        """Compute full Stokes residual L2 norm for the current state.
        u_vec: (Nu,) flattened interior u unknowns -> reshape ((N-1), N)
        v_vec: (Nv,) flattened interior v unknowns -> reshape (N, (N-1))
        p_vec: (Np,) flattened p -> reshape (N, N)
        """
        h = self.h
        N = self.N

        # Reshape vectors to grid layout
        u_grid = u_vec.reshape((N-1, N))
        v_grid = v_vec.reshape((N, N-1))
        p_arr = p_vec.reshape((N, N))

        # Expand to full arrays with Dirichlet boundaries
        u_ext = np.zeros((N + 1, N))
        u_ext[1:-1, :] = u_grid

        v_ext = np.zeros((N, N + 1))
        v_ext[:, 1:-1] = v_grid

        # Residuals in matrix form: A u + B p = F, div = B^T u
        p_vec = p_arr.flatten()
        r_u = self.F - (self.A_u.dot(u_vec) + self.Bx.dot(p_vec))
        r_v = self.G - (self.A_v.dot(v_vec) + self.By.dot(p_vec))
        div = self.Bx.T.dot(u_vec) + self.By.T.dot(v_vec)

        total_sq = np.sum(r_u**2) + np.sum(r_v**2) + np.sum(div**2)
        return np.sqrt(total_sq)

    def get_exact_solution(self):
        """Exact solutions for error calc [cite: 136-137]."""
        # Exact u
        I = np.arange(1, self.N)
        J = np.arange(0, self.N)
        II, JJ = np.meshgrid(I, J, indexing='ij')
        X_u = II * self.h
        Y_u = (JJ + 0.5) * self.h
        u_exact = (1 - np.cos(2*np.pi*X_u)) * np.sin(2*np.pi*Y_u)
        
        # Exact v
        I = np.arange(0, self.N)
        J = np.arange(1, self.N)
        II, JJ = np.meshgrid(I, J, indexing='ij')
        X_v = (II + 0.5) * self.h
        Y_v = JJ * self.h
        v_exact = -(1 - np.cos(2*np.pi*Y_v)) * np.sin(2*np.pi*X_v)
        
        return u_exact.flatten(), v_exact.flatten()

    def solve(self, logger=None):
        """Main Uzawa Loop"""
        if logger is not None:
            logger.info(f"--- Solving N={self.N} ---")
        else:
            print(f"--- Solving N={self.N} ---")
        t0 = time.time()
        
        # Initial Guesses
        u = np.zeros(self.Nu)
        v = np.zeros(self.Nv)
        p = np.zeros(self.Np)
        
        # Initial residual will be set using the divergence after first velocity update
        r0 = None
        
        # Iteration Parameters
        tol = 1e-8
        max_iter = 1000
        rel_res = 1.0
        res_history = [rel_res]
        k = 0
        
        # [cite: 92-97] Uzawa Iteration
        while rel_res > tol and k < max_iter:
            # 1. Update Velocity: Solve A U = F - B P
            rhs_u = self.F - self.Bx.dot(p)
            rhs_v = self.G - self.By.dot(p)
            
            # CG Solve (Inner loop)
            # Must be tighter than outer loop to maintain convergence
            u, _ = spla.cg(self.A_u, rhs_u, x0=u, rtol=1e-10, atol=1e-12)
            v, _ = spla.cg(self.A_v, rhs_v, x0=v, rtol=1e-10, atol=1e-12)
            
            # 2. Update Pressure: P = P + alpha * Div(U)
            # Div(U) = B.T * U
            div = self.Bx.T.dot(u) + self.By.T.dot(v)
            if r0 is None:
                r0 = np.linalg.norm(div)
                if r0 < 1e-12:
                    r0 = 1.0
            p = p + self.alpha * div
            # Ensure zero-mean pressure (fix null-space)
            p -= np.mean(p)
            
            # 3. Check Convergence (Divergence residual)
            rel_res = np.linalg.norm(div) / r0
            res_history.append(rel_res)
            
            k += 1
            if k % 10 == 0:
                msg = f"Iter {k}: Rel Res = {rel_res:.4e}"
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)
                
        cpu_time = time.time() - t0
        
        # Error Calculation
        u_ex, v_ex = self.get_exact_solution()
        err_sq = np.sum((u - u_ex)**2) + np.sum((v - v_ex)**2)
        e_N = self.h * np.sqrt(err_sq)
        
        if logger is not None:
            logger.info(f"Finished: Iters={k}, CPU(s)={cpu_time:.6f}, Error={e_N:.6e}")
        return k, cpu_time, e_N, res_history

if __name__ == "__main__":
    # Task 2 Requirement: N = 64, 128, 256, 512
    # N=512 runs significantly slower due to system size.
    N_values = [64, 128, 256, 512] # Expand list for full assignment
    
    print(f"{'N':<6} | {'Iters':<8} | {'CPU Time(s)':<12} | {'Error e_N':<12}")
    print("-" * 45)
    
    # Create base results directory and script-specific folder
    results_base = os.path.join(os.path.dirname(__file__), "results")
    script_name = "uzawa"
    results_dir = os.path.join(results_base, script_name)
    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []
    summary_file = os.path.join(results_dir, "results_uzawa.csv")
    with open(summary_file, "w", newline='') as sf:
        writer = csv.writer(sf)
        writer.writerow(["N", "Iters", "CPU Time(s)", "Error e_N", "InitialRelRes", "FinalRelRes", "ResidualFile"])
        for N in N_values:
            solver = StokesUzawaSolver(N)
            # per-run subfolder
            run_dir = os.path.join(results_dir, f"N{N}")
            os.makedirs(run_dir, exist_ok=True)

            # set up per-run logger
            log_file = os.path.join(run_dir, f"uzawa_N{N}.log")
            logger = logging.getLogger(f"uzawa_{N}")
            for h in list(logger.handlers):
                logger.removeHandler(h)
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info(f"Run Uzawa: N={N}, h={solver.h}")

            it, t, err, res_hist = solver.solve(logger=logger)

            # Write residual history to CSV
            residuals_file = os.path.join(run_dir, f"residuals_uzawa_N{N}.csv")
            with open(residuals_file, "w", newline='') as rf:
                w2 = csv.writer(rf)
                w2.writerow(["iter", "rel_res"])
                for idx, val in enumerate(res_hist):
                    w2.writerow([idx, val])
                    logger.info(f"Iter {idx}: Rel Res = {val:.6e}")

            summary_row = (N, it, t, err, res_hist[0], res_hist[-1], os.path.basename(residuals_file))
            writer.writerow(summary_row)
            sf.flush()
            summary_rows.append(summary_row)
            print(f"{N:<6} | {it:<8} | {t:<12.4f} | {err:<12.4e}")
            logger.info(f"Summary: N={N}, Iters={it}, CPU(s)={t:.6f}, Error={err:.6e}, ResidualFile={os.path.basename(residuals_file)}")
            logger.removeHandler(fh)
            fh.close()

    # File already written incrementally inside 'with' context
    print(f"Summary results saved to: {summary_file}")
    print(f"Per-run files saved in: {results_dir}/*/N<value>")