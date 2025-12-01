import numpy as np
import time
import os
import csv
import logging

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
        self.f[0, :] = 0
        self.f[-1, :] = 0

        # --- Initialize g on v-grid ---
        I_v, J_v = np.meshgrid(np.arange(self.N), np.arange(self.N + 1), indexing='ij')
        X_v = (I_v + 0.5) * self.h
        Y_v = J_v * self.h
        
        self.g = 4 * np.pi**2 * (2 * np.cos(2*np.pi*Y_v) - 1) * np.sin(2*np.pi*X_v)
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

    def dgs_smoother(self, u, v, p, f, g, h):
        """DGS Smoother: Momentum relaxation + Divergence correction"""
        h2 = h * h
        N = p.shape[0]
        
        # Adaptive omega based on grid size for stability
        omega = 0.5 if N >= 512 else 0.5 + 0.2 * (512 - N) / 512
        
        # Momentum relaxation for u with adaptive omega
        px = np.zeros((N+1, N))
        px[1:-1, :] = (p[1:, :] - p[:-1, :]) / h
        
        u_pad_y = np.pad(u, ((0,0), (1,1)), 'constant')
        u_new = u.copy()
        u_new[1:-1, :] = (1-omega)*u[1:-1, :] + omega*0.25 * (
            u[0:-2, :] + u[2:, :] + u_pad_y[1:-1, 0:-2] + u_pad_y[1:-1, 2:] +
            h2 * (f[1:-1, :] - px[1:-1, :])
        )
        u[:] = u_new
        
        # Momentum relaxation for v
        py = np.zeros((N, N+1))
        py[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / h
        
        v_pad_x = np.pad(v, ((1,1), (0,0)), 'constant')
        v_new = v.copy()
        v_new[:, 1:-1] = (1-omega)*v[:, 1:-1] + omega*0.25 * (
            v_pad_x[0:-2, 1:-1] + v_pad_x[2:, 1:-1] + v[:, 0:-2] + v[:, 2:] +
            h2 * (g[:, 1:-1] - py[:, 1:-1])
        )
        v[:] = v_new

        # Divergence correction with adaptive damping
        div = (u[1:, :] - u[:-1, :])/h + (v[:, 1:] - v[:, :-1])/h
        r = -div
        
        # More conservative for larger grids
        alpha = 0.3 if N >= 512 else 0.5
        delta = alpha * r * h / 4.0
        
        u[:-1, :] -= delta
        u[1:, :] += delta
        v[:, :-1] -= delta
        v[:, 1:] += delta
        
        # Simplified pressure update and enforce zero-mean for pressure (to fix null-space)
        p += alpha * r
        # Ensure zero-mean pressure to keep pressure uniquely defined
        p -= np.mean(p)

    def compute_residual_norm(self, u, v, p, f, g, h):
        """Calculate L2 Norm of Residuals"""
        # Momentum u residual
        px = (p[1:, :] - p[:-1, :]) / h
        u_pad = np.pad(u, ((0,0),(1,1)), 'constant')
        lap_u = (u[0:-2,:] + u[2:,:] + u_pad[1:-1,0:-2] + u_pad[1:-1,2:] - 4*u[1:-1,:]) / h**2
        res_u = f[1:-1, :] - (-lap_u + px)
        
        # Momentum v residual
        py = (p[:, 1:] - p[:, :-1]) / h
        v_pad = np.pad(v, ((1,1),(0,0)), 'constant')
        lap_v = (v_pad[0:-2,1:-1] + v_pad[2:,1:-1] + v[:,0:-2] + v[:,2:] - 4*v[:,1:-1]) / h**2
        res_v = g[:, 1:-1] - (-lap_v + py)
        
        # Continuity residual
        div = (u[1:, :] - u[:-1, :])/h + (v[:, 1:] - v[:, :-1])/h
        
        return np.sqrt(np.sum(res_u**2) + np.sum(res_v**2) + np.sum(div**2))

    def restrict(self, r_u, r_v, r_p):
        """Restriction Operator (Fine -> Coarse)"""
        N = r_p.shape[0]
        Nc = N // 2
        
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
        """Prolongation Operator (Coarse -> Fine) - FIXED VERSION"""
        Nc = dp_c.shape[0]
        Nf = Nc * 2
        
        # Pressure: Nearest neighbor
        dp = np.kron(dp_c, np.ones((2,2)))
        
        # Velocity U: Bilinear interpolation
        du = np.zeros((Nf + 1, Nf))
        for i in range(Nc + 1):
            for j in range(Nc):
                i_f, j_f = 2*i, 2*j
                if i_f < Nf + 1:
                    # Copy to even y-indices
                    du[i_f, j_f] = du_c[i, j]
                    if j_f + 1 < Nf:
                        du[i_f, j_f+1] = du_c[i, j]
        
        # Interpolate odd x-indices
        for i in range(Nc):
            for j in range(Nc):
                i_f, j_f = 2*i+1, 2*j
                if i_f < Nf + 1:
                    val = 0.5 * (du_c[i, j] + du_c[i+1, j])
                    du[i_f, j_f] = val
                    if j_f + 1 < Nf:
                        du[i_f, j_f+1] = val
        
        # Velocity V: Bilinear interpolation
        dv = np.zeros((Nf, Nf + 1))
        for i in range(Nc):
            for j in range(Nc + 1):
                i_f, j_f = 2*i, 2*j
                if j_f < Nf + 1:
                    # Copy to even x-indices
                    dv[i_f, j_f] = dv_c[i, j]
                    if i_f + 1 < Nf:
                        dv[i_f+1, j_f] = dv_c[i, j]
        
        # Interpolate odd y-indices
        for i in range(Nc):
            for j in range(Nc):
                i_f, j_f = 2*i, 2*j+1
                if j_f < Nf + 1:
                    val = 0.5 * (dv_c[i, j] + dv_c[i, j+1])
                    dv[i_f, j_f] = val
                    if i_f + 1 < Nf:
                        dv[i_f+1, j_f] = val
        
        return du, dv, dp

    def v_cycle(self, u, v, p, f, g, h, level=0):
        """V-Cycle"""
        N = p.shape[0]
        
        # Adaptive damping for coarse grid correction
        beta = 0.6 if N >= 512 else 0.8
        
        # Pre-smoothing
        for _ in range(self.nu1):
            self.dgs_smoother(u, v, p, f, g, h)
            
        # Base case
        # Base case: use coarse grid when the number of cells is small or when levels are exhausted
        if N <= 4 or level >= (self.L - 1):
            for _ in range(20):
                self.dgs_smoother(u, v, p, f, g, h)
            return

        # Compute residuals
        px = (p[1:, :] - p[:-1, :]) / h
        u_pad = np.pad(u, ((0,0),(1,1)), 'constant')
        lap_u = (u[0:-2,:] + u[2:,:] + u_pad[1:-1,0:-2] + u_pad[1:-1,2:] - 4*u[1:-1,:]) / h**2
        r_u = np.zeros_like(u); r_u[1:-1, :] = f[1:-1, :] - (-lap_u + px)
        
        py = (p[:, 1:] - p[:, :-1]) / h
        v_pad = np.pad(v, ((1,1),(0,0)), 'constant')
        lap_v = (v_pad[0:-2,1:-1] + v_pad[2:,1:-1] + v[:,0:-2] + v[:,2:] - 4*v[:,1:-1]) / h**2
        r_v = np.zeros_like(v); r_v[:, 1:-1] = g[:, 1:-1] - (-lap_v + py)
        
        div = (u[1:, :] - u[:-1, :])/h + (v[:, 1:] - v[:, :-1])/h
        r_p = -div

        # Restrict
        rc_u, rc_v, rc_p = self.restrict(r_u, r_v, r_p)
        
        # Solve coarse - damped correction
        ec_u = np.zeros_like(rc_u)
        ec_v = np.zeros_like(rc_v)
        ec_p = np.zeros_like(rc_p)
        self.v_cycle(ec_u, ec_v, ec_p, rc_u, rc_v, 2*h, level+1)
        
        # Prolongate and correct with adaptive damping
        du, dv, dp = self.prolongate(ec_u, ec_v, ec_p)
        u += beta * du
        v += beta * dv
        p += beta * dp
        
        # Post-smoothing
        for _ in range(self.nu2):
            self.dgs_smoother(u, v, p, f, g, h)

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
        res_history = [r0]
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
    N_list = [64, 128, 256, 512, 1024, 2048]
    
    print(f"\n{'N':<6} | {'V-Cycles':<8} | {'CPU Time(s)':<12} | {'Error e_N':<12}")
    print("-" * 46)
    
    results = []
    # Create base results directory and script-specific folder
    results_base = os.path.join(os.path.dirname(__file__), "results")
    script_name = "v_cycle"
    results_dir = os.path.join(results_base, script_name)
    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []
    summary_file = os.path.join(results_dir, "results_v_cycle.csv")
    with open(summary_file, "w", newline='') as sf:
        writer = csv.writer(sf)
        writer.writerow(["N", "V-Cycles", "CPU Time(s)", "Error e_N", "nu1", "nu2", "L", "InitialRelRes", "FinalRelRes", "ResidualFile"])
        for N in N_list:
            solver = StokesMultigrid(N)
            # per-run subfolder
            run_dir = os.path.join(results_dir, f"N{N}")
            os.makedirs(run_dir, exist_ok=True)
            # set up per-run logger
            log_file = os.path.join(run_dir, f"v_cycle_N{N}.log")
            logger = logging.getLogger(f"v_cycle_{N}")
            for h in list(logger.handlers):
                logger.removeHandler(h)
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info(f"Run V-cycle: N={N}, h={solver.h}")
            it, t, err, res_hist = solver.solve(logger=logger)
            results.append((N, it, t, err))
            # Write residual history to CSV
            residuals_file = os.path.join(run_dir, f"residuals_v_cycle_N{N}.csv")
            with open(residuals_file, "w", newline='') as rf:
                w2 = csv.writer(rf)
                w2.writerow(["iter", "rel_res"])
                for idx, val in enumerate(res_hist):
                    w2.writerow([idx, val])

            summary_row = (N, it, t, err, solver.nu1, solver.nu2, solver.L, res_hist[0], res_hist[-1], os.path.basename(residuals_file))
            writer.writerow(summary_row)
            sf.flush()
            summary_rows.append(summary_row)
            print(f"{N:<6} | {it:<8} | {t:<12.4f} | {err:<12.4e}")
            logger.info(f"Summary: N={N}, V-Cycles={it}, CPU(s)={t:.6f}, Error={err:.6e}, ResidualFile={os.path.basename(residuals_file)}")
            logger.removeHandler(fh)
            fh.close()

    # File already written incrementally inside 'with' context
    print(f"Summary results saved to: {summary_file}")
    print(f"Per-run files saved in: {results_dir}/*/N<value>")
    
    print("\n")
