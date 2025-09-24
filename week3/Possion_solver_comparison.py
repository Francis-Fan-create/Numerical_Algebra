import numpy as np
import scipy.linalg
import time

def assemble_full_matrix(N):
    """
    Assembles the full (N-1)^2 x (N-1)^2 block tridiagonal matrix A.
    This is memory-intensive and not feasible for large N.

    Args:
        N (int): The number of divisions along one axis.

    Returns:
        np.ndarray: The full coefficient matrix A.
    """
    m = N - 1
    M = m * m

    # Create the diagonal block X
    X = np.zeros((m, m))
    for i in range(m):
        X[i, i] = 4.0
        if i > 0:
            X[i, i - 1] = -1.0
        if i < m - 1:
            X[i, i + 1] = -1.0

    # Create the full matrix A
    A = np.zeros((M, M))
    for i in range(m):
        start_row, end_row = i * m, (i + 1) * m
        # Place X on the main diagonal of blocks
        A[start_row:end_row, start_row:end_row] = X
        # Place -I on the off-diagonals of blocks
        if i > 0:
            start_prev = (i - 1) * m
            A[start_row:end_row, start_prev:start_row] = -np.identity(m)
        if i < m - 1:
            start_next = (i + 1) * m
            A[start_row:end_row, start_next:start_next + m] = -np.identity(m)
            
    return A

def assemble_banded_cholesky(N):
    """
    Assembles the matrix A in the compressed format required for SciPy's
    banded Cholesky solver (cholesky_banded).

    The matrix is symmetric positive-definite. We only store the main diagonal
    and the super-diagonals. The half-bandwidth is k = N-1.

    Args:
        N (int): The number of divisions along one axis.

    Returns:
        np.ndarray: The banded matrix in SciPy's format.
    """
    m = N - 1
    M = m * m
    k = m  # Half-bandwidth
    
    # Shape is (k + 1, M) for the upper triangle
    ab_chol = np.zeros((k + 1, M))

    # Main diagonal (offset 0) -> goes into the last row k
    ab_chol[k, :] = 4.0

    # Super-diagonal (offset 1) -> goes into row k-1
    for i in range(M - 1):
        if (i + 1) % m != 0:  # Avoids linking block rows
            ab_chol[k - 1, i + 1] = -1.0

    # Super-diagonal (offset m) -> goes into row k-m = 0
    for i in range(M - m):
        ab_chol[k - m, i + m] = -1.0

    return ab_chol

def setup_rhs_and_exact(N):
    """
    Sets up the right-hand side vector F_h and the exact solution grid u.
    
    Args:
        N (int): The number of divisions along one axis.

    Returns:
        tuple: (F_h vector, exact solution on (N+1)x(N+1) grid)
    """
    m = N - 1
    h = 1.0 / N
    
    # Create grid points for interior nodes (from x_1 to x_{N-1})
    x = np.linspace(h, 1.0 - h, m)
    y = np.linspace(h, 1.0 - h, m)
    xv, yv = np.meshgrid(x, y)

    # Source function f = 2*pi^2*sin(pi*x)*sin(pi*y)
    f_grid = 2 * (np.pi**2) * np.sin(np.pi * xv) * np.sin(np.pi * yv)
    
    # Flatten the grid into the F_h vector (column-major order)
    F_h = f_grid.flatten('F')

    # Create grid points for the full (N+1)x(N+1) domain (from 0 to 1)
    x_full = np.linspace(0, 1, N + 1)
    y_full = np.linspace(0, 1, N + 1)
    xv_full, yv_full = np.meshgrid(x_full, y_full)
    
    # Exact solution u = sin(pi*x)*sin(pi*y) on the full grid
    u_exact_grid = np.sin(np.pi * xv_full) * np.sin(np.pi * yv_full)
    
    return F_h, u_exact_grid

def calculate_error(U_h, u_exact_full_grid, N):
    """
    Calculates the discrete error norm e_N.

    Args:
        U_h (np.ndarray): The numerical solution vector for interior nodes.
        u_exact_full_grid (np.ndarray): The exact solution on the full grid.
        N (int): The number of divisions.

    Returns:
        float: The calculated error e_N.
    """
    m = N - 1
    h = 1.0 / N
    
    # Reshape numerical solution vector to an (m x m) grid
    U_interior_grid = U_h.reshape((m, m), order='F')
    
    # Create a full (N+1)x(N+1) grid for the numerical solution, with 0s on boundary
    U_full_grid = np.zeros((N + 1, N + 1))
    U_full_grid[1:N, 1:N] = U_interior_grid
    
    # Calculate the error e_N
    diff = u_exact_full_grid - U_full_grid
    error = h * np.sqrt(np.sum(diff**2))
    
    return error

def run_poisson_analysis():
    """
    Main function to run the comparison for different N values.
    """
    N_values = [16, 32, 64, 128, 256]
    
    print("Comparison of Solvers for the 2D Poisson Equation")
    print("-" * 100)
    print(f"{'N':>4} | {'System Size':>12} | {'Method':<20} | {'Time (s)':>15} | {'Error (e_N)':>20}")
    print("-" * 100)

    for N in N_values:
        m = N - 1
        M = m * m
        h = 1.0 / N

        Fh, u_exact_grid = setup_rhs_and_exact(N)
        b = (h**2) * Fh

        # --- Method 1: Gaussian Elimination (General Dense Solver) ---
        # Note: We use np.linalg.solve as a fast, standard implementation.
        # This method is skipped for large N due to excessive memory requirements.
        if M > 20000: # Heuristic limit for N>=128 to avoid MemoryError
            time_gauss = float('inf')
            error_gauss = float('nan')
        else:
            A_full = assemble_full_matrix(N)
            start_time = time.perf_counter()
            U_gauss = np.linalg.solve(A_full, b.copy())
            time_gauss = time.perf_counter() - start_time
            error_gauss = calculate_error(U_gauss, u_exact_grid, N)
            del A_full
        
        print(f"{N:>4} | {M:>12} | {'Gaussian Elim.':<20} | {time_gauss:>15.6f} | {error_gauss:>20.6e}")

        # --- Method 2: LDL^T / Cholesky (Dense Symmetric Solver) ---
        # This is also skipped for large N.
        if M > 20000:
            time_ldlt = float('inf')
            error_ldlt = float('nan')
        else:
            A_full = assemble_full_matrix(N)
            start_time = time.perf_counter()
            # CORRECTED: Use `assume_a='pos'` for symmetric positive-definite matrices.
            U_ldlt = scipy.linalg.solve(A_full, b.copy(), assume_a='pos')
            time_ldlt = time.perf_counter() - start_time
            error_ldlt = calculate_error(U_ldlt, u_exact_grid, N)
            del A_full
        
        print(f"{N:>4} | {M:>12} | {'Cholesky (LDL^T)':<20} | {time_ldlt:>15.6f} | {error_ldlt:>20.6e}")

        # --- Method 3: Banded LDL^T / Cholesky (Banded Symmetric Solver) ---
        # This is the most efficient method as it exploits the matrix structure.
        A_banded_chol = assemble_banded_cholesky(N)
        start_time = time.perf_counter()
        # Factorize the banded matrix
        c_factor = scipy.linalg.cholesky_banded(A_banded_chol, lower=False)
        # Solve using the factorization
        U_banded = scipy.linalg.cho_solve_banded((c_factor, False), b.copy())
        time_banded = time.perf_counter() - start_time
        error_banded = calculate_error(U_banded, u_exact_grid, N)
        
        print(f"{N:>4} | {M:>12} | {'Banded Cholesky':<20} | {time_banded:>15.6f} | {error_banded:>20.6e}")
        print("-" * 100)

if __name__ == '__main__':
    run_poisson_analysis()

