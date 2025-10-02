import numpy as np
import scipy.linalg

def estimate_norm_1(B, max_iter=10):
    """
    Estimates the 1-norm of a matrix B using the optimization-based power method.
    This corresponds to Algorithm 2.5.1 from the prompt.

    Args:
        B (np.ndarray): The input matrix.
        max_iter (int): The maximum number of iterations to prevent infinite loops.

    Returns:
        float: The estimated 1-norm of B.
    """
    n = B.shape[0]
    if n == 0:
        return 0.0

    # Start with x = [1/n, 1/n, ..., 1/n]^T, which satisfies ||x||_1 = 1
    x = np.ones(n) / n
    
    # The estimate of the norm is gamma
    gamma = 0.0

    for _ in range(max_iter):
        w = B @ x
        gamma = np.linalg.norm(w, 1) # This is a direct estimate: ||B*x||_1
        
        # In the algorithm's notation: gamma can also be seen as z.T @ x at convergence
        # v = sign(w)
        # Using a definition where sign(0) is 1, as is common in this context.
        v = np.ones_like(w)
        v[w < 0] = -1.0
        
        z = B.T @ v
        
        z_inf_norm = np.linalg.norm(z, np.inf)
        z_dot_x = z.T @ x

        # Stopping condition
        if z_inf_norm <= z_dot_x:
            # The loop terminates. The current estimate is the best one.
            # z_dot_x converges to ||B||_1, and equals ||w||_1 in this state.
            return z_dot_x

        # Update x for the next iteration
        j = np.argmax(np.abs(z))
        x = np.zeros(n)
        x[j] = 1.0
        
    # Return the last computed estimate if convergence is not reached
    return gamma

def solve_hilbert_problem():
    """
    Estimates the infinity-norm condition number for Hilbert matrices
    of orders 5 to 20.
    """
    print("--- Problem 2.2: Hilbert Matrix Condition Numbers ---")
    print(f"{'Order (n)':<10} | {'Est. κ_inf(H)':<20} | {'Actual κ_inf(H)':<20}")
    print("-" * 55)

    for n in range(5, 21):
        # 1. Create the Hilbert matrix
        H = scipy.linalg.hilbert(n)
        
        # 2. Calculate the inverse explicitly
        # Note: For larger n, this is numerically unstable, but fine for this range.
        try:
            H_inv = scipy.linalg.inv(H)
        except np.linalg.LinAlgError:
            print(f"{n:<10} | {'Matrix is singular':<43}")
            continue

        # 3. Calculate ||H||_inf directly (max absolute row sum)
        norm_H_inf = np.linalg.norm(H, np.inf)

        # 4. Estimate ||H_inv||_inf using our algorithm
        # We use the property ||A||_inf = ||A^T||_1
        # So we estimate the 1-norm of the transpose of H_inv
        est_norm_H_inv_inf = estimate_norm_1(H_inv.T)
        
        # 5. Calculate the estimated condition number
        est_kappa_inf = norm_H_inf * est_norm_H_inv_inf
        
        # 6. For comparison, calculate the actual condition number using numpy
        actual_kappa_inf = np.linalg.cond(H, np.inf)

        print(f"{n:<10} | {est_kappa_inf:<20.4e} | {actual_kappa_inf:<20.4e}")

def create_A(n):
    """Creates the special matrix A_n as described in the problem."""
    A = np.tril(np.full((n, n), -1.0))
    np.fill_diagonal(A, 1.0)
    A[:, -1] = 1.0
    return A

def solve_special_matrix_problem():
    """
    For the special matrix A_n, estimates the precision of a computed
    solution and compares it to the true relative error.
    """
    print("\n--- Problem 2.3: Precision Estimation for Special Matrix A_n ---")
    print(f"{'Order (n)':<10} | {'True Rel. Error':<20} | {'Estimated Error Bound':<25}")
    print("-" * 60)

    # Machine epsilon for standard double precision float
    machine_epsilon = np.finfo(float).eps

    for n in range(5, 31):
        # 1. Create the matrix A_n
        A = create_A(n)
        
        # 2. Generate a known solution and the corresponding right-hand side
        x_true = np.random.rand(n)
        b = A @ x_true
        
        # 3. Solve the system A * x_hat = b
        x_computed = scipy.linalg.solve(A, b)
        
        # 4. Calculate the true relative error
        true_relative_error = (np.linalg.norm(x_true - x_computed, np.inf) / 
                               np.linalg.norm(x_true, np.inf))
        
        # 5. Estimate the condition number kappa_inf(A)
        try:
            A_inv = scipy.linalg.inv(A)
        except np.linalg.LinAlgError:
            print(f"{n:<10} | {'Matrix is singular':<48}")
            continue
            
        norm_A_inf = np.linalg.norm(A, np.inf)
        # Estimate ||A_inv||_inf by finding the 1-norm of its transpose
        est_norm_A_inv_inf = estimate_norm_1(A_inv.T)
        est_kappa_inf = norm_A_inf * est_norm_A_inv_inf
        
        # 6. Calculate the estimated error bound
        error_bound = est_kappa_inf * machine_epsilon
        
        print(f"{n:<10} | {true_relative_error:<20.4e} | {error_bound:<25.4e}")

if __name__ == '__main__':
    # Run the solution for Problem 2.2
    solve_hilbert_problem()
    
    # Run the solution for Problem 2.3
    solve_special_matrix_problem()