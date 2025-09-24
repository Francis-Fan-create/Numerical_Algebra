import numpy as np
import matplotlib.pyplot as plt

def gaussian_elimination(A, b):
    """
    Solves the linear system Ax = b using Gaussian elimination without pivoting.

    Args:
        A (np.ndarray): The coefficient matrix.
        b (np.ndarray): The constant vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    # Forward Elimination
    for k in range(n - 1):
        # This check is for cases where the pivot is zero, which would halt the algorithm.
        # For the specific matrix in this problem, this will not happen.
        if A[k, k] == 0:
            raise ZeroDivisionError("Pivot element is zero, cannot continue.")
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Backward Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x

def gaussian_elimination_pivoting(A, b):
    """
    Solves the linear system Ax = b using Gaussian elimination 
    with partial (column) pivoting.

    Args:
        A (np.ndarray): The coefficient matrix.
        b (np.ndarray): The constant vector.

    Returns:
        np.ndarray: The solution vector x.
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    # Forward Elimination with Pivoting
    for k in range(n - 1):
        # Find the row with the largest pivot element in the current column
        max_index = k + np.argmax(np.abs(A[k:, k]))
        
        # Swap rows in A and b if necessary
        if max_index != k:
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]

        if A[k, k] == 0:
            raise ZeroDivisionError("Matrix is singular.")

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Backward Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        
    return x

def run_analysis():
    """
    Runs the full analysis for different system sizes, prints the results,
    and generates a comparison plot.
    """
    n_values = [2, 12, 24, 48, 84]
    
    # Store results for plotting
    results = {
        'no_pivot_l2': [],
        'pivot_l2': [],
        'no_pivot_inf': [],
        'pivot_inf': []
    }

    # Print table header
    print("Table 1: Comparison of numerical errors for Gaussian elimination with and without partial pivoting.")
    print("-" * 110)
    print(f"{'n':<5} | {'Without Pivoting (l2 error)':<30} | {'With Pivoting (l2 error)':<28} | {'Without Pivoting (l-inf error)':<30} | {'With Pivoting (l-inf error)':<28}")
    print("-" * 110)

    for n in n_values:
        # Construct the matrix A
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = 6
            if i > 0:
                A[i, i - 1] = 8
            if i < n - 1:
                A[i, i + 1] = 1
        
        # Construct the vector b
        b = np.full(n, 15.0)
        b[0] = 7.0
        b[-1] = 14.0

        # Exact solution
        x_exact = np.ones(n)

        # --- Solve without pivoting ---
        x_no_pivot = gaussian_elimination(A.copy(), b.copy())
        error_no_pivot_l2 = np.linalg.norm(x_exact - x_no_pivot, 2)
        error_no_pivot_inf = np.linalg.norm(x_exact - x_no_pivot, np.inf)
        results['no_pivot_l2'].append(error_no_pivot_l2)
        results['no_pivot_inf'].append(error_no_pivot_inf)

        # --- Solve with pivoting ---
        x_pivot = gaussian_elimination_pivoting(A.copy(), b.copy())
        error_pivot_l2 = np.linalg.norm(x_exact - x_pivot, 2)
        error_pivot_inf = np.linalg.norm(x_exact - x_pivot, np.inf)
        results['pivot_l2'].append(error_pivot_l2)
        results['pivot_inf'].append(error_pivot_inf)
        
        # Print results in a table row
        print(f"{n:<5} | {error_no_pivot_l2:<30.4e} | {error_pivot_l2:<28.4e} | {error_no_pivot_inf:<30.4e} | {error_pivot_inf:<28.4e}")

    print("-" * 110)

    # --- Plotting the results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    
    ax.plot(n_values, results['no_pivot_l2'], 'o-', label='$l^2$ norm (no pivoting)', color='blue')
    ax.plot(n_values, results['no_pivot_inf'], 's--', label='$l^\\infty$ norm (no pivoting)', color='deepskyblue')
    ax.plot(n_values, results['pivot_l2'], 'o-', label='$l^2$ norm (with pivoting)', color='green')
    ax.plot(n_values, results['pivot_inf'], 's--', label='$l^\\infty$ norm (with pivoting)', color='limegreen')

    ax.set_yscale('log')
    ax.set_xlabel('System Size (n)', fontsize=12)
    ax.set_ylabel('Error Norm', fontsize=12)
    ax.set_title('Error Growth in Gaussian Elimination Methods', fontsize=14, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", c='0.7')
    
    fig.tight_layout()
    plt.savefig('week3/figure.png')
    print("\nPlot saved as 'week3/figure.png'")
    plt.show()


if __name__ == '__main__':
    run_analysis()
