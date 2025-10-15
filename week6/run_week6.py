import numpy as np
import pandas as pd
from qr_solver import qr_linear_solver, qr_least_squares

def solve_task1():
    print("--- Task 1: Solving Linear Systems ---")
    
    # Equation 1
    n1 = 84
    A1 = np.zeros((n1, n1))
    for i in range(n1):
        if i > 0:
            A1[i, i-1] = 8
        A1[i, i] = 6
        if i < n1 - 1:
            A1[i, i+1] = 1
    b1 = np.full(n1, 15.0)
    b1[0] = 7
    b1[-1] = 14
    
    x1_qr = qr_linear_solver(A1, b1)
    x1_gauss = np.linalg.solve(A1, b1)
    
    print("\nEquation 1 (n=84):")
    print("Solution with QR:", x1_qr)
    print("Solution with Gauss (numpy.linalg.solve):", x1_gauss)
    print("Difference:", np.linalg.norm(x1_qr - x1_gauss))

    # Equation 2
    n2 = 100
    A2 = np.zeros((n2, n2))
    for i in range(n2):
        if i > 0:
            A2[i, i-1] = 1
        A2[i, i] = 10
        if i < n2 - 1:
            A2[i, i+1] = 1
    b2 = np.full(n2, 12.0)
    b2[0] = 11
    b2[-1] = 11
    
    x2_qr = qr_linear_solver(A2, b2)
    # Cholesky decomposition for comparison
    L = np.linalg.cholesky(A2)
    y = np.linalg.solve(L, b2)
    x2_cholesky = np.linalg.solve(L.T, y)

    print("\nEquation 2 (n=100):")
    print("Solution with QR:", x2_qr)
    print("Solution with Cholesky:", x2_cholesky)
    print("Difference:", np.linalg.norm(x2_qr - x2_cholesky))

    # Equation 3 (Hilbert Matrix)
    n3 = 40
    A3 = np.zeros((n3, n3))
    b3 = np.zeros(n3)
    for i in range(n3):
        for j in range(n3):
            A3[i, j] = 1.0 / (i + j + 1)
        b3[i] = np.sum([1.0 / (k + i + 1) for k in range(n3)])

    x3_qr = qr_linear_solver(A3, b3)
    x3_numpy = np.linalg.solve(A3, b3)
    print("\nEquation 3 (Hilbert Matrix, n=40):")
    print("Solution with QR:", x3_qr)
    print("Solution with numpy.linalg.solve:", x3_numpy)
    print("Difference:", np.linalg.norm(x3_qr - x3_numpy))
    print("Note: The Hilbert matrix is ill-conditioned. Large differences are expected.")


def solve_task2():
    print("\n--- Task 2: Polynomial Curve Fitting ---")
    
    t = np.array([-1, -0.75, -0.5, 0, 0.25, 0.5, 0.75])
    y = np.array([1.00, 0.8125, 0.75, 1.00, 1.3125, 1.75, 2.3125])
    
    A = np.vstack([t**2, t, np.ones(len(t))]).T
    
    # Solve for [a, b, c]
    coeffs = qr_least_squares(A, y)
    
    print("The polynomial is y = a*t^2 + b*t + c")
    print(f"Coefficients: a = {coeffs[0]}, b = {coeffs[1]}, c = {coeffs[2]}")
    
    # Also solve with numpy's lstsq for comparison
    coeffs_numpy = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"Numpy's lstsq result: a = {coeffs_numpy[0]}, b = {coeffs_numpy[1]}, c = {coeffs_numpy[2]}")


def solve_task3():
    print("\n--- Task 3: Real Estate Price Prediction ---")
    
    # Load data
    A_df = pd.read_csv('Week6 Data/Week6 Data/A.csv', header=None)
    y_df = pd.read_csv('Week6 Data/Week6 Data/y.csv', header=None)
    
    A = A_df.values
    y = y_df.values.flatten()
    
    # Add a column of ones for the intercept term
    A_with_intercept = np.hstack([np.ones((A.shape[0], 1)), A])
    
    # Solve for the parameters
    params = qr_least_squares(A_with_intercept, y)
    
    print("Model: y = a0 + a1*x1 + ... + a11*x11")
    print("Parameters (a0, a1, ..., a11):")
    print(params)
    
    # Also solve with numpy's lstsq for comparison
    params_numpy = np.linalg.lstsq(A_with_intercept, y, rcond=None)[0]
    print("\nNumpy's lstsq result:")
    print(params_numpy)
    print("\nDifference:", np.linalg.norm(params - params_numpy))


if __name__ == '__main__':
    solve_task1()
    solve_task2()
    solve_task3()
