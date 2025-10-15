import numpy as np

def householder_qr(A):
    """
    Performs QR decomposition of a matrix A using Householder reflections.
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(n):
        # For non-square matrices, we only need to process up to the smaller dimension
        if j >= m:
            continue
            
        x = R[j:, j]
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        
        # The sign is chosen to avoid cancellation
        v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
        v = v / np.linalg.norm(v)
        
        # Apply the reflection to R
        R[j:, :] = R[j:, :] - 2.0 * np.outer(v, v @ R[j:, :])
        
        # Apply the reflection to Q
        Q[:, j:] = Q[:, j:] - 2.0 * (Q[:, j:] @ v)[:, np.newaxis] * v[np.newaxis, :]
        
    return Q, R

def qr_linear_solver(A, b):
    """
    Solves the linear system Ax = b using QR decomposition.
    """
    m, n = A.shape
    if m != n:
        raise ValueError("Input matrix must be square.")
        
    Q, R = householder_qr(A)
    
    # Solve Rx = Q^T b using back substitution
    y = Q.T @ b
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - (R[i, i+1:] @ x[i+1:])) / R[i, i]
        
    return x

def qr_least_squares(A, y):
    """
    Solves the least squares problem min ||Ax - y||_2 using QR decomposition.
    """
    m, n = A.shape
    Q, R = householder_qr(A)
    
    # We need to solve Rx = (Q^T y)[:n]
    qty = Q.T @ y
    
    # R_hat is the upper n x n part of R
    R_hat = R[:n, :n]
    qty_hat = qty[:n]
    
    x = np.zeros(n)
    
    # Back substitution on R_hat x = qty_hat
    for i in range(n - 1, -1, -1):
        x[i] = (qty_hat[i] - (R_hat[i, i+1:] @ x[i+1:])) / R_hat[i, i]
        
    return x
