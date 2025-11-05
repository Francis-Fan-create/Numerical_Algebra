import numpy as np
import sys

# 设置Numpy打印选项，使其更容易阅读
np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ============================================================================
# 求解器实现 (SOLVER IMPLEMENTATIONS)
# (这部分与 v1 相同)
# ============================================================================

def jacobi_method(A, b, x0, tol=1e-8, max_iter=1000):
    """
    实现 Jacobi 迭代法求解 Ax = b.
    """
    n = len(b)
    x = x0.copy()
    D = np.diag(A)
    R = A - np.diag(D)  # R = L + U

    for k in range(max_iter):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x) < tol:
            return x_new, k + 1  # 成功收敛
        x = x_new
    return x, max_iter  # 达到最大迭代次数

def gauss_seidel_method(A, b, x0, tol=1e-8, max_iter=1000):
    """
    实现 Gauss-Seidel 迭代法求解 Ax = b.
    """
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum_j_lt_i = A[i, :i] @ x[:i]
            sum_j_gt_i = A[i, i+1:] @ x_old[i+1:]
            x[i] = (b[i] - sum_j_lt_i - sum_j_gt_i) / A[i, i]
            
        if np.linalg.norm(x - x_old) < tol:
            return x, k + 1  # 成功收敛
    return x, max_iter  # 达到最大迭代次数

def conjugate_gradient_method(A, b, x0, tol=1e-8, max_iter=1000):
    """
    实现共轭梯度法 (CG) 求解 Ax = b.
    **注意: 此方法要求 A 是对称正定 (SPD) 矩阵.
    """
    x = x0.copy()
    r = b - A @ x        # 初始残量
    p = r.copy()         # 初始搜索方向
    rs_old = r.T @ r     # r_k^T * r_k

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p.T @ Ap)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        rs_new = r.T @ r
        if np.sqrt(rs_new) < tol:
            return x, k + 1  # 成功收敛
            
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, max_iter  # 达到最大迭代次数

# ============================================================================
# 辅助函数 (HELPER FUNCTIONS)
# (这部分与 v1 相同)
# ============================================================================

def create_hilbert(n):
    """创建 n 阶 Hilbert 矩阵 H_n"""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1) # (i+1) + (j+1) - 1
    return H

def check_sdd(A):
    """检查矩阵 A 是否为严格对角占优 (SDD)"""
    D_abs = np.diag(np.abs(A))
    R_sums = np.sum(np.abs(A), axis=1) - D_abs
    
    if np.all(D_abs > R_sums):
        return "是严格对角占优 (SDD)"
    if np.all(D_abs >= R_sums) and np.any(D_abs > R_sums):
        return "是弱对角占优"
    return "非对角占优"

# ============================================================================
# 任务 (1): HILBERT 矩阵求解
# (这部分与 v1 相同)
# ============================================================================

def task_1():
    print("=" * 70)
    print(" 任务 (1): 求解 Hilbert 矩阵线性方程组 H_n * x = b")
    print(" " + "=" * 68)
    print(" b_i = (1/3) * sum(H_ij for j=1..n)\n")
    
    dims = [5, 8, 10, 12, 14]
    
    print(f"{'n':>3} | {'Cond(H_n)':>15} | {'CG 迭代次数':>12} | {'最终相对残量':>15}")
    print("-" * 70)
    
    for n in dims:
        H = create_hilbert(n)
        b = (1.0 / 3.0) * np.sum(H, axis=1)
        x0 = np.zeros(n)
        cond_H = np.linalg.cond(H)
        max_iter_cg = max(2000, 2 * n) 
        x_cg, iters_cg = conjugate_gradient_method(H, b, x0, tol=1e-10, max_iter=max_iter_cg)
        
        final_residual = b - H @ x_cg
        rel_residual_norm = np.linalg.norm(final_residual) / np.linalg.norm(b)
        
        iter_str = str(iters_cg)
        if iters_cg == max_iter_cg:
            iter_str = f"> {max_iter_cg} (未收敛)"
            
        print(f"{n:3d} | {cond_H:15.5e} | {iter_str:>12} | {rel_residual_norm:15.5e}")

# ============================================================================
# 任务 (2): 比较 JACOBI, GAUSS-SEIDEL, CG
# (这部分已根据您的更正进行更新)
# ============================================================================

def task_2():
    print("\n" + "=" * 70)
    print(" 任务 (2): 比较 Jacobi, Gauss-Seidel 和 CG 方法 (使用更正后数据)")
    print(" " + "=" * 68)

    # --- 使用您提供的正确数据 ---
    A = np.array([
        [10.,  1.,  2.,  3.,  4.],
        [ 1.,  9., -1.,  2., -3.],
        [ 2., -1.,  7.,  3., -5.], # A(3,5) = -5
        [ 3.,  2.,  3., 12., -1.],
        [ 4., -3., -5., -1., 15.] # A(5,3) = -5, A(5,4) = -1
    ])
    
    b = np.array([12., -27., 14., -17., 12.]) # b(4) = -17
    
    x0 = np.zeros(5)
    tol = 1e-8
    max_iter = 500

    print("求解系统 Ax = b，其中：")
    print("A = \n", A)
    print("\nb = \n", b)
    print("-" * 70)

    # --- 1. 检查应用条件 ---
    print("--- 1. 检查算法应用条件 ---")
    
    # 检查对称性 (CG)
    is_symmetric = np.allclose(A, A.T)
    print(f"A 是否对称? {is_symmetric}")
    
    # 检查正定性 (CG)
    is_pd = False
    if is_symmetric:
        try:
            # 尝试计算特征值
            eigenvalues = np.linalg.eigvalsh(A)
            is_pd = np.all(eigenvalues > 0)
            print(f"A 是否正定? {is_pd} (最小特征值: {eigenvalues.min():.4f})")
            if is_pd:
                print("  -> 结论: 矩阵 A 是对称正定 (SPD) 的。CG 方法适用。")
            else:
                 print(f"  -> 警告: 矩阵 A 对称但非正定 (最小特征值 <= 0)。CG 方法不适用。")
        except np.linalg.LinAlgError:
            print("  -> 警告: 计算特征值时出错。")
    else:
        print("  -> 警告: 矩阵 A 不是对称的。标准的 CG 方法不适用。")


    # 检查对角占优 (Jacobi, G-S)
    sdd_status = check_sdd(A)
    print(f"\nA 是否对角占优? {sdd_status}")
    if sdd_status == "非对角占优":
        print("  -> 警告: 矩阵 A 不是对角占优的。Jacobi 和 G-S 迭代法不保证收敛。")
        print("\n    手动检查对角占优性:")
        print("    R1: |10| = |1|+|2|+|3|+|4| = 10 (弱)")
        print("    R2: |9|  > |1|+|-1|+|2|+|-3| = 7 (严格)")
        print("    R3: |7|  < |2|+|-1|+|3|+|-5| = 11 (不满足)")
        print("    R4: |12| > |3|+|2|+|3|+|-1| = 9 (严格)")
        print("    R5: |15| > |4|+|-3|+|-5|+|-1| = 13 (严格)")
        print("    -> 结论: 由于第 3 行，A 不是对角占优。")

    print("-" * 70)

    # --- 2. 运行求解器 ---
    print("--- 2. 运行求解器 ---")
    
    # Jacobi
    try:
        x_j, it_j = jacobi_method(A, b, x0, tol=tol, max_iter=max_iter)
        print(f"\nJacobi 迭代:")
        if it_j >= max_iter:
            print(f"  在 {max_iter} 次迭代后未收敛。")
        else:
            print(f"  在 {it_j} 次迭代后收敛。")
            print(f"  解 x_j = {x_j}")
            print(f"  残量 ||Ax_j - b|| = {np.linalg.norm(A @ x_j - b):.2e}")
            
    except Exception as e:
        print(f"\nJacobi 迭代失败: {e}")

    # Gauss-Seidel
    try:
        x_gs, it_gs = gauss_seidel_method(A, b, x0, tol=tol, max_iter=max_iter)
        print(f"\nGauss-Seidel 迭代:")
        if it_gs >= max_iter:
            print(f"  在 {max_iter} 次迭代后未收敛。")
        else:
            print(f"  在 {it_gs} 次迭代后收敛。")
            print(f"  解 x_gs = {x_gs}")
            print(f"  残量 ||Ax_gs - b|| = {np.linalg.norm(A @ x_gs - b):.2e}")
            
    except Exception as e:
        print(f"\nGauss-Seidel 迭代失败: {e}")

    # Conjugate Gradient
    print(f"\nConjugate Gradient (CG) 迭代:")
    if not is_symmetric or not is_pd:
        print("  ** 警告: 矩阵非 SPD，强行运行 CG... **")
        
    try:
        x_cg, it_cg = conjugate_gradient_method(A, b, x0, tol=tol, max_iter=max_iter)
        
        if it_cg >= max_iter:
            print(f"  在 {max_iter} 次迭代后未收敛。")
        else:
            print(f"  在 {it_cg} 次迭代后收敛。") 
            print(f"  解 x_cg = {x_cg}")
            print(f"  残量 ||Ax_cg - b|| = {np.linalg.norm(A @ x_cg - b):.2e}")
            
    except Exception as e:
        print(f"\nCG 迭代失败: {e}")

    print("=" * 70)


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    task_1()
    task_2()