# Numerical_Algebra

本仓库包含数值代数课程作业与大作业代码（Python）。下面给出**大作业（Stokes 问题）**相关脚本的运行说明与结果输出位置。

## 运行环境

- Python 3.10+（本项目使用 3.12 测试）
- 依赖：`numpy`，`scipy`，可选 `numba`

安装依赖（任选其一环境）：

```
pip install numpy scipy numba
```

> 若未安装 `numba`，代码会自动回退到非 JIT 版本。

## 脚本位置

大作业代码位于：

```
大作业/数值代数+大作业+樊泽羲+2200010816/
```

主要脚本：

- `stokes_V_cycle.py`：题 1，DGS-V-cycle 多重网格
- `stokes_Uzawa_Iteration.py`：题 2，Uzawa 迭代
- `stokes_Inexact_Uzawa_Iteration.py`：题 3，Inexact Uzawa + PCG(MG)

## 运行方式

以下示例为 PowerShell 形式（每行一个命令）：

```
python .\大作业\数值代数+大作业+樊泽羲+2200010816\stokes_V_cycle.py
python .\大作业\数值代数+大作业+樊泽羲+2200010816\stokes_Uzawa_Iteration.py
python .\大作业\数值代数+大作业+樊泽羲+2200010816\stokes_Inexact_Uzawa_Iteration.py
```

## 常用环境变量（可选）

### V-cycle（题 1）

- `MAX_N`：限制最大网格尺寸（默认 2048）
- `ONLY_CONFIG`：仅运行某一组参数（如 `nu4_L4`）
- `FORCE_RERUN=1`：覆盖已有 CSV
- `USE_NUMBA=0|1`：关闭/开启 Numba

示例：

```
$env:ONLY_CONFIG="nu4_L4"
$env:MAX_N="512"
python .\大作业\数值代数+大作业+樊泽羲+2200010816\stokes_V_cycle.py
```

### Uzawa（题 2）

脚本内固定运行 $N=\{64,128,256,512\}$，默认无需额外参数。

### Inexact Uzawa（题 3）

- `USE_MG=0|1`：是否使用多重网格预条件
- `MG_DIRECT=0|1`：MG 作为直接解算或预条件
- `PRECOND_VCYCLES`：预条件 V-cycle 次数
- `INNER_MAXITER`：内层 PCG 最大迭代
- `OUTER_TOL`：外层收敛阈值（默认 $10^{-8}$）
- `COUPLED_VCYCLE=0|1`：耦合 $u,v$ V-cycle
- `ONLY_CONFIG`：仅运行某一参数组（如 `tau1e-3_eps1e-8_alpha1_nu2_L2`）
- `MAX_N`、`FORCE_RERUN`、`USE_NUMBA` 同上

示例：

```
$env:USE_MG="1"
$env:PRECOND_VCYCLES="3"
$env:INNER_MAXITER="120"
$env:OUTER_TOL="1e-8"
$env:COUPLED_VCYCLE="1"
$env:ONLY_CONFIG="tau1e-3_eps1e-8_alpha1_nu2_L2"
python .\大作业\数值代数+大作业+樊泽羲+2200010816\stokes_Inexact_Uzawa_Iteration.py
```

## 结果输出

所有结果将写入：

```
大作业/数值代数+大作业+樊泽羲+2200010816/results/
```

子目录结构：

- `v_cycle/`：题 1 结果（CSV + residuals）
- `uzawa/`：题 2 结果
- `inexact_uzawa/`：题 3 结果（按参数组分文件夹）

## 说明

- 压力采用零均值归一化处理。
- 若重复运行同一配置且未开启 `FORCE_RERUN`，脚本会跳过已有的 $N$ 条目。
