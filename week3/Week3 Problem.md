# Week3 Problem

### 1.1 不选主元与列主元 Gauss 消去法的比较

考虑下列线性方程组
$$
\begin{bmatrix}6&1\\8&6&1\\&\ddots&\ddots&\ddots\\&&8&6&1\\&&&8&6\end{bmatrix}\begin{bmatrix}x_1\\x_2\\\vdots\\x_{n-1}\\x_n\end{bmatrix}=\begin{bmatrix}7\\15\\\vdots\\15\\14\end{bmatrix}
$$
已知精确解是 $x^{*}=(1,\cdot\cdot\cdot,1)^{T}$。对于数值解，我们定义其与精确解的误差的$l^2$范数和$l^\infty$范数 $||x^{*}-x||_{2}$，$||x^{*}-x||_{\infty}$ 为
$$
||x^{*}-x||_{2}=(\sum_{i=1}^{n}|x_{i}^{*}-x_{i}|^{2})^{\frac{1}{2}}, \quad ||x^{*}-x||_{\infty}=\max_{1\le i\le n}\{|x_{i}^{*}-x_{i}|\}
$$
现分别取 $n=2,12,24,48,84$，分别采用 Gauss 消去法和列主元 Gauss 消去法求解上述方程组，计算数值解与精确解的误差的两种范数，并将误差随$n$的变化用表格或者图像进行展示，由此谈谈你对Gauss 消去法的看法。

### 1.2 Gauss 消去法、平方根法与带状平方根法的比较

利用五点差分格式去近似求解 Dirichlet 边界的二维 Possion 方程
$$
\begin{cases}
-\frac{\partial^{2}u}{\partial x^{2}}-\frac{\partial^{2}u}{\partial y^{2}}=f, & (x,y)\in\Omega, \\ 
u=0, & (x,y)\in\partial\Omega,
\end{cases}
$$
其中区域 $\Omega=(0,1)\times(0,1)$，源函数 $f=2\pi^{2}sin(\pi x)sin(\pi y)$， 真解 $u=sin(\pi x)sin(\pi y)$。


对于区域进行均匀剖分，沿x轴和沿y轴方向分别以间距 $h_{x}$ 和 $h_{y}$ 来进行等分，在这里只考虑 $h=h_{x}=h_{y}=\frac{1}{N}$ 的N等分情况。对于 $1\le i, j\le N+1,$ 记 $x_{i}=(i-1)h_{x}$，$y_{j}=(j-1)h_{y}$ 以及 $u_{i,j}=u(x_{i},y_{j})$，$f_{i,j}=f(x_{i},y_{j})$。

当 $(x_{i},y_{j})$ 为$\Omega$内部点时，即 $2\le i, j\le N$，利用二阶中心差分格式去近似 $\frac{\partial^{2}u}{\partial x^{2}}$ 和 $\frac{\partial^{2}u}{\partial y^{2}}$ 即
$$
\frac{\partial^{2}u}{\partial x^{2}}\approx\frac{u_{i+1,j}-2u_{i,j}+u_{i-1,j}}{h_{x}^{2}}
$$
$$
\frac{\partial^{2}u}{\partial y^{2}}\approx\frac{u_{i,j+1}-2u_{i,j}+u_{i,j-1}}{h_{y}^{2}},
$$
进而得到近似方程
$$
4u_{i,j}-u_{i+1,j}-u_{i-1,j}-u_{i,j+1}-u_{i,j-1}\approx h^{2}f_{i,j}, \quad 2\le i, j \le N.
$$
再由边界条件，可知
$$
u_{1,j}=u_{N+1,j}=u_{i,1}=u_{i,N+1}=0, \quad 1\le i,j\le N+1.
$$
令 $U_{i,j}$ 为如下方程组的数值解，$1\le i,j \le N + 1$
$$
\begin{cases}
4U_{i,j}-U_{i+1,j}-U_{i-1,j}-U_{i,j+1}-U_{i,j-1} = h^{2}f_{i,j}, & 2\le i,j\le N, \\ 
U_{1,j}=U_{N+1,j}=U_{i,1}=U_{i,N+1}=0, & 1\le i,j\le N+1.
\end{cases}
$$
那么 $U_{i,j}$，$2\le i,j\le N$ 可视为 $u_{i,j}$ 的数值近似，其中需求解的未知数只有内部点 $U_{i,j}$，将其按列排列成一个向量
$$
U_{h}=(U_{2,2},U_{2,3},...,U_{2,N},U_{3,2},U_{3,3},...,U_{3,N},...,U_{N,2},U_{N,3},...,U_{N,N})^{T},
$$
同时将 $f_{i,j}$ 也按列排列成一个向量
$$
F_{h}=(f_{2,2},f_{2,3},...,f_{2,N},f_{3,2},f_{3,3},...,f_{3,N},...,f_{N,2},f_{N,3},...,f_{N,N})^{T},
$$
可以得到一个关于 $U_{h}$ 的线性方程组
$$
A_{h}U_{h}=h^{2}F_{h}
$$
其中
$$
A_h =
\begin{bmatrix}
X & -I & & & \\
-I & X & -I & & \\
& \ddots & \ddots & \ddots & \\
& & -I & X & -I \\
& & & -I & X
\end{bmatrix}_{(N-1)^2 \times (N-1)^2}
$$
$$
X =
\begin{bmatrix}
4 & -1 & & & \\
-1 & 4 & -1 & & \\
& \ddots & \ddots & \ddots & \\
& & -1 & 4 & -1 \\
& & & -1 & 4
\end{bmatrix}_{(N-1) \times (N-1)}
$$
为了衡量数值解的精度，我们定义离散误差为
$$
e_{N}=h\left(\sum_{i=1}^{N+1}\sum_{j=1}^{N+1}|u_{i,j}-U_{i,j}|^{2}\right)^{1/2}
$$
对于 $N=16,32,64,128,256$ 时，分别利用 Gauss 消去法、$LDL^{T}$ 方法与带状 $LDL^{T}$ 方法求解上述线性方程组，将不同算法的计算求解时间与计算误差通过表格或图像进行展示， 由此来比较、分析各情况、方法的效果。

---

截止时间为10月13日下午15:00，需要同时提交代码和上机报告，上机报告需使用LaTeX 或Markdown完成，并提交 PDF文档，总页数不得超过10页。在提交上机作业时，需将代码、上机报告源码、上机报告打包为 zip压缩包，作为附件发送至邮箱 `2501110054@stu.pku.edu.cn`，其中邮件名、压缩包、上机报告需按“数值代数+第一次上机作业+姓名+学号(.zip/.pdf)" 的格式命名，如：“数值代数+第一次上机作业+许盛+2501110054(.zip/.pdf)”。