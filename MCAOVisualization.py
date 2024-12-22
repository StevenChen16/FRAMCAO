import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
from scipy.signal import lfilter

# 1. 分数阶导数的实现（Caputo定义）
def fractional_derivative(f, gamma_order, dt):
    N = len(f)
    derivative = np.zeros(N)
    for n in range(N):
        sum_term = 0
        for k in range(n):
            sum_term += (f[k] - f[0]) / ((n - k)**gamma_order)
        derivative[n] = sum_term * gamma(1 - gamma_order) / (dt**gamma_order)
    return derivative

# 2. 非线性耦合函数 σ(x, y) = tanh(phi*(x - y)) * psi(|x| + |y|)
def sigma(x, y, phi=1.0, psi=1.0):
    return np.tanh(phi * (x - y)) * psi * (np.abs(x) + np.abs(y))

# 3. 全局影响项的实现
def global_influence(f, G_func, K_func, t, tau, dt):
    # Numerical integration from t-tau to t
    start = max(0, t - tau)
    end = t
    integrand = lambda s: G_func(f, s) * K_func(t - s)
    result, _ = quad(integrand, start * dt, end * dt)
    return result

# 示例全局状态函数 G 和核函数 K
def G(f, s):
    # 简单取 f 的平均作为全局状态
    return np.mean(f[:int(s)])

def K(t_minus_s):
    # 指数衰减核
    return np.exp(-t_minus_s)

# 4. MCAO 算子的实现
def MCAO_operator(f, alpha, beta, eta, gamma_order, tau, dt):
    N, I = f.shape
    M_f = np.zeros_like(f)
    
    # Compute fractional derivative for each feature
    for i in range(I):
        M_f[:, i] += alpha * fractional_derivative(f[:, i], gamma_order, dt)
    
    # Nonlinear coupling
    for i in range(I):
        for j in range(I):
            if j != i:
                w_ij = 1.0 / (I - 1)  # 简化权重，均等分配
                M_f[:, i] += beta * w_ij * sigma(f[:, i], f[:, j])
    
    # Global influence
    for t in range(N):
        for i in range(I):
            M_f[t, i] += eta * global_influence(f[:, i], G, K, t, tau, dt)
    
    return M_f

# 生成示例数据
np.random.seed(0)
N = 200  # 时间步
I = 3    # 特征数
dt = 0.1
t = np.linspace(0, (N-1)*dt, N)
f = np.sin(t)[:, np.newaxis] + 0.1 * np.random.randn(N, I)

# 参数设置
alpha = 0.5
beta = 0.3
eta = 0.2
gamma_order = 1.5
tau = 10  # 时间窗口

# 计算 MCAO 算子作用后的结果
M_f = MCAO_operator(f, alpha, beta, eta, gamma_order, tau, dt)

# 可视化
plt.figure(figsize=(14, 8))

for i in range(I):
    plt.subplot(I, 1, i+1)
    plt.plot(t, f[:, i], label=f'Original f_{i+1}')
    plt.plot(t, M_f[:, i], label=f'MCAO M(f)_{i+1}')
    plt.legend()
    plt.xlabel('时间')
    plt.ylabel(f'特征 {i+1} 值')

plt.tight_layout()
plt.show()
