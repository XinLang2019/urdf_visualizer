import numpy as np
import matplotlib.pyplot as plt

def compute_diffusion_kernel(N, i, H, h, beta1, beta2, dim):
    """
    计算扩散噪声核 Σ_{t+h}^i.

    参数:
        N (int): 总迭代次数。
        i (int): 当前采样编号 (1 <= i <= N)。
        H (int): 预测时域长度。
        h (int): 当前时间步 (0 <= h <= H)。
        beta1 (float): 调整扩散噪声的参数 β1。
        beta2 (float): 调整扩散噪声的参数 β2。
        dim (int): 控制输入的维度。

    返回:
        numpy.ndarray: 扩散噪声核 (dim x dim)。
    """
    scaling_factor = np.exp(- (N - i) / (beta1 * N) - (H - h) / (beta2 * H))
    return scaling_factor * np.eye(dim)

def sample_diffusion_noise(N, i, H, h, beta1, beta2, dim, N_W):
    """
    从扩散噪声核中采样扩散噪声.

    参数:
        N, i, H, h, beta1, beta2, dim: 与扩散核计算相关的参数。
        N_W (int): 噪声样本的数量。

    返回:
        numpy.ndarray: 扩散噪声样本 (dim x N_W)。
    """
    Sigma_t_h_i = compute_diffusion_kernel(N, i, H, h, beta1, beta2, dim)
    L = np.linalg.cholesky(Sigma_t_h_i)  # Cholesky 分解
    Z = np.random.randn(dim, N_W)       # 标准正态分布
    W = L @ Z                           # 线性变换
    return W

# 参数设置
N = 100         # 总迭代次数
H = 10          # 预测时域长度
beta1 = 0.5     # β1 参数
beta2 = 0.5     # β2 参数
dim = 2         # 控制输入的维度 (为了便于可视化，选择2维)
i = 50          # 当前采样编号
h = 5           # 当前时间步
N_W = 500       # 噪声样本数量

# 生成噪声样本
W_samples = sample_diffusion_noise(N, i, H, h, beta1, beta2, dim, N_W)

# 绘制噪声样本分布
plt.figure(figsize=(8, 6))
plt.scatter(W_samples[0, :], W_samples[1, :], alpha=0.6, edgecolor='k', s=20)
plt.title("扩散噪声样本分布", fontsize=16)
plt.xlabel("$W_1$", fontsize=14)
plt.ylabel("$W_2$", fontsize=14)
plt.grid(alpha=0.3)
plt.axis('equal')
plt.show()