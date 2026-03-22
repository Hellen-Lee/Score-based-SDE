# SDE 相关类和方法
# 初学者可先理解两条线：
# 1）VariancePreservingSDE：「前向」扩散——从干净数据 x 加噪得到 y_t（训练时从中采样一对 (t, y)）。
# 2）PluginReverseSDE：把神经网络 drift_a 接到 VP-SDE 上，得到反向漂移 mu/扩散 sigma，并提供 dsm 训练目标。

import torch
import numpy as np

class VariancePreservingSDE(torch.nn.Module):
    """方差保持（VP）SDE：漂移把状态拉向 0，扩散强度由 β(t) 决定；t 常归一化到 [0, T]。"""

    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.t_epsilon = t_epsilon

    def beta(self, t):
        """噪声强度调度：t=0 弱，t=1 强（线性插值）。"""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def mean_weight(self, t):
        """条件分布 y_t|x_0 的均值系数 e^{-∫β/2}（VP 闭式）。"""
        return torch.exp(-0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min)

    def var(self, t):
        """给定 t 时，y_t 相对 x_0 的条件方差（VP 闭式）。"""
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max - self.beta_min) - t * self.beta_min)

    def f(self, t, y):
        """SDE 漂移项 f(t,y)，与 VP 形式对应。"""
        return - 0.5 * self.beta(t) * y

    def g(self, t, y):
        """扩散系数 √β(t)，各向同性加噪。"""
        beta_t = self.beta(t)
        return torch.ones_like(y) * beta_t**0.5

    def euler_maruyama_sampling(self, y0, num_steps, T):
        """Euler–Maruyama：离散近似 SDE，随机项含 sqrt(dt)（标准写法）。"""
        dt = T / num_steps
        y = y0
        for _ in range(num_steps):
            t = torch.ones_like(y0) * (T - dt)
            f_val = self.f(t, y)
            g_val = self.g(t, y)
            noise = torch.randn_like(y)
            y = y + f_val * dt + g_val * torch.sqrt(torch.tensor(dt)) * noise
        return y

    def stochastic_euler_sampling(self, y0, num_steps, T):
        """另一种离散化：随机项不乘 sqrt(dt)，与上式步长标度不同。"""
        dt = T / num_steps
        y = y0
        for _ in range(num_steps):
            t = torch.ones_like(y0) * (T - dt)
            f_val = self.f(t, y)
            g_val = self.g(t, y)
            noise = torch.randn_like(y)
            y = y + f_val * dt + g_val * noise
        return y

    def sample(self, t, y0, return_noise=False):
        """由 x_0=y0 一步采样 y_t ~ N(μ(t)y0, var(t)I)；DSM 需要 ε、std、g 时设 return_noise=True。"""
        mu = self.mean_weight(t) * y0
        std = self.var(t) ** 0.5
        epsilon = torch.randn_like(y0)
        yt = epsilon * std + mu
        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t, yt)

    def sample_debiasing_t(self, shape):
        """非均匀采样时间 t，缓解小 t 附近损失权重问题；与 utils 中截断分布配套。"""
        from utils import sample_vp_truncated_q
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)

class PluginReverseSDE(torch.nn.Module):
    """在 base_sde 上「插件」可学习漂移 a(y,t)（通常为 UNet），构造反向 SDE 的 μ、σ 与损失。"""

    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = torch.tensor(T, dtype=torch.float32)  # 参与运算/优化时需为张量（如记录到 TensorBoard）
        self.vtype = vtype
        self.debias = debias

    def mu(self, t, y, lmbd=0.):
        """反向 SDE 漂移：含网络项 g·a 与前向漂移的修正 -f；t 为反向积分时间标度。"""
        return (1. - 0.5 * lmbd) * self.base_sde.g(self.T - t, y) * self.a(y, self.T - t.squeeze()) - \
               self.base_sde.f(self.T - t, y)

    def sample_euler_maruyama(self, x, num_steps, T):
        """当前实现：直接调用前向 VP-SDE 的离散积分（与 get_grid 使用的 μ/σ 反向链不同）。"""
        return self.base_sde.euler_maruyama_sampling(x, num_steps, T)

    def sample_stochastic_euler(self, x, num_steps, T):
        """同族接口：转调前向 stochastic_euler（见 VariancePreservingSDE）。"""
        return self.base_sde.stochastic_euler_sampling(x, num_steps, T)

    def sigma(self, t, y, lmbd=0.):
        """反向 SDE 扩散系数，正比于前向 g(T−t, y)。"""
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T - t, y)

    @torch.enable_grad()
    def dsm(self, x):
        """Denoising score matching：随机 t，前向采样 y，令网络输出与噪声项对齐（内部用 MSE 型损失）。"""
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        y, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        a = self.a(y, t_.squeeze())  # target 即前向采样中的 ε；std、g 用于配平尺度

        return ((a * std / g + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def elbo_random_t_slice(self, x):
        """变分下界相关目标：用随机向量 v 估计散度项（Hutchinson）；默认训练脚本未调用。"""
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.a(y, t_.squeeze())
        # 与反向 SDE 漂移相关的组合量，对 y 求导时需要保留计算图
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        from utils import sample_v
        v = sample_v(x.shape, vtype=self.vtype).to(y)

        # v^T ∇_y μ 的蒙特卡洛估计，用于散度相关项
        Mu = - (
            torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = - (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        # 极大 t 下近似标准高斯端点的对数密度（此处 mean=0, log_var=0）
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        from utils import log_normal
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu