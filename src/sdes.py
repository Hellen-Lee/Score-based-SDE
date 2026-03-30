# SDE相关类和方法

import torch
import numpy as np

class VariancePreservingSDE(torch.nn.Module):
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def mean_weight(self, t):
        return torch.exp(-0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min)

    def var(self, t):
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max - self.beta_min) - t * self.beta_min)

    def f(self, t, y):
        return - 0.5 * self.beta(t) * y

    def g(self, t, y):
        beta_t = self.beta(t)
        return torch.ones_like(y) * beta_t**0.5

    def sample(self, t, y0, return_noise=False):
        mu = self.mean_weight(t) * y0
        std = self.var(t) ** 0.5
        epsilon = torch.randn_like(y0)
        yt = epsilon * std + mu
        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t, yt)

    def sample_debiasing_t(self, shape):
        from utils import sample_vp_truncated_q
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)

class PluginReverseSDE(torch.nn.Module):
    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = torch.tensor(T, dtype=torch.float32)  # 转换为张量类型
        self.vtype = vtype
        self.debias = debias

    def mu(self, t, y, lmbd=0.):
        return (1. - 0.5 * lmbd) * self.base_sde.g(self.T - t, y) * self.a(y, self.T - t.squeeze()) - \
               self.base_sde.f(self.T - t, y)

    @torch.no_grad()
    def sample_euler_maruyama(self, x, num_steps, T=None):
        """Euler-Maruyama 法采样（论文 Algorithm 1）
        对反向 SDE 做标准 EM 离散化：y_{i+1} = y_i + mu*dt + sigma*sqrt(dt)*z
        """
        T = self.T if T is None else torch.tensor(T, dtype=torch.float32)
        dt = T / num_steps
        y = x
        ones = torch.ones(x.size(0), *([1] * (x.ndim - 1))).to(x.device)
        for i in range(num_steps):
            t = ones * (i * dt)
            drift = self.mu(t, y)
            diffusion = self.sigma(t, y)
            y = y + drift * dt + diffusion * dt ** 0.5 * torch.randn_like(y)
        return y

    @torch.no_grad()
    def sample_stochastic_euler(self, x, num_steps, T=None):
        """随机化欧拉法 REM 采样（论文 Algorithm 2 / 公式 4.1）
        每步先用均匀随机 U_n 计算中间点，再在中间点处评估 score 完成全步更新，
        使用双噪声 xi' 和 xi 以降低离散化误差。
        """
        T = self.T if T is None else torch.tensor(T, dtype=torch.float32)
        h = T / num_steps
        y = x
        ones = torch.ones(x.size(0), *([1] * (x.ndim - 1))).to(x.device)
        for n in range(num_steps):
            t_n = ones * (n * h)
            U_n = torch.rand(1).item()

            drift_n = self.mu(t_n, y)
            sigma_n = self.sigma(t_n, y)
            xi_prime = torch.randn_like(y)
            y_mid = y + h * U_n * drift_n + sigma_n * (h * U_n) ** 0.5 * xi_prime

            t_mid = ones * ((n + U_n) * h)
            drift_mid = self.mu(t_mid, y_mid)
            sigma_mid = self.sigma(t_mid, y_mid)
            xi = torch.randn_like(y)
            y = y + h * drift_mid + sigma_mid * h ** 0.5 * xi
        return y

    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T - t, y)

    @torch.enable_grad()
    def dsm(self, x):
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        y, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        a = self.a(y, t_.squeeze())

        return ((a * std / g + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def elbo_random_t_slice(self, x):
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.a(y, t_.squeeze())
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        from utils import sample_v
        v = sample_v(x.shape, vtype=self.vtype).to(y)

        Mu = - (
            torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = - (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        from utils import log_normal
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu