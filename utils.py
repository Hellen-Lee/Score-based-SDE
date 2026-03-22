# 工具函数与实验辅助
# - 与 sdes 配合：随机向量 v（Hutchinson）、对数高斯密度、VP 截断时间采样（debias）。
# - 与训练脚本配合：create / logging 管理实验目录与文本日志。
# - 其余指数分布、get_beta 等为通用片段，本仓库主训练流程可不关心。

import numpy as np
import torch

def log_standard_normal(x):
    """标准正态 N(0,1) 的对数密度（逐元素，numpy）。"""
    return - 0.5 * x ** 2 - np.log(2 * np.pi) / 2

def sample_rademacher(shape):
    """独立同分布 Rademacher：取 ±1，概率各半；用于 Hutchinson 迹估计时方差较小。"""
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1

def sample_gaussian(shape):
    """标准高斯向量，用途同上与 v 配合 autograd.grad。"""
    return torch.randn(*shape)

def sample_v(shape, vtype='rademacher'):
    """按类型采样随机方向 v，供 elbo_random_t_slice 等估计 v^T ∇(·)。"""
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        raise Exception(f'vtype {vtype} not supported')

Log2PI = float(np.log(2 * np.pi))

def log_normal(x, mean, log_var, eps=0.00001):
    """对角高斯 log p(x)；log_var 为 log σ²。elbo 里端点项会用到。"""
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z

def exponential_CDF(t, lamb):
    return 1 - torch.exp(- lamb * t)

def sample_truncated_exponential(shape, lamb, T):
    """[0,T] 上截断指数分布采样（通用工具，主脚本未用）。"""
    if lamb > 0:
        return - torch.log(1 - torch.rand(*shape).to(T) * exponential_CDF(T, lamb) + 1e-10) / lamb
    elif lamb == 0:
        return torch.rand(*shape).to(T) * T
    else:
        raise Exception(f'lamb must be nonnegative, got {lamb}')

def truncated_exponential_density(t, lamb, T):
    if lamb > 0:
        return lamb * torch.exp(-lamb * t) / exponential_CDF(T, lamb)
    elif lamb == 0:
        return 1 / T
    else:
        raise Exception(f'lamb must be nonnegative, got {lamb}')

def get_beta(iteration, anneal, beta_min=0.0, beta_max=1.0):
    """线性升温系数 β，常用于其它目标的退火（本训练未用）。"""
    assert anneal >= 1
    beta_range = beta_max - beta_min
    return min(beta_range * iteration / anneal + beta_min, beta_max)

class VariancePreservingTruncatedSampling:
    """在 VP-SDE 前提下构造与 r(t)=β(t)/var(t) 相关的截断时间分布，用于 debias 采样 t。

    核心对外接口：inv_Phi(u) — 均匀 u 映射到 t，使训练时更多关注某些时间段。
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20., t_epsilon=1e-3):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_max - self.beta_min) + t * self.beta_min

    def mean_weight(self, t):
        return torch.exp(-0.5 * self.integral_beta(t))

    def var(self, t):
        return 1. - torch.exp(- self.integral_beta(t))

    def std(self, t):
        return self.var(t) ** 0.5

    def g(self, t):
        beta_t = self.beta(t)
        return beta_t ** 0.5

    def r(self, t):
        """瞬时信噪比相关量 β/var，debias 密度由其构造。"""
        return self.beta(t) / self.var(t)

    def t_new(self, t):
        """将过小 t 截到 t_epsilon，避免数值奇异性。"""
        mask_le_t_eps = (t <= self.t_epsilon).float()
        t_eps = torch.tensor(float(self.t_epsilon))
        t_new = mask_le_t_eps * t_eps + (1. - mask_le_t_eps) * t
        return t_new

    def unpdf(self, t):
        t_new = self.t_new(t)
        unprob = self.r(t_new)
        return unprob

    def antiderivative(self, t):
        """用于构造 CDF 的辅助原函数。"""
        return torch.log(1. - torch.exp(- self.integral_beta(t))) + self.integral_beta(t)

    def phi_t_le_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.r(t_eps).item() * t

    def phi_t_gt_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.phi_t_le_t_eps(t_eps).item() + self.antiderivative(t) - self.antiderivative(t_eps).item()

    def normalizing_constant(self, T):
        return self.phi_t_gt_t_eps(T)

    def pdf(self, t, T):
        Z = self.normalizing_constant(T)
        prob = self.unpdf(t) / Z
        return prob

    def Phi(self, t, T):
        """截断分布的 CDF（分段公式）。"""
        Z = self.normalizing_constant(T)
        t_new = self.t_new(t)
        mask_le_t_eps = (t <= self.t_epsilon).float()
        phi = mask_le_t_eps * self.phi_t_le_t_eps(t) + (1. - mask_le_t_eps) * self.phi_t_gt_t_eps(t_new)
        return phi / Z

    def inv_Phi(self, u, T):
        """CDF 的逆：逆变换采样入口。u ~ Uniform(0,1) → t。"""
        t_eps = torch.tensor(float(self.t_epsilon))
        Z = self.normalizing_constant(T)
        r_t_eps = self.r(t_eps).item()
        antdrv_t_eps = self.antiderivative(t_eps).item()
        mask_le_u_eps = (u <= self.t_epsilon * r_t_eps / Z).float()
        a = self.beta_max - self.beta_min
        b = self.beta_min
        inv_phi = mask_le_u_eps * Z / r_t_eps * u + (1. - mask_le_u_eps) * \
                  (-b + (b ** 2 + 2. * a * torch.log(
                      1. + torch.exp(Z * u + antdrv_t_eps - r_t_eps * self.t_epsilon))) ** 0.5) / a
        return inv_phi

def sample_vp_truncated_q(shape, beta_min, beta_max, t_epsilon, T):
    """批量生成 debias 时间：与 VariancePreservingSDE.sample_debiasing_t 对接。"""
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(beta_min=beta_min, beta_max=beta_max, t_epsilon=t_epsilon)
    return vpsde.inv_Phi(u.view(-1), T).view(*shape)

import os
import datetime

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def create(*args):
    """将多级目录名拼成路径并创建（参数来自 train_mnist 的 saveroot 等）。"""
    path = '/'.join(a for a in args)
    if not os.path.isdir(path):
        os.makedirs(path)

def logging(s, path='./', filename='log.txt', print_=True, log_=True):
    """打印一行并追加到 path/filename；train_mnist 里 print_ 闭包绑定到实验目录。"""
    s = str(datetime.datetime.now()) + '\t' + str(s)
    if print_:
        print(s)
    if log_:
        assert path, 'path is not define. path: {}'.format(path)
    with open(os.path.join(path, filename), 'a+') as f_log:
        f_log.write(s + '\n')
