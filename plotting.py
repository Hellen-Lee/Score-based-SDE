# 可视化：从纯噪声出发，按「学到的反向 SDE」做欧拉离散，拼成网格图。
# 要求传入的 sde 实现 mu(t,y)、sigma(t,y) 与属性 T（本项目中为 PluginReverseSDE）。

import torch

def get_grid(sde, input_channels, input_height, n=4, num_steps=20, transform=None,
             mean=0, std=1, clip=True, device='cuda'):
    """生成 n×n 张样本拼成的一张大图（numpy，形状约为 [C, n*H, n*W]）。

    步骤：从 N(mean, std²) 采样初始噪声 → 对 i=0..num_steps-1 做
    y ← y + Δt·μ(t_i,y) + √Δt·σ(t_i,y)·ξ（与训练时 DSM 对应的反向动力学）。

    无 CUDA 时请传 device='cpu'，并与模型所在设备一致。
    """
    num_samples = n ** 2
    delta = sde.T / num_steps
    
    # 将初始噪声移到指定设备
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(device)
    y0 = y0 * std + mean
    # 将时间步移到指定设备
    # ts[i] 从 0 走到 T；与 sde.mu 内部使用的 T−t 一起决定网络看到的时间索引
    ts = torch.linspace(0, 1, num_steps + 1).to(device) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(device)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

    if transform is not None:
        y0 = transform(y0)

    if clip:
        y0 = torch.clip(y0, 0, 1)

    y0 = y0.view(
        n, n, input_channels, input_height, input_height).permute(
        2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)

    # 将最终结果移回CPU并转换为numpy数组
    y0 = y0.cpu().numpy()
    return y0
