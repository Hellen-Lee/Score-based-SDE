# 主训练文件
# 流程概览：MNIST（仅数字 2）→ 用 VP-SDE 前向加噪 + UNet 估计 score/漂移 → DSM 损失训练
# → 周期性从纯噪声反向积分生成样本图；训练结束后保存最终网格图。

import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sdes import VariancePreservingSDE, PluginReverseSDE
from plotting import get_grid
from models import UNet
from utils import logging, create
from tensorboardX import SummaryWriter
import json

# 用于拼实验子目录名，便于区分不同超参组合（与 saveroot 下路径一一对应）
_folder_name_keys = ['dataset', 'real', 'debias', 'batch_size', 'lr', 'num_iterations']

def get_args():
    """解析命令行；初学者可先全用默认，只改 batch_size / num_iterations / lr。"""
    parser = argparse.ArgumentParser()

    # i/o：数据与输出路径
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--dataroot', type=str, default='~/.datasets')
    parser.add_argument('--saveroot', type=str, default='~/.saved')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='从噪声生成图像时，反向 SDE 离散步数（越大越精细、越慢）')

    # 优化与 SDE 时间范围
    parser.add_argument('--T0', type=float, default=1.0,
                        help='integration time')
    parser.add_argument('--vtype', type=str, choices=['rademacher', 'gaussian'], default='rademacher',
                        help='仅在使用 elbo 类目标时相关；本脚本主训练用 dsm，可忽略')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--num_iterations', type=int, default=10000)

    parser.add_argument('--sampling_method', type=str, choices=['euler_maruyama', 'stochastic_euler'], default='euler_maruyama',
                        help='采样方法：欧拉丸山法或随机化欧拉法')

    # 模型与损失选项
    # 注意：当前脚本未根据 real 做 logit 变换，数据仍在 [0,1]；该参数仅写入实验目录名与 args.txt
    parser.add_argument('--real', type=eval, choices=[True, False], default=True,
                        help='是否打算在 logit 空间建模（本文件主循环未启用，扩展时可接）')
    parser.add_argument('--debias', type=eval, choices=[True, False], default=False,
                        help='using non-uniform sampling to debias the denoising score matching loss')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # 创建image文件夹，保存训练过程中生成的样本拼图（与 TensorBoard 日志目录无关）
    image_folder = os.path.join(os.getcwd(), 'image')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # 初始化SDE和插件反向SDE
    # 前向过程：把真实图像逐渐加噪成近似高斯（由 beta 调度与 VP-SDE 闭式给出）
    base_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=args.T0, t_epsilon=0.001)
    input_channels = 1
    input_height = 28
    # 可学习部分：输入当前噪声图 y 与时间 t，输出向量场 a(y,t)，供反向 SDE 使用
    drift_q = UNet(
        input_channels=input_channels,
        input_height=input_height,
        ch=32,
        ch_mult=(1, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
    )
    # 把 UNet 接到 VP-SDE 上：对外提供 dsm 损失、以及采样用的 mu/sigma（见 plotting.get_grid）
    gen_sde = PluginReverseSDE(base_sde, drift_q, T=args.T0, vtype=args.vtype, debias=args.debias)

    # 数据：MNIST 单通道 28×28；此处只保留标签为 2 的样本，相当于在单类分布上训练生成模型
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=args.dataroot, train=True,
                                          download=True, transform=transform)
    train_idx = [i for i, (_, label) in enumerate(trainset) if label == 2]
    trainset = torch.utils.data.Subset(trainset, train_idx)
    # shuffle=True：每个 epoch 打乱顺序，有利于 SGD 类优化收敛
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    cuda = torch.cuda.is_available()
    if cuda:
        gen_sde.cuda()  # 将模型参数与缓冲搬到 GPU；输入 x 下面也要 .cuda()

    # Adam：自适应学习率，深度学习中常用的默认选择之一
    optim = torch.optim.Adam(gen_sde.parameters(), lr=args.lr)

    folder_tag = 'sde-flow'
    folder_name = '-'.join([str(getattr(args, k)) for k in _folder_name_keys])
    create(args.saveroot, folder_tag, args.expname, folder_name)
    folder_path = os.path.join(args.saveroot, folder_tag, args.expname, folder_name)
    print_ = lambda s: logging(s, folder_path)
    print_(f'folder path: {folder_path}')
    print_(str(args))
    with open(os.path.join(folder_path, 'args.txt'), 'w') as out:
        out.write(json.dumps(args.__dict__, indent=4))
    # TensorBoard：在终端运行 tensorboard --logdir=<folder_path> 可查看 loss 曲线
    writer = SummaryWriter(folder_path)

    count = 0
    not_finished = True
    while not_finished:
        for x, _ in trainloader:
            if cuda:
                x = x.cuda()
            # 将 [0,1] 像素微扰到 (0,1) 内部，避免数值边界；ToTensor 已把 0–255 归一化到 0–1
            x = x * 255 / 256 + torch.rand_like(x) / 256

            # DSM：去噪得分匹配；内部随机时间 t、前向加噪，再让网络预测与噪声相关的量
            loss = gen_sde.dsm(x).mean()

            # 标准三步：清空旧梯度 → 反传算梯度 → 更新参数
            optim.zero_grad()
            loss.backward()
            optim.step()

            count += 1
            if count == 1 or count % args.print_every == 0:
                writer.add_scalar('loss', loss.item(), count)
                writer.add_scalar('T', gen_sde.T.item(), count)

                print_(f'Iteration {count} \tLoss {loss.item()}')

            if count >= args.num_iterations:
                not_finished = False
                print_('Finished training')
                break

            if count % args.sample_every == 0:
                # eval：关闭 dropout 等训练期行为；推理/可视化时常用
                gen_sde.eval()
                samples = get_grid(gen_sde, input_channels, input_height, n=4,
                                   num_steps=args.num_steps, transform=None)
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(image_folder, f'samples_{count}.png'), samples[0], cmap='gray')
                gen_sde.train()  # 回到训练模式，继续下一轮迭代

            if count % args.checkpoint_every == 0:
                # 保存完整训练状态，便于中断后继续（需自行写加载逻辑）
                torch.save([gen_sde, optim, not_finished, count], os.path.join(folder_path, 'checkpoint.pt'))

    # 训练结束：从标准高斯噪声出发，沿学到的反向 SDE 积分生成样本
    gen_sde.eval()
    x = torch.randn(args.batch_size, input_channels, input_height, input_height)
    if cuda:
        x = x.cuda()
    if args.sampling_method == 'euler_maruyama':
        samples = gen_sde.sample_euler_maruyama(x, args.num_steps, args.T0)
    elif args.sampling_method == 'stochastic_euler':
        samples = gen_sde.sample_stochastic_euler(x, args.num_steps, args.T0)
    # 实际写入磁盘的网格图由 get_grid 生成（内部用 gen_sde.mu / sigma 逐步更新）
    samples = get_grid(gen_sde, input_channels, input_height, n=4,
                       num_steps=args.num_steps, transform=None)
    plt.imsave(os.path.join(image_folder, f'final_samples.png'), samples[0], cmap='gray')