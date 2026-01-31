# 主训练文件

import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sdes import VariancePreservingSDE, PluginReverseSDE
from plotting import get_grid
from models import UNet
from utils import logging, create
from tensorboardX import SummaryWriter
import json

_folder_name_keys = ['dataset', 'real', 'debias', 'batch_size', 'lr', 'num_iterations']

def get_args():
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--dataroot', type=str, default='~/.datasets')
    parser.add_argument('--saveroot', type=str, default='~/.saved')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='number of integration steps for sampling')

    # optimization
    parser.add_argument('--T0', type=float, default=1.0,
                        help='integration time')
    parser.add_argument('--vtype', type=str, choices=['rademacher', 'gaussian'], default='rademacher',
                        help='random vector for the Hutchinson trace estimator')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--num_iterations', type=int, default=10000)

    parser.add_argument('--sampling_method', type=str, choices=['euler_maruyama', 'stochastic_euler'], default='euler_maruyama',
                        help='采样方法：欧拉丸山法或随机化欧拉法')

    # model
    parser.add_argument('--real', type=eval, choices=[True, False], default=True,
                        help='transforming the data from [0,1] to the real space using the logit function')
    parser.add_argument('--debias', type=eval, choices=[True, False], default=False,
                        help='using non-uniform sampling to debias the denoising score matching loss')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # 创建image文件夹
    image_folder = os.path.join(os.getcwd(), 'image')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # 初始化SDE和插件反向SDE
    base_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=args.T0, t_epsilon=0.001)
    input_channels = 1
    input_height = 28
    drift_q = UNet(
        input_channels=input_channels,
        input_height=input_height,
        ch=32,
        ch_mult=(1, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
    )
    gen_sde = PluginReverseSDE(base_sde, drift_q, T=args.T0, vtype=args.vtype, debias=args.debias)

    # 加载MNIST数据集，只选择数字2
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=args.dataroot, train=True,
                                          download=True, transform=transform)
    train_idx = [i for i, (_, label) in enumerate(trainset) if label == 2]
    trainset = torch.utils.data.Subset(trainset, train_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    cuda = torch.cuda.is_available()
    if cuda:
        gen_sde.cuda()

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
    writer = SummaryWriter(folder_path)

    count = 0
    not_finished = True
    while not_finished:
        for x, _ in trainloader:
            if cuda:
                x = x.cuda()
            x = x * 255 / 256 + torch.rand_like(x) / 256

            loss = gen_sde.dsm(x).mean()

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
                gen_sde.eval()
                samples = get_grid(gen_sde, input_channels, input_height, n=4,
                                   num_steps=args.num_steps, transform=None)
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(image_folder, f'samples_{count}.png'), samples[0], cmap='gray')
                gen_sde.train()

            if count % args.checkpoint_every == 0:
                torch.save([gen_sde, optim, not_finished, count], os.path.join(folder_path, 'checkpoint.pt'))

    # 最终采样
    gen_sde.eval()
    x = torch.randn(args.batch_size, input_channels, input_height, input_height)
    if cuda:
        x = x.cuda()
    if args.sampling_method == 'euler_maruyama':
        samples = gen_sde.sample_euler_maruyama(x, args.num_steps, args.T0)
    elif args.sampling_method == 'stochastic_euler':
        samples = gen_sde.sample_stochastic_euler(x, args.num_steps, args.T0)
    samples = get_grid(gen_sde, input_channels, input_height, n=4,
                       num_steps=args.num_steps, transform=None)
    plt.imsave(os.path.join(image_folder, f'final_samples.png'), samples[0], cmap='gray')