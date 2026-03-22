# 模型定义
# UNet：输入「当前带噪图像 x」与「连续时间 temp」，输出与图像同形状的向量场，用作 PluginReverseSDE 中的 a(x,t)。
# 结构：编码器下采样提特征 → 瓶颈 → 解码器上采样并与 skip 拼接（经典 U-Net）。

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """时间条件的 2D U-Net，用于 MNIST 等单通道小图；temp 通常与扩散时间 t∈[0,1] 对应。"""

    def __init__(self,
                 input_channels,
                 input_height,
                 ch,
                 output_channels=None,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.,
                 resamp_with_conv=True,
                 act=nn.SiLU(),
                 num_groups=32,  # GroupNorm 分组数，需整除各层通道数
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.ch = ch
        self.output_channels = output_channels = input_channels if output_channels is None else output_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        # 仅当特征图边长属于该元组时插入 SelfAttention（如 16 表示 16×16 特征图上做注意力）
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.act = act
        self.num_groups = num_groups

        self.num_resolutions = num_resolutions = len(ch_mult)
        in_ht = input_height
        in_ch = input_channels
        temb_ch = ch * 4
        # 多次 /2 下采样后高宽仍为整数
        assert in_ht % 2 ** (num_resolutions - 1) == 0, "input_height doesn't satisfy the condition"

        self.temb_net = TimestepEmbedding(
            embedding_dim=ch,
            hidden_dim=temb_ch,
            output_dim=temb_ch,
            act=act,
        )

        self.begin_conv = nn.Conv2d(in_ch, ch, kernel_size=3, padding=1)
        unet_chs = [ch]  # 记录各尺度通道，供解码阶段 concat skip
        in_ht = in_ht
        in_ch = ch
        down_modules = []
        for i_level in range(num_resolutions):
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules[f'{i_level}a_{i_block}a_block'] = \
                    ResidualBlock(
                        in_ch=in_ch,
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        num_groups=num_groups,
                    )
                if in_ht in attn_resolutions:
                    block_modules[f'{i_level}a_{i_block}b_attn'] = SelfAttention(out_ch, num_groups=num_groups)
                unet_chs += [out_ch]
                in_ch = out_ch
            if i_level != num_resolutions - 1:
                block_modules[f'{i_level}b_downsample'] = downsample(out_ch, with_conv=resamp_with_conv)
                in_ht //= 2
                unet_chs += [out_ch]
            down_modules += [nn.ModuleDict(block_modules)]
        self.down_modules = nn.ModuleList(down_modules)

        mid_modules = []
        mid_modules += [
            ResidualBlock(in_ch, temb_ch=temb_ch, out_ch=in_ch, dropout=dropout, act=act, num_groups=num_groups)]
        mid_modules += [SelfAttention(in_ch, num_groups=num_groups)]
        mid_modules += [
            ResidualBlock(in_ch, temb_ch=temb_ch, out_ch=in_ch, dropout=dropout, act=act, num_groups=num_groups)]
        self.mid_modules = nn.ModuleList(mid_modules)

        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block_modules[f'{i_level}a_{i_block}a_block'] = \
                    ResidualBlock(
                        in_ch=in_ch + unet_chs.pop(),
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        num_groups=num_groups,
                    )
                if in_ht in attn_resolutions:
                    block_modules[f'{i_level}a_{i_block}b_attn'] = SelfAttention(out_ch, num_groups=num_groups)
                in_ch = out_ch
            if i_level != 0:
                block_modules[f'{i_level}b_upsample'] = upsample(out_ch, with_conv=resamp_with_conv)
                in_ht *= 2
            up_modules += [nn.ModuleDict(block_modules)]
        self.up_modules = nn.ModuleList(up_modules)
        assert not unet_chs

        # 修改：使用 num_groups 创建归一化层
        self.end_conv = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            self.act,
            nn.Conv2d(in_ch, output_channels, kernel_size=3, padding=1, bias=False),
        )

    def _compute_cond_module(self, module, x, temp):
        """瓶颈处若干子模块顺序前向（均吃 temb）。"""
        for m in module:
            x = m(x, temp)
        return x

    def forward(self, x, temp):
        B, C, H, W = x.size()

        # 时间条件：每条样本一个标量，扩展成 (B,) 与 batch 对齐
        if temp.dim() == 0:
            temp = temp.expand(B)
        elif temp.dim() == 1 and temp.size(0) == 1:
            temp = temp.expand(B)

        # 动态推断设备，使用第一个参数的设备
        if self.parameters():
            device = next(self.parameters()).device
            temp = temp.to(device)
        
        temb = self.temb_net(temp)
        assert list(temb.shape) == [B, self.ch * 4]

        hs = [self.begin_conv(x)]
        # 编码：逐层提特征并下采样；hs 供解码跳连
        for i_level in range(self.num_resolutions):
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block = block_modules[f'{i_level}a_{i_block}a_block']
                h = resnet_block(hs[-1], temb)
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules[f'{i_level}a_{i_block}b_attn']
                    h = attn_block(h, temb)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                downsample = block_modules[f'{i_level}b_downsample']
                hs.append(downsample(hs[-1]))

        h = hs[-1]
        h = self._compute_cond_module(self.mid_modules, h, temb)

        # 解码：与编码对称，上采样并与 pop 出的 skip 拼接
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                resnet_block = block_modules[f'{i_level}a_{i_block}a_block']
                h = resnet_block(torch.cat([h, hs.pop()], dim=1), temb)
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules[f'{i_level}a_{i_block}b_attn']
                    h = attn_block(h, temb)
            if i_level != 0:
                upsample = block_modules[f'{i_level}b_upsample']
                h = upsample(h)
        assert not hs

        h = self.end_conv(h)
        assert list(h.size()) == [x.size(0), self.output_channels, x.size(2), x.size(3)]
        return h

class TimestepEmbedding(nn.Module):
    """将标量时间编码为向量，注入各 ResidualBlock（与扩散模型里「时间嵌入」作用相同）。"""

    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.SiLU()):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = act
        
        # 用固定正弦表初始化 Embedding，类似 Transformer 位置编码
        self.time_embedding = nn.Embedding(1000, embedding_dim)
        self.linear_1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        
        # 初始化时间嵌入层的权重为正弦编码
        self._init_time_embedding()
    
    def _init_time_embedding(self):
        # 创建正弦位置编码
        position = torch.arange(1000).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-torch.log(torch.tensor(10000.0)) / self.embedding_dim))
        pe = torch.zeros(1000, self.embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.time_embedding.weight.data.copy_(pe)
        # 冻结时间嵌入层的权重
        self.time_embedding.weight.requires_grad = False
    
    def forward(self, temp):
        # 将时间标量转换为整数索引
        # 假设 temp 在 [0, 1] 范围内，将其映射到 [0, 999]
        temp = (temp * 999).long()
        
        # 确保 temp 在与模型相同的设备上
        temp = temp.to(self.time_embedding.weight.device)
        
        # 获取时间嵌入
        temp = self.time_embedding(temp)
        temp = self.linear_1(temp)
        temp = self.act(temp)
        temp = self.linear_2(temp)
        return temp

class ResidualBlock(nn.Module):
    """Conv + GroupNorm 残差块；时间嵌入 temb 经线性层后加到中间特征（DDPM/ADM 类结构常用）。"""

    def __init__(self, in_ch, temb_ch, out_ch, dropout, act, num_groups):
        super().__init__()
        self.in_ch = in_ch
        self.temb_ch = temb_ch
        self.out_ch = out_ch
        self.dropout = dropout
        self.act = act

        # 修改：动态创建归一化层，使其通道数与输入匹配
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)

        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, temb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # 时间嵌入按通道加到特征图上（广播）
        h += self.temb_proj(self.act(temb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = nn.Dropout(self.dropout)(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)

class SelfAttention(nn.Module):
    """空间维自注意力：在 H×W 上建立长程依赖；通道压缩为 Q/K 降计算量。"""

    def __init__(self, in_channels, num_groups):
        super().__init__()
        self.in_channels = in_channels
        
        # 修改：动态创建归一化层，使其通道数与输入匹配
        self.normalize = nn.GroupNorm(num_groups, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, temb=None):
        # temb 仅为与 ResidualBlock 调用签名一致，此处未使用
        h_ = self.normalize(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        # scaled dot-product attention
        attn = torch.bmm(q, k) * (int(c) ** (-0.5))
        attn = nn.Softmax(dim=2)(attn)

        v = v.reshape(b, -1, h * w)
        attn = attn.permute(0, 2, 1)
        a = torch.bmm(v, attn)
        a = a.reshape(b, -1, h, w)

        return x + self.proj_out(a)

def downsample(in_channels, with_conv):
    """空间尺寸减半：卷积 stride=2 或平均池化。"""
    if with_conv:
        return nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
    else:
        return nn.AvgPool2d(2)

def upsample(in_channels, with_conv):
    """空间尺寸翻倍：转置卷积或最近邻上采样。"""
    if with_conv:
        return nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
    else:
        return nn.Upsample(scale_factor=2, mode='nearest')