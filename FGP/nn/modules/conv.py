# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "DP_CAF",
)




class DP_CAF(nn.Module):
    def __init__(self, c1, c2=None, c_out=None, *args, **kwargs):
        """
        DP-CAF (Improved BK-CCBCM): 改进版双边卡尔曼去中心化协方差模块
        1. 解决 Q 和 R 标度不匹配问题（统一归一化至 [0, 1]）。
        2. 引入 L2 归一化，彻底保障 FP16 半精度训练数值稳定性。
        3. 引入空间与通道双重卡尔曼调控机制。
        """
        super().__init__()

        # 我们兼容 YOLO parse_model 参数解析
        if isinstance(c1, (list, tuple)):
            c_out = c2 if c2 is not None else (c1[0] + c1[1])
            c1, c2 = c1[0], c1[1]
        elif c_out is None:
            c_out = c1 + c2

        self.c_mid = c_out // 2

        # 基础通道投影层
        self.proj_s = nn.Conv2d(c1, self.c_mid, 1, bias=False)
        self.proj_d = nn.Conv2d(c2, self.c_mid, 1, bias=False)
        self.bn_s = nn.BatchNorm2d(self.c_mid)
        self.bn_d = nn.BatchNorm2d(self.c_mid)
        self.act = nn.SiLU()

        # 浅层高频边缘提取算子
        self.high_pass = nn.Sequential(
            nn.Conv2d(self.c_mid, self.c_mid, 3, padding=1, groups=self.c_mid, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU()
        )

        # Q (过程噪声) 标度映射层
        self.q_gate = nn.Sequential(
            nn.Conv2d(self.c_mid, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # R (测量噪声) 生成层
        self.r_gate = nn.Sequential(
            nn.Conv2d(self.c_mid, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # 通道卡尔曼增益生成层
        self.channel_k = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c_mid, self.c_mid, 1, bias=False),
            nn.Sigmoid()
        )

        # 可学习温度参数
        self.tau = nn.Parameter(torch.ones(1))

        # 后置融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.c_mid * 2, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        x_shallow, x_deep = x

        # 我们自动对齐深浅特征图的空间尺寸
        if x_deep.shape[2:] != x_shallow.shape[2:]:
            x_deep_interp = F.interpolate(x_deep.float(), size=x_shallow.shape[2:], mode='bilinear',
                                          align_corners=False).to(x_deep.dtype)
        else:
            x_deep_interp = x_deep

        # 主轨道：我们保留物理无损 Concat 拼接
        identity_concat = torch.cat([x_shallow, x_deep_interp], dim=1)

        # 旁路：我们映射基础通道
        fs = self.act(self.bn_s(self.proj_s(x_shallow)))
        fd = self.act(self.bn_d(self.proj_d(x_deep_interp)))

        # 步骤 1：浅层提取高频细节残差
        fs_high = self.high_pass(fs) - fs

        # 步骤 2：我们计算标度对齐的卡尔曼噪声
        # 过程噪声 Q：结合局部方差与卷积响应，归一化到 [0, 1]
        fs_local_mean = F.avg_pool2d(fs_high, 3, stride=1, padding=1)
        var_q = torch.mean((fs_high - fs_local_mean) ** 2, dim=1, keepdim=True)
        Q = self.q_gate(fs_high) * torch.sigmoid(var_q)

        # 测量噪声 R：深层背景不确定度，控制在 [0, 1]
        R = 1.0 - self.r_gate(fd)

        # 计算空间卡尔曼增益 K_spatial [B, 1, H, W]
        k_spatial = Q / (Q + R + 1e-6)

        # 计算通道卡尔曼增益 K_channel [B, C, 1, 1]
        k_channel = self.channel_k(fs_high)

        # 步骤 3：我们执行带 FP16 保护的去中心化协方差计算
        b, c, h, w = fs.shape
        n = h * w

        fs_flat = fs_high.view(b, c, n)
        fd_flat = fd.view(b, c, n)

        # 空间去中心化
        fs_centered = fs_flat - fs_flat.mean(dim=2, keepdim=True)
        fd_centered = fd_flat - fd_flat.mean(dim=2, keepdim=True)

        # L2 归一化：防止半精度矩阵乘法溢出
        fs_norm = F.normalize(fs_centered, p=2, dim=2, eps=1e-8)
        fd_norm = F.normalize(fd_centered, p=2, dim=2, eps=1e-8)

        # 计算协方差与投影矩阵
        cov_matrix = torch.bmm(fs_norm, fd_norm.transpose(1, 2))
        safe_tau = torch.clamp(self.tau, min=0.01)
        proj_matrix = F.softmax(cov_matrix / safe_tau, dim=-1)

        # 重投影特征
        fs_aligned_flat = torch.bmm(proj_matrix, fd_flat)
        fs_aligned = fs_aligned_flat.view(b, c, h, w)

        # 步骤 4：双重卡尔曼状态更新
        k_dual = k_spatial * k_channel
        fs_updated = fs_high + k_dual * (fs_aligned - fs_high)

        # 步骤 5：我们融合更新后的特征并输出
        fused_side = torch.cat([fs_updated, fd], dim=1)
        res_patch = self.act(self.fusion_conv(fused_side))

        return identity_concat + res_patch







def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))




class ParallelWaveletConvDownsample(nn.Module):
    """
    全新设计的动态多统计量小波下采样模块 (PWCD-Module)

    1. 路径一：标准卷积下采样路径，负责保留宏观轮廓。
    2. 路径二：小波绝对值高频直接注入路径，负责挽救微弱边缘。
    3. 特征聚合：融合包含 mean, max, min, sum 的 4 维特征注意力，专门针对发丝裂纹的极值进行空间锁定。
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p: int = None, g: int = 1, d: int = 1, act=True):
        """初始化并行下采样模块"""
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s

        # 1. 如果下采样步长不等于 2，我们让模块直接退化为普通卷积
        if s != 2:
            self.conv_path = Conv(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act)
            return

        # 路径一：标准卷积下采样（直接输出完整的 c2 通道）
        self.conv_path = Conv(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act)

        # 路径二：原生小波分解核
        w = torch.ones(4, 1, 2, 2)
        w[1, 0, 0, 1] = -1;
        w[1, 0, 1, 1] = -1  # 水平方向高频
        w[2, 0, 1, 0] = -1;
        w[2, 0, 1, 1] = -1  # 垂直方向高频
        w[3, 0, 1, 0] = -1;
        w[3, 0, 0, 1] = -1  # 对角线方向高频

        haar_weights = torch.cat([w] * c1, dim=0)
        self.register_buffer('filters', haar_weights)

        # 小波路径转换层（同样输出完整的 c2 通道）
        self.wavelet_proj = Conv(c1, c2, k=1, s=1, p=0, g=1, d=1, act=act)

        # 4类数值计算空间注意力层（参考 CVPR 2025 CASAB 核心创新点）
        # 接收拼接后的特征，通过计算 4 个方向的统计极值生成空间注意力
        self.casab_spatial = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False),
            nn.SiLU(),
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 最终通道整合提炼层（将 2*c2 还原回 c2）
        self.final_merge = Conv(c2 * 2, c2, k=1, s=1, p=0, g=1, d=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果步长不是 2，我们直接返回普通卷积的结果
        if self.s != 2:
            return self.conv_path(x)

        # 1. 运行第一路：标准卷积下采样
        out_conv = self.conv_path(x)

        # 2. 对图像进行长宽的奇数填充保护
        b, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode='reflect')

        # 3. 运行第二路：原生无损小波特征抽取
        haar_out = F.conv2d(x, self.filters, bias=None, stride=2, groups=self.c1) / 4.0
        haar_out = haar_out.view(b, self.c1, 4, haar_out.shape[2], haar_out.shape[3])

        ll = haar_out[:, :, 0, :, :]  # 低频形状
        lh = haar_out[:, :, 1, :, :]  # 水平细节
        hl = haar_out[:, :, 2, :, :]  # 垂直细节
        hh = haar_out[:, :, 3, :, :]  # 对角细节

        # 核心改进一：我们计算高频特征的绝对值。这个操作能将边缘的突变特征全部变为正数。
        high_freq = torch.abs(lh) + torch.abs(hl) + torch.abs(hh)

        # 我们把高频边缘直接加到低频形状上。这一步能在下采样时强行锁死裂纹的能量。
        wavelet_feat = ll + high_freq
        out_wavelet = self.wavelet_proj(wavelet_feat)

        # 4. 核心改进二：我们将常规特征与小波补充特征进行横向拼接（总通道数变为 2 * c2）
        total_fused = torch.cat([out_conv, out_wavelet], dim=1)

        # 5. 核心改进三：运行 CASAB 多统计量计算（融合平均值、最大值、最小值、总和）
        # 因为发丝裂纹通常很暗，所以这里的 min_out 能够精准抓取裂纹的细微轮廓
        mean_out = torch.mean(total_fused, dim=1, keepdim=True)
        max_out, _ = torch.max(total_fused, dim=1, keepdim=True)
        min_out, _ = torch.min(total_fused, dim=1, keepdim=True)
        sum_out = torch.sum(total_fused, dim=1, keepdim=True)

        # 拼接 4 种数值指标，通过 7x7 大核生成没有信息断裂的空间掩码
        stats_pool = torch.cat([mean_out, max_out, min_out, sum_out], dim=1)
        spatial_weight = self.casab_spatial(stats_pool)

        # 6. 注意力增强并融合通道
        enhanced_feat = total_fused * spatial_weight
        return self.final_merge(enhanced_feat)























class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)












class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
