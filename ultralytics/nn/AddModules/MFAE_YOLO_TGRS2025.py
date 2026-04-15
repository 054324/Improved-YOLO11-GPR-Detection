import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f, C3, Conv

__all__ = ['GFFP', 'C3k2_FPEU', 'C2f_FPEUs']


class FPEU(nn.Module):
    def __init__(self, channels):
        super(FPEU, self).__init__()

        # 定义CBR层 (Conv -> BatchNorm -> ReLU)
        self.cbr = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # 定义Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_ = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # CBR层处理
        x = self.cbr(x)
        # 激活函数
        x = self.sigmoid(x) * x_
        return x


class DSC3k_FPEU(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                FPEU(
                    c_
                )
                for _ in range(n)
            )
        )


class C3k_FPEU(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, HW=40, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FPEU(c_) for _ in range(n)))


class C3k2_FPEU(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, HW=40, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_FPEU(self.c, self.c, 2, HW, shortcut, g) if c3k else FPEU(self.c) for _ in range(n)
        )


class C2f_FPEUs(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(FPEU(self.c) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# 定义通道注意力模块 (Channel Attention)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)  # Global Average Pooling
        avg_pool = self.conv1(avg_pool)
        return x * self.sigmoid(avg_pool)


# 定义像素注意力模块 (Pixel Attention)
class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv1(x))


# 定义空间注意力模块 (Spatial Attention)
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # MaxPooling + AveragePooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # 最大池化
        concat = torch.cat([avg_pool, max_pool], dim=1)  # 拼接
        return x * self.sigmoid(self.conv1(concat))


# 定义FCPS模块（包含CA, PA, SA机制）
class FCPSModule(nn.Module):
    def __init__(self, in_channels):
        super(FCPSModule, self).__init__()
        # 通道注意力
        self.channel_attention = ChannelAttention(in_channels)
        # 像素注意力
        self.pixel_attention = PixelAttention(in_channels)
        # 空间注意力
        self.spatial_attention = SpatialAttention(in_channels)
        # 用于进一步处理特征的卷积层
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 先通过通道注意力
        x = self.channel_attention(x)
        # 再通过像素注意力
        x = self.pixel_attention(x)
        # 再通过空间注意力
        x = self.spatial_attention(x)
        # 最后通过卷积层进行特征融合
        return self.conv(x)


# 定义通道注意力模块 (Channel Attention)
class ChannelAttention1(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention1, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 2)
        self.conv2 = Conv(in_channels // 2, in_channels // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 处理 batch_size=1 的情况
        if self.training and x.shape[0] == 1:
            # 临时将所有 BatchNorm 设置为 eval 模式
            self._set_bn_eval(True)
            result = self._forward_impl(x)
            self._set_bn_eval(False)
            return result
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x):
        """实际的前向传播实现"""
        avg_pool = F.adaptive_avg_pool2d(x, 1)  # Global Average Pooling
        avg_pool = self.conv1(avg_pool)
        avg_pool = self.conv2(avg_pool)
        return self.sigmoid(avg_pool)

    def _set_bn_eval(self, eval_mode=True):
        """设置所有 BatchNorm 模块的模式"""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if eval_mode:
                    module.eval()
                else:
                    module.train()


# 定义GFFP模块（Global Feature Fusion Processing）
class GFFP(nn.Module):
    def __init__(self, channels):
        super(GFFP, self).__init__()
        # 定义卷积层
        self.conv = Conv(channels, channels, k=3, s=1, p=1)
        # 定义FCPS模块
        self.fcps1 = FCPSModule(channels)
        self.fcps2 = FCPSModule(channels)
        self.CA = ChannelAttention1(channels * 2)
        self.PA = PixelAttention(channels)

    def forward(self, x):
        F1 = x  # 用于最后残差连接
        # 通过第一个卷积层
        x = self.conv(x)
        # 通过FCPS模块增强特征
        x1 = self.fcps1(x)
        x2 = self.fcps2(x1)
        x3 = torch.cat([x1, x2], dim=1)
        Wca = self.CA(x3)
        F2 = Wca * x1 + Wca * x2
        # 通过卷积层进行最后的特征融合
        x = self.conv(self.PA(F2))
        return F1 + x

