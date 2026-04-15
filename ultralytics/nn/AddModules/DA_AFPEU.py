import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f, C3, Conv

__all__ = ["DA_AFPEU", "C3k2_DA_AFPEU", "C2f_DA_AFPEUs", "GFFP"]


# -------------------------
# 1) 安全的注意力模块：绝不改变通道数
# -------------------------
class ChannelAttention(nn.Module):
    """Channel attention: returns x * w, w in [B,C,1,1]."""
    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        hidden = max(c // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)  # [B,C,1,1]
        w = self.mlp(w)                  # [B,C,1,1]
        return x * w


class PixelAttention(nn.Module):
    """Pixel attention: returns x * w, w in [B,C,H,W]."""
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv(x)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention: returns x * w, w in [B,1,H,W] then broadcast to channels."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)          # [B,1,H,W]
        max_pool = torch.max(x, dim=1, keepdim=True)[0]        # [B,1,H,W]
        w = self.conv(torch.cat([avg_pool, max_pool], dim=1))  # [B,1,H,W]
        return x * w


# -------------------------
# 2) FCPSModule：CA+PA+SA + 局部卷积融合（GN 替代 BN，稳定 batch=1）
# -------------------------
class FCPSModule(nn.Module):
    """
    Feature Channel-Pixel-Spatial module
    - 不改变通道数
    - 用 GroupNorm 避免 batch=1 BN 问题
    """
    def __init__(self, in_channels: int, gn_groups: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.pa = PixelAttention(in_channels)
        self.sa = SpatialAttention()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        g = min(gn_groups, in_channels)
        while g > 1 and in_channels % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.pa(x)
        x = self.sa(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# -------------------------
# 3) 你的 GFFP：保证输入输出通道一致
# -------------------------
class GFFP(nn.Module):
    """
    Global Feature Fusion Processing
    - 输入输出 shape 完全一致
    - 内部两次 FCPS 增强 + 轻量融合
    """
    def __init__(self, channels: int):
        super().__init__()
        self.pre = Conv(channels, channels, k=3, s=1, p=1)
        self.fcps1 = FCPSModule(channels)
        self.fcps2 = FCPSModule(channels)

        # 融合：把 x1,x2 拼接后压回 channels
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),  # 类似 LayerNorm 效果（对 batch 也稳）
            nn.ReLU(inplace=True),
        )
        self.post = Conv(channels, channels, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.pre(x)
        x1 = self.fcps1(x)      # [B,C,H,W]
        x2 = self.fcps2(x1)     # [B,C,H,W]
        x = self.fuse(torch.cat([x1, x2], dim=1))  # [B,C,H,W]
        x = self.post(x)
        return r + x


# -------------------------
# 4) DA-AFPEU：深度敏感 + 轴向非对称池化注意力
# -------------------------
class DA_AFPEU(nn.Module):
    """
    Depth-Aware Asymmetric Feature Pooling Enhancement Unit
    For GPR B-scan: H=depth axis, W=scanline axis.
    """
    def __init__(self, channels: int, attn_reduction: int = 1):
        super().__init__()
        hidden = max(channels // attn_reduction, 1)

        # W轴池化(保留H): concat(avg,max)-> 2C, 1x1 conv -> C
        self.dir_w = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # H轴池化(保留W)
        self.dir_h = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # 深度敏感门控：用深度相关描述 -> 通道门控 [B,C,1,1]
        self.depth_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # 可选：轻量局部增强（更利于双曲线顶点/边界）
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x  # residual

        # 1) 非对称轴向池化
        # along W -> [B,C,H,1]
        avg_w = torch.mean(x, dim=3, keepdim=True)
        max_w = torch.max(x, dim=3, keepdim=True)[0]
        pw = torch.cat([avg_w, max_w], dim=1)  # [B,2C,H,1]

        # along H -> [B,C,1,W]
        avg_h = torch.mean(x, dim=2, keepdim=True)
        max_h = torch.max(x, dim=2, keepdim=True)[0]
        ph = torch.cat([avg_h, max_h], dim=1)  # [B,2C,1,W]

        # 2) 方向注意力
        aw = self.dir_w(pw)  # [B,C,H,1]
        ah = self.dir_h(ph)  # [B,C,1,W]

        # 3) 深度敏感门控（由深度相关统计得到）
        depth_desc = F.adaptive_avg_pool2d(avg_w, 1)  # [B,C,1,1]
        g = self.depth_gate(depth_desc)               # [B,C,1,1]

        # 4) 融合 + 残差
        y = x * aw * ah * g
        y = self.pw(self.dw(y))  # 局部几何强化
        return r + y


# -------------------------
# 5) 给 YOLO11 YAML 用的封装模块（保持你原来 C3k2_* 的风格）
# -------------------------
class C3k_DA_AFPEU(C3):
    """C3 variant using DA_AFPEU blocks."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(DA_AFPEU(c_) for _ in range(n)))


class C3k2_DA_AFPEU(C2f):
    """
    C2f-style faster CSP bottleneck with DA-AFPEU.

    YAML args 对齐：
      - [-1, repeats, C3k2_DA_AFPEU, [c2, c3k, e]]
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_DA_AFPEU(self.c, self.c, 2, shortcut, g, e=1.0) if c3k else DA_AFPEU(self.c)
            for _ in range(n)
        )


class C2f_DA_AFPEUs(nn.Module):
    """Standalone C2f-like block using DA_AFPEU (和你之前 C2f_FPEUs 同用法)."""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DA_AFPEU(self.c) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))