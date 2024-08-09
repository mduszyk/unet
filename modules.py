import torch
from torch import nn
from torch.nn.functional import softmax


class Residual(nn.Module):

    def __init__(self, in_channels, out_channels=None, dropout=0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return self.residual(x) + h


class Attention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.normalize = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.Q = nn.Conv2d(channels, channels, kernel_size=1)
        self.K = nn.Conv2d(channels, channels, kernel_size=1)
        self.V = nn.Conv2d(channels, channels, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -.5

    def forward(self, x):
        h = self.normalize(x)
        q = self.Q(h)
        k = self.K(h)
        v = self.V(h)
        w = torch.einsum("bchw,bcHW->bhwHW", q, k) * self.scale
        s = w.shape
        w = w.reshape((s[0], s[1], s[2], s[3] * s[4]))
        w = softmax(w, dim=3)
        w = w.reshape(s)
        h = torch.einsum("bhwHW,bcHW->bchw", w, v)
        h = self.out(h)
        return x + h


def spatial_downsample(in_channels, out_channels=None):
    return nn.Conv2d(in_channels, out_channels or in_channels, kernel_size=3, stride=2, padding=1)


def spatial_upsample(in_channels, out_channels=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_channels, out_channels or in_channels, kernel_size=3, padding=1)
    )
