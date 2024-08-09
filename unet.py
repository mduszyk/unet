import torch
from torch import nn

from unet.modules import Residual, Attention, spatial_downsample, spatial_upsample


class UNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 channels=128,
                 channels_mult=(1, 2, 2, 2),
                 resnet_blocks=2,
                 attention_steps=(1,),
                 dropout=0,
                 skip_connections=True):
        super().__init__()

        self.skip_connections = skip_connections

        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        if self.skip_connections:
            down_outputs = []
        in_channels = channels
        num_resolutions = len(channels_mult)
        for i in range(num_resolutions):
            out_channels = channels * channels_mult[i]
            for _ in range(resnet_blocks):
                residual = Residual(in_channels, out_channels, dropout)
                if self.skip_connections:
                    down_outputs.append(out_channels)
                in_channels = out_channels
                self.down.append(residual)
                if i in attention_steps:
                    self.down.append(Attention(out_channels))
            if i != num_resolutions - 1:
                self.down.append(spatial_downsample(in_channels=out_channels, out_channels=out_channels))

        self.middle = nn.ModuleList([
            Residual(in_channels, out_channels=in_channels, dropout=dropout),
            Attention(in_channels),
            Residual(in_channels, out_channels=in_channels, dropout=dropout),
        ])

        self.up = nn.ModuleList()
        for i in reversed(range(num_resolutions)):
            out_channels = channels * channels_mult[i]
            for _ in range(resnet_blocks):
                down_channels = down_outputs.pop() if self.skip_connections else 0
                residual = Residual(in_channels + down_channels, out_channels, dropout)
                in_channels = out_channels
                self.up.append(residual)
                if i in attention_steps:
                    self.up.append(Attention(out_channels))
            if i != 0:
                self.up.append(spatial_upsample(in_channels=out_channels, out_channels=out_channels))

        assert len(self.down) == len(self.up)

        self.end = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)

        if self.skip_connections:
            down_outputs = []
        for module in self.down:
            x = module(x)
            if self.skip_connections and isinstance(module, Residual):
                down_outputs.append(x)

        for module in self.middle:
            x = module(x)

        for module in self.up:
            if self.skip_connections and isinstance(module, Residual):
                x = torch.concat(tensors=(x, down_outputs.pop()), dim=1)
            x = module(x)

        x = self.end(x)

        return x
