import torch

from unet.unet import UNet


def test_unet():
    torch.random.manual_seed(19)
    x = torch.randn((8, 3, 32, 32))
    unet = UNet()
    y = unet(x)
    assert y.shape == x.shape


def test_unet_no_skip_conns():
    torch.random.manual_seed(19)
    x = torch.randn((8, 3, 32, 32))
    unet = UNet(skip_connections=False)
    y = unet(x)
    assert y.shape == x.shape
