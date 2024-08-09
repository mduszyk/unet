from unet.modules import *


def test_attention():
    torch.random.manual_seed(19)
    x = torch.randn((8, 64, 128, 128))
    m = Attention(channels=64)
    y = m(x)
    assert y.shape == x.shape


def test_residual():
    torch.random.manual_seed(19)
    x = torch.randn((8, 64, 128, 128))
    residual_module = Residual(in_channels=64)
    y = residual_module(x)
    assert y.shape == (8, 64, 128, 128)
