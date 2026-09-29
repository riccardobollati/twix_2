"""A small U-Net implemented with basic PyTorch layers (no pretrained weights)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, classes: int = 3, base_channels: int = 16):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a multiple of 8 and at least 8")
        b = base_channels
        self.encoder1 = DoubleConv(3, b)
        self.encoder2 = DoubleConv(b, b * 2)
        self.encoder3 = DoubleConv(b * 2, b * 4)
        self.encoder4 = DoubleConv(b * 4, b * 8)
        self.center = DoubleConv(b * 8, b * 16)
        self.decoder4 = DoubleConv(b * 24, b * 8)
        self.decoder3 = DoubleConv(b * 12, b * 4)
        self.decoder2 = DoubleConv(b * 6, b * 2)
        self.decoder1 = DoubleConv(b * 3, b)
        self.head = nn.Conv2d(b, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.encoder1(x)
        b = self.encoder2(F.max_pool2d(a, 2))
        c = self.encoder3(F.max_pool2d(b, 2))
        d = self.encoder4(F.max_pool2d(c, 2))
        x = self.center(F.max_pool2d(d, 2))
        for skip, decoder in ((d, self.decoder4), (c, self.decoder3),
                              (b, self.decoder2), (a, self.decoder1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
        return self.head(x)


class WindowUNet(nn.Module):
    """Full-page segmentation with a 16×16 kernel in the first convolution."""

    def __init__(self, classes: int = 3, base_channels: int = 16):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a multiple of 8 and at least 8")
        b = base_channels
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, b, kernel_size=16, stride=16, bias=False),
            nn.GroupNorm(min(8, b), b),
            nn.ReLU(inplace=True),
        )
        self.encoder1 = DoubleConv(b, b)
        self.encoder2 = DoubleConv(b, b * 2)
        self.encoder3 = DoubleConv(b * 2, b * 4)
        self.encoder4 = DoubleConv(b * 4, b * 8)
        self.center = DoubleConv(b * 8, b * 16)
        self.decoder4 = DoubleConv(b * 24, b * 8)
        self.decoder3 = DoubleConv(b * 12, b * 4)
        self.decoder2 = DoubleConv(b * 6, b * 2)
        self.decoder1 = DoubleConv(b * 3, b)
        self.head = nn.Conv2d(b, classes * 16 * 16, 1)
        self.pixel_shuffle = nn.PixelShuffle(16)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2] % 256 or image.shape[-1] % 256:
            raise ValueError("WindowUNet requires image sides divisible by 256")
        a = self.encoder1(self.patch_embed(image))
        b = self.encoder2(F.max_pool2d(a, 2))
        c = self.encoder3(F.max_pool2d(b, 2))
        d = self.encoder4(F.max_pool2d(c, 2))
        x = self.center(F.max_pool2d(d, 2))
        for skip, decoder in ((d, self.decoder4), (c, self.decoder3),
                              (b, self.decoder2), (a, self.decoder1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
        return self.pixel_shuffle(self.head(x))


def model_from_checkpoint(checkpoint: dict) -> nn.Module:
    """Load either the original U-Net or the 16×16 first-kernel variant."""
    model_class = WindowUNet if checkpoint.get("first_kernel", 3) == 16 else UNet
    model = model_class(base_channels=checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    return model
