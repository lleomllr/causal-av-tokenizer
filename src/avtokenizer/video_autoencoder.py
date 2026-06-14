from __future__ import annotations
import torch
from torch import nn


class VideoAutoencoder(nn.Module):
    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, latent_channels),
        )
        self.decoder = nn.Sequential(
            UpBlock(latent_channels, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        latents = self.encoder(frames)
        return self.decoder(latents)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )


class UpBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )


def psnr_from_l1(l1_loss: torch.Tensor) -> torch.Tensor:
    mse_proxy = l1_loss.clamp_min(1e-8) ** 2
    return -10.0 * torch.log10(mse_proxy)
