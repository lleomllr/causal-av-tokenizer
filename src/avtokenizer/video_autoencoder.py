from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class VideoAutoencoder(nn.Module):
    def __init__(
        self,
        latent_channels: int = 128,
        bottleneck: str = "ae",
        fsq_levels: tuple[int, ...] = (8, 8, 8, 8, 8, 8),
    ) -> None:
        super().__init__()
        if bottleneck not in {"ae", "vae", "fsq"}:
            raise ValueError(f"Unsupported bottleneck: {bottleneck}")

        self.bottleneck = bottleneck
        self.latent_channels = len(fsq_levels) if bottleneck == "fsq" else latent_channels
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        if bottleneck == "vae":
            self.to_mu = nn.Conv2d(128, self.latent_channels, kernel_size=1)
            self.to_logvar = nn.Conv2d(128, self.latent_channels, kernel_size=1)
        else:
            self.to_latent = nn.Conv2d(128, self.latent_channels, kernel_size=1)

        self.quantizer = FSQ(fsq_levels) if bottleneck == "fsq" else None
        self.decoder = nn.Sequential(
            UpBlock(self.latent_channels, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.encoder(frames)
        aux: dict[str, torch.Tensor] = {}

        if self.bottleneck == "vae":
            mu = self.to_mu(features)
            logvar = self.to_logvar(features).clamp(-10.0, 10.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                latents = mu + std * torch.randn_like(std)
            else:
                latents = mu
            aux["mu"] = mu
            aux["logvar"] = logvar
            aux["kl_loss"] = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).mean()
            return latents, aux

        latents = self.to_latent(features)
        if self.quantizer is not None:
            latents, indices = self.quantizer(latents)
            aux["code_indices"] = indices
        aux["kl_loss"] = latents.new_tensor(0.0)
        return latents, aux

    def forward(
        self,
        frames: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latents, aux = self.encode(frames)
        recon = self.decoder(latents)
        if return_aux:
            aux["latents"] = latents
            return recon, aux
        return recon


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
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )


class FSQ(nn.Module):
    """Finite scalar quantization with a straight-through estimator."""

    def __init__(self, levels: tuple[int, ...]) -> None:
        super().__init__()
        if not levels:
            raise ValueError("FSQ levels cannot be empty")
        if any(level < 2 for level in levels):
            raise ValueError("Every FSQ level must be >= 2")
        self.levels = tuple(int(level) for level in levels)
        self.register_buffer("levels_tensor", torch.tensor(self.levels, dtype=torch.float32))
        basis = []
        product = 1
        for level in self.levels:
            basis.append(product)
            product *= level
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.shape[1] != len(self.levels):
            raise ValueError(
                f"FSQ expected {len(self.levels)} latent channels, got {latents.shape[1]}"
            )

        levels = self.levels_tensor.view(1, -1, 1, 1).to(latents.device)
        bounded = torch.tanh(latents)
        shifted = (bounded + 1.0) * 0.5 * (levels - 1.0)
        rounded = shifted.round()
        quantized = rounded / (levels - 1.0) * 2.0 - 1.0
        quantized = bounded + (quantized - bounded).detach()

        indices = (rounded.long() * self.basis.view(1, -1, 1, 1).to(latents.device)).sum(dim=1)
        return quantized, indices


def psnr_from_l1(l1_loss: torch.Tensor) -> torch.Tensor:
    mse_proxy = l1_loss.clamp_min(1e-8) ** 2
    return -10.0 * torch.log10(mse_proxy)


def edge_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sobel_edges(recon), sobel_edges(target))


def sobel_edges(frames: torch.Tensor) -> torch.Tensor:
    channels = frames.shape[1]
    kernel_x = frames.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = frames.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    grad_x = F.conv2d(frames, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(frames, kernel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
