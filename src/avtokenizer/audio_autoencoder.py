from __future__ import annotations
import torch
from torch import nn
from avtokenizer.video_autoencoder import FSQ


class AudioMelAutoencoder(nn.Module):
    def __init__(
        self,
        latent_channels: int = 64,
        bottleneck: str = "ae",
        fsq_levels: tuple[int, ...] = (8, 8, 8, 8, 8, 8, 8, 8),
    ) -> None:
        super().__init__()
        if bottleneck not in {"ae", "fsq"}:
            raise ValueError(f"Unsupported audio bottleneck: {bottleneck}")

        self.bottleneck = bottleneck
        self.latent_channels = len(fsq_levels) if bottleneck == "fsq" else latent_channels
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.to_latent = nn.Conv2d(128, self.latent_channels, kernel_size=1)
        self.quantizer = FSQ(fsq_levels) if bottleneck == "fsq" else None
        self.decoder = nn.Sequential(
            UpBlock(self.latent_channels, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Softplus(),
        )

    def encode(self, mel: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latents = self.to_latent(self.encoder(mel))
        aux: dict[str, torch.Tensor] = {}
        if self.quantizer is not None:
            latents, indices = self.quantizer(latents)
            aux["code_indices"] = indices
        return latents, aux

    def forward(
        self,
        mel: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latents, aux = self.encode(mel)
        recon = self.decoder(latents)
        recon = recon[..., : mel.shape[-2], : mel.shape[-1]]
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
