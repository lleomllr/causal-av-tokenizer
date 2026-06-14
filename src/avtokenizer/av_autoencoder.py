from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from avtokenizer.audio_autoencoder import AudioMelAutoencoder
from avtokenizer.video_autoencoder import VideoAutoencoder


class AudioVisualAutoencoder(nn.Module):
    """Lightweight joint audio-video autoencoder with latent-level fusion."""

    def __init__(
        self,
        *,
        video_latent_channels: int = 128,
        video_bottleneck: str = "fsq",
        video_fsq_levels: tuple[int, ...] = (8,) * 16,
        audio_latent_channels: int = 64,
        audio_bottleneck: str = "ae",
        audio_fsq_levels: tuple[int, ...] = (8,) * 8,
    ) -> None:
        super().__init__()
        self.video_model = VideoAutoencoder(
            latent_channels=video_latent_channels,
            bottleneck=video_bottleneck,
            fsq_levels=video_fsq_levels,
        )
        self.audio_model = AudioMelAutoencoder(
            latent_channels=audio_latent_channels,
            bottleneck=audio_bottleneck,
            fsq_levels=audio_fsq_levels,
        )
        video_channels = self.video_model.latent_channels
        audio_channels = self.audio_model.latent_channels

        self.audio_to_video = nn.Sequential(
            nn.Conv2d(audio_channels, video_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(video_channels, video_channels, kernel_size=3, padding=1),
        )
        self.video_fusion = nn.Sequential(
            nn.Conv2d(video_channels * 2, video_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(video_channels, video_channels, kernel_size=3, padding=1),
        )
        self.video_to_audio = nn.Sequential(
            nn.Conv2d(video_channels, audio_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(audio_channels, audio_channels, kernel_size=3, padding=1),
        )
        self.audio_fusion = nn.Sequential(
            nn.Conv2d(audio_channels * 2, audio_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(audio_channels, audio_channels, kernel_size=3, padding=1),
        )
        zero_last_conv(self.video_fusion)
        zero_last_conv(self.audio_fusion)

    def forward(
        self,
        video: torch.Tensor,
        audio_mel: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_frames, channels, height, width = video.shape
        frames = video.reshape(batch_size * num_frames, channels, height, width)
        mel = frame_aligned_mel_to_image(audio_mel)

        video_latents, video_aux = self.video_model.encode(frames)
        audio_latents, audio_aux = self.audio_model.encode(mel)
        _, video_latent_channels, video_latent_h, video_latent_w = video_latents.shape

        video_latents_bt = video_latents.reshape(
            batch_size,
            num_frames,
            video_latent_channels,
            video_latent_h,
            video_latent_w,
        )

        audio_condition = self.audio_to_video(audio_latents)
        audio_condition = F.interpolate(
            audio_condition,
            size=(video_latent_h, video_latent_w),
            mode="bilinear",
            align_corners=False,
        )
        audio_condition = audio_condition[:, None].expand(-1, num_frames, -1, -1, -1)
        audio_condition = audio_condition.reshape_as(video_latents)

        fused_video_latents = video_latents + self.video_fusion(
            torch.cat([video_latents, audio_condition], dim=1)
        )
        recon_frames = self.video_model.decoder(fused_video_latents)
        recon_video = recon_frames.reshape(batch_size, num_frames, channels, height, width)

        video_summary = video_latents_bt.mean(dim=1)
        video_condition = self.video_to_audio(video_summary)
        video_condition = F.interpolate(
            video_condition,
            size=audio_latents.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_audio_latents = audio_latents + self.audio_fusion(
            torch.cat([audio_latents, video_condition], dim=1)
        )
        recon_mel = self.audio_model.decoder(fused_audio_latents)
        recon_mel = recon_mel[..., : mel.shape[-2], : mel.shape[-1]]
        recon_audio_mel = mel_image_to_frame_aligned(recon_mel, audio_mel.shape)

        output = {
            "video": recon_video,
            "audio_mel": recon_audio_mel,
        }
        if return_aux:
            output["video_latents"] = video_latents
            output["audio_latents"] = audio_latents
            if "code_indices" in video_aux:
                output["video_code_indices"] = video_aux["code_indices"]
            if "code_indices" in audio_aux:
                output["audio_code_indices"] = audio_aux["code_indices"]
        return output


def frame_aligned_mel_to_image(audio_mel: torch.Tensor) -> torch.Tensor:
    batch_size, num_frames, n_mels, steps_per_frame = audio_mel.shape
    return audio_mel.permute(0, 2, 1, 3).reshape(
        batch_size,
        1,
        n_mels,
        num_frames * steps_per_frame,
    )


def mel_image_to_frame_aligned(mel: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    batch_size, num_frames, n_mels, steps_per_frame = target_shape
    return mel.reshape(batch_size, n_mels, num_frames, steps_per_frame).permute(0, 2, 1, 3)


def zero_last_conv(module: nn.Sequential) -> None:
    for layer in reversed(module):
        if isinstance(layer, nn.Conv2d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return
