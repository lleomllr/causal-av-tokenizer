# causal-av-tokenizer

Minimal starting point for a lightweight causal audio-visual codec project.

## Visual batch check

Use this script to save a QA image for 2-3 synchronized audio-video windows:

```bash
python scripts/visualize_batch.py \
  --root data/raw \
  --clip-seconds 2.0 \
  --fps 12 \
  --image-size 128 \
  --batch-size 3 \
  --output outputs/av_batch_preview.png
```

You can also use a manifest:

```bash
python scripts/visualize_batch.py \
  --manifest data/manifest.csv \
  --split train \
  --batch-size 3 \
  --output outputs/av_batch_preview.png
```

The generated PNG shows, for each sample:

- selected video frames from the sampled window;
- the matching frame-aligned log-mel spectrogram;
- cyan vertical lines marking the mel positions corresponding to the displayed frames.

The structural alignment is correct when the script prints `alignment structure: OK`.
The semantic alignment should be checked visually: sound events should appear in the same temporal region as the corresponding visual events.

## Inspect shapes

```bash
python scripts/inspect_dataset.py \
  --root data/raw \
  --clip-seconds 2.0 \
  --fps 12 \
  --image-size 128 \
  --batch-size 2
```

## Video tokenizer baselines

Create a manifest with durations and train/val splits:

```bash
python scripts/make_manifest.py \
  --root data/raw \
  --output data/manifest.csv \
  --window-level \
  --clip-seconds 2.0 \
  --stride-seconds 0.5 \
  --val-ratio 0.2
```

Train a continuous spatial autoencoder with edge loss:

```bash
python scripts/train_video_autoencoder.py \
  --manifest data/manifest.csv \
  --bottleneck ae \
  --edge-loss-weight 0.1 \
  --epochs 5 \
  --batch-size 1 \
  --max-train-batches 50 \
  --max-val-batches 50 \
  --output-dir outputs/video_ae_edge
```

Train a spatial VAE tokenizer:

```bash
python scripts/train_video_autoencoder.py \
  --manifest data/manifest.csv \
  --bottleneck vae \
  --edge-loss-weight 0.1 \
  --kl-weight 0.0001 \
  --epochs 5 \
  --batch-size 1 \
  --max-train-batches 50 \
  --max-val-batches 50 \
  --output-dir outputs/video_vae
```

Train a spatial FSQ tokenizer:

```bash
python scripts/train_video_autoencoder.py \
  --manifest data/manifest.csv \
  --bottleneck fsq \
  --fsq-levels 8,8,8,8 \
  --edge-loss-weight 0.1 \
  --epochs 5 \
  --batch-size 1 \
  --max-train-batches 50 \
  --max-val-batches 50 \
  --output-dir outputs/video_fsq
```

Visualize reconstructions from any checkpoint:

```bash
python scripts/visualize_reconstructions.py \
  --checkpoint outputs/video_fsq/best.pt \
  --manifest data/manifest.csv \
  --split val \
  --output outputs/video_fsq/reconstructions_eval.png
```

Evaluate several checkpoints on the same validation windows:

```bash
python scripts/evaluate_checkpoints.py \
  --manifest data/manifest.csv \
  --split val \
  --max-batches 50 \
  --output outputs/tokenizer_eval.csv \
  --checkpoints \
  video_fsq_8ch=outputs/video_fsq_8ch_window_split/best.pt \
  video_fsq_16ch=outputs/video_fsq_16ch_window_split/best.pt \
  video_ae_64ch=outputs/video_ae_64ch_window_split/best.pt
```

## Current results

All models below use `128x128` frames, `2s` windows, `12 fps`, a spatial `8x8` latent grid, and a window-level validation split.

| model | bottleneck | latent shape | discrete | val L1 | val PSNR |
| --- | --- | --- | --- | ---: | ---: |
| video_fsq_8ch | FSQ | 8x8x8 | yes | 0.1110 | 19.73 |
| video_fsq_16ch | FSQ | 16x8x8 | yes | 0.0783 | 23.00 |
| video_ae_64ch | AE | 64x8x8 | no | 0.1563 | 18.38 |

The FSQ 16-channel tokenizer currently gives the best reconstruction-quality/compression tradeoff among these small baselines.

## Audio branch

Train a log-mel audio autoencoder without decoding video frames:

```bash
python scripts/train_audio_autoencoder.py \
  --manifest data/manifest.csv \
  --bottleneck ae \
  --latent-channels 64 \
  --epochs 5 \
  --batch-size 4 \
  --max-train-batches 100 \
  --max-val-batches 50 \
  --output-dir outputs/audio_mel_ae
```

Train a discrete FSQ audio tokenizer:

```bash
python scripts/train_audio_autoencoder.py \
  --manifest data/manifest.csv \
  --bottleneck fsq \
  --fsq-levels 8,8,8,8,8,8,8,8 \
  --epochs 5 \
  --batch-size 4 \
  --max-train-batches 100 \
  --max-val-batches 50 \
  --output-dir outputs/audio_mel_fsq
```

The audio model reconstructs full frame-aligned log-mel windows shaped `1x80x96` for the default `2s`, `12 fps`, and `4` mel steps per video frame. Its latent grid is `C x 10 x 12`, preserving more temporal resolution than the current video tokenizer.

## Joint audio-video codec

Train a first joint codec by initializing from the best video and audio baselines:

```bash
python scripts/train_av_autoencoder.py \
  --manifest data/manifest.csv \
  --video-checkpoint outputs/video_fsq_16ch_window_split/best.pt \
  --audio-checkpoint outputs/audio_mel_ae/best.pt \
  --epochs 5 \
  --batch-size 1 \
  --max-train-batches 100 \
  --max-val-batches 50 \
  --output-dir outputs/av_joint
```

The joint model keeps the video tokenizer discrete (`FSQ 16x8x8`) and the audio tokenizer continuous (`AE 64x10x12`) for stability. Audio and video interact through residual latent fusion layers initialized to zero, so the model starts from the pretrained unimodal reconstructions and learns cross-modal corrections.

For a cleaner fusion-only ablation, freeze the pretrained audio/video backbones and train only the cross-modal latent fusion layers:

```bash
python scripts/train_av_autoencoder.py \
  --manifest data/manifest.csv \
  --video-checkpoint outputs/video_fsq_16ch_window_split/best.pt \
  --audio-checkpoint outputs/audio_mel_ae/best.pt \
  --freeze-backbones \
  --epochs 5 \
  --batch-size 1 \
  --max-train-batches 100 \
  --max-val-batches 50 \
  --output-dir outputs/av_joint_fusion_only
```

This makes the comparison explicit: full joint fine-tuning updates the unimodal tokenizers plus fusion layers, while fusion-only keeps the learned tokenizers fixed and measures what cross-modal residual correction can add.

Train an explicit cross-modal reconstruction objective by also reconstructing video from audio-only inputs and audio from video-only inputs:

```bash
python scripts/train_av_autoencoder.py \
  --manifest data/manifest.csv \
  --video-checkpoint outputs/video_fsq_16ch_window_split/best.pt \
  --audio-checkpoint outputs/audio_mel_ae/best.pt \
  --freeze-backbones \
  --cross-loss-weight 0.5 \
  --epochs 5 \
  --batch-size 1 \
  --max-train-batches 100 \
  --max-val-batches 50 \
  --output-dir outputs/av_joint_latent_mask_fusion_only
```

This keeps the normal reconstruction objective and adds `video_from_audio` plus `audio_from_video` losses. Missing modalities are represented with learned latent mask tokens, so the available modality is encoded normally and the absent modality is replaced after encoding rather than by out-of-distribution pixel or mel zeros. With frozen backbones, the experiment isolates whether the latent fusion layers can learn cross-modal prediction paths.

Evaluate whether the model really uses cross-modal information by corrupting one modality at validation time:

```bash
python scripts/evaluate_av_ablations.py \
  --manifest data/manifest.csv \
  --split val \
  --batch-size 4 \
  --max-batches 50 \
  --output outputs/av_ablation_eval.csv \
  --checkpoints \
  av_joint=outputs/av_joint/best.pt \
  av_joint_fusion_only=outputs/av_joint_fusion_only/best.pt \
  av_joint_latent_mask_fusion_only=outputs/av_joint_latent_mask_fusion_only/best.pt
```

The CSV compares `normal`, `audio_zeroed`, `video_zeroed`, `audio_masked`, `video_masked`, `audio_shuffled`, and `video_shuffled`, including deltas from the normal condition. Larger degradation in video metrics when audio is corrupted, or in audio metrics when video is corrupted, is evidence that latent fusion is using the other modality rather than behaving as two independent autoencoders. The masked conditions use the learned latent mask tokens; the zeroed conditions are kept as an out-of-distribution baseline.

## AV fusion ablations

All models below are evaluated on the same validation windows with `128x128` video, `2s` clips, `12 fps`, video latents `16x8x8`, and audio latents `64x10x12`.

| model | normal loss | video masked delta video L1 | audio masked delta audio L1 | note |
| --- | ---: | ---: | ---: | --- |
| av_joint | 0.2416 | +0.2246 | +0.2682 | full joint fine-tuning |
| av_joint_fusion_only | 0.2499 | +0.2071 | +0.3527 | frozen unimodal tokenizers, fusion layers only |
| av_joint_latent_mask_fusion_only | 0.2552 | +0.1458 | +0.2361 | fusion-only with explicit cross-modal loss and learned latent mask tokens |

Learned latent mask tokens improve missing-modality reconstruction while preserving normal reconstruction quality reasonably well. The effect is strongest for video reconstruction from masked video inputs, where the degradation drops from `+0.2071` to `+0.1458` under the same frozen-backbone setting.
